from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import io
import base64
import json
from datetime import datetime
import os
import sys
import uuid
from typing import Optional, Dict, Any

# Add your project path
sys.path.append('/home/vulcan/Abhay/Projects/strideaide-ml')

# Import your existing components
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Import chatbot components
from langchain_core.tools import tool
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StrideAide ML - Skin Cancer Detection & Appointment Booking", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
feedback_counter = Counter('feedback_total', 'Total feedback received', ['type'])
prediction_time = Histogram('prediction_duration_seconds', 'Time spent on predictions')
appointment_counter = Counter('appointments_total', 'Total appointments booked')

# ==================== SKIN CANCER DETECTION MODELS ====================

class EfficientNetB0SkinCancer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB0SkinCancer, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = self.backbone.classifier
        
    def forward(self, x):
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.classifier(features)

# ==================== CHATBOT COMPONENTS ====================

# Initialize Supabase
def init_supabase():
    """Initialize Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
    
    supabase: Client = create_client(url, key)
    return supabase

# Google Calendar integration
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_google_calendar_service():
    """Get authenticated Google Calendar service using service account"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service-account-key.json',
            scopes=SCOPES
        )
        return build('calendar', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Error creating calendar service: {e}")
        return None

def create_calendar_event(doctor_info, appointment_details, patient_info):
    """Create a Google Calendar event for the appointment"""
    try:
        service = get_google_calendar_service()
        
        if not service:
            return {
                'success': False,
                'error': 'Could not initialize Google Calendar service',
                'message': 'Calendar service unavailable'
            }
        
        from datetime import datetime, timedelta
        start_datetime = f"{appointment_details['appointment_date']}T{appointment_details['appointment_time']}:00"
        end_time = (datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S') + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
        
        event = {
            'summary': f'Patient Appointment: {patient_info["patient_name"]}',
            'description': f"""
🏥 Medical Appointment Details:

👤 Patient: {patient_info["patient_name"]}
📧 Email: {patient_info["patient_email"]}
📞 Phone: {patient_info["patient_phone"]}
🩺 Reason: {appointment_details.get("reason", "Skin cancer consultation")}
💼 Type: {appointment_details.get("consultation_type", "in_person")}
👨‍⚕️ Doctor: {doctor_info["name"]}
🏥 Clinic: {doctor_info["clinic_name"]}
💰 Fee: ₹{doctor_info["consultation_fee"]}

Please arrive 15 minutes early for your appointment.
            """.strip(),
            'start': {
                'dateTime': start_datetime,
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Asia/Kolkata',
            },
            'attendees': [
                {
                    'email': patient_info["patient_email"],
                    'displayName': patient_info["patient_name"],
                    'responseStatus': 'needsAction'
                }
            ],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 30},
                ],
            },
            'colorId': '2',
            'location': doctor_info.get('clinic_name', 'Clinic'),
        }
        
        calendar_id = doctor_info.get('calendar_id', 'primary')
        
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates='all'
        ).execute()
        
        return {
            'success': True,
            'event_id': created_event['id'],
            'event_link': created_event.get('htmlLink', ''),
            'message': 'Calendar event created successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to create calendar event: {str(e)}'
        }

# Import your existing tools
@tool
def find_available_doctors(specialty: str = "dermatology", date: str = "", location: str = "Bangalore") -> str:
    """
    Find available doctors based on appointments table and working hours.
    
    Args:
        specialty (str): Doctor specialty (default: 'dermatology')
        date (str): Preferred date in YYYY-MM-DD format. Leave empty for next 7 days
        location (str): Doctor location (default: 'Bangalore')
    
    Returns:
        str: JSON string with available doctors and their next available slots
    """
    try:
        supabase = init_supabase()
        
        doctors_query = supabase.table('doctors').select('*').eq('is_available', True)
        
        if specialty:
            doctors_query = doctors_query.ilike('specialty', f'%{specialty}%')
        if location:
            doctors_query = doctors_query.ilike('location', f'%{location}%')
            
        doctors = doctors_query.execute()
        
        if not doctors.data:
            return json.dumps({
                'status': 'error',
                'message': 'No doctors found matching criteria'
            })
        
        available_doctors = []
        # Use current date if date is empty or None
        search_start_date = date if date and date.strip() else datetime.now().strftime('%Y-%m-%d')
        
        for doctor in doctors.data:
            appointments = supabase.table('appointments').select('*').eq(
                'doctor_id', doctor['id']
            ).gte('appointment_date', search_start_date).execute()
            
            next_slots = find_next_available_slots(doctor, appointments.data, search_start_date)
            
            if next_slots:
                available_doctors.append({
                    'doctor_id': doctor['id'],
                    'name': doctor['name'],
                    'specialty': doctor['specialty'],
                    'location': doctor['location'],
                    'clinic_name': doctor['clinic_name'],
                    'bio': doctor['bio'],
                    'years_experience': doctor['years_experience'],
                    'rating': float(doctor['rating']) if doctor['rating'] else 0,
                    'consultation_fee': float(doctor['consultation_fee']) if doctor['consultation_fee'] else 0,
                    'qualifications': doctor['qualifications'],
                    'languages': doctor['languages'],
                    'next_available_slots': next_slots,
                    'contact': {
                        'phone': doctor['phone'],
                        'email': doctor['email']
                    }
                })
        
        available_doctors.sort(key=lambda x: (-x['rating'], x['next_available_slots'][0]['datetime']))
        
        return json.dumps({
            'status': 'success',
            'available_doctors': available_doctors,
            'total_found': len(available_doctors),
            'search_criteria': {
                'specialty': specialty,
                'date': search_start_date,
                'location': location
            }
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f'Error finding doctors: {str(e)}'
        })

def find_next_available_slots(doctor: Dict, existing_appointments: list, start_date: str, num_slots: int = 3) -> list:
    """Find next available appointment slots within working hours"""
    from datetime import datetime, timedelta
    
    working_hours = doctor.get('working_hours', {
        'start': 9, 
        'end': 17, 
        'slot_duration': 30, 
        'days': [1, 2, 3, 4, 5]
    })
    
    booked_times = []
    for app in existing_appointments:
        if app['status'] in ['scheduled', 'confirmed']:
            try:
                app_date_str = app['appointment_date']
                if '+' in app_date_str:
                    app_time = datetime.fromisoformat(app_date_str)
                else:
                    app_time = datetime.fromisoformat(app_date_str.replace('Z', ''))
                
                if app_time.tzinfo is not None:
                    app_time = app_time.replace(tzinfo=None)
                booked_times.append(app_time)
            except Exception as e:
                print(f"Warning: Could not parse appointment date {app_date_str}: {e}")
                continue
    
    available_slots = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    for day_offset in range(14):
        check_date = current_date + timedelta(days=day_offset)
        
        weekday = check_date.weekday() + 1
        if weekday not in working_hours.get('days', [1, 2, 3, 4, 5]):
            continue
        
        start_hour = working_hours.get('start', 9)
        end_hour = working_hours.get('end', 17)
        slot_duration = working_hours.get('slot_duration', 30)
        
        start_hour = max(0, min(23, start_hour))
        end_hour = max(0, min(23, end_hour))
        
        if check_date.date() == datetime.now().date():
            current_hour = max(start_hour, min(23, datetime.now().hour + 1))
            start_time = check_date.replace(hour=current_hour, minute=0, second=0, microsecond=0)
        else:
            start_time = check_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        end_time = check_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        if start_time >= end_time:
            continue
        
        current_slot = start_time
        while current_slot < end_time and len(available_slots) < num_slots:
            is_available = True
            for booked_time in booked_times:
                time_diff = abs((current_slot - booked_time).total_seconds())
                if time_diff < slot_duration * 60:
                    is_available = False
                    break
            
            if is_available:
                available_slots.append({
                    'date': current_slot.strftime('%Y-%m-%d'),
                    'time': current_slot.strftime('%H:%M'),
                    'datetime': current_slot.isoformat(),
                    'day_of_week': current_slot.strftime('%A'),
                    'formatted': current_slot.strftime('%A, %B %d, %Y at %I:%M %p'),
                    'ist_time': (current_slot + timedelta(hours=5, minutes=30)).strftime('%I:%M %p IST')
                })
            
            current_slot += timedelta(minutes=slot_duration)
        
        if len(available_slots) >= num_slots:
            break
    
    return available_slots

@tool
def book_appointment(
    doctor_id: int, 
    patient_name: str, 
    patient_email: str, 
    patient_phone: str, 
    appointment_date: str, 
    appointment_time: str, 
    reason: str = "General consultation", 
    consultation_type: str = "in_person"
) -> str:
    """
    Book an appointment with a doctor and update the appointments table.
    
    Args:
        doctor_id (int): ID of the doctor
        patient_name (str): Full name of the patient
        patient_email (str): Patient's email address
        patient_phone (str): Patient's phone number
        appointment_date (str): Date in YYYY-MM-DD format
        appointment_time (str): Time in HH:MM format
        reason (str): Reason for the appointment (default: "General consultation")
        consultation_type (str): Type of consultation - 'in_person', 'video_call', or 'phone' (default: "in_person")
    
    Returns:
        str: JSON string with booking confirmation
    """
    try:
        supabase = init_supabase()
        
        doctor = supabase.table('doctors').select('*').eq('id', doctor_id).eq('is_available', True).execute()
        
        if not doctor.data:
            return json.dumps({
                'status': 'error',
                'message': 'Doctor not found or not available'
            })
        
        doctor_info = doctor.data[0]
        
        appointment_datetime = f"{appointment_date} {appointment_time}:00"
        appointment_dt = datetime.strptime(appointment_datetime, '%Y-%m-%d %H:%M:%S')
        
        working_hours = doctor_info.get('working_hours', {
            'start': 9, 'end': 17, 'days': [1, 2, 3, 4, 5]
        })
        
        hour = appointment_dt.hour
        weekday = appointment_dt.weekday() + 1
        
        if (hour < working_hours.get('start', 9) or 
            hour >= working_hours.get('end', 17) or 
            weekday not in working_hours.get('days', [1, 2, 3, 4, 5])):
            return json.dumps({
                'status': 'error',
                'message': 'Appointment time is outside doctor\'s working hours'
            })
        
        existing = supabase.table('appointments').select('*').eq(
            'doctor_id', doctor_id
        ).eq('appointment_date', appointment_dt.isoformat()).execute()
        
        if existing.data:
            return json.dumps({
                'status': 'error',
                'message': 'This time slot is already booked'
            })
        
        new_appointment = {
            'doctor_id': doctor_id,
            'patient_name': patient_name,
            'patient_email': patient_email,
            'patient_phone': patient_phone,
            'appointment_date': appointment_dt.isoformat(),
            'reason': reason,
            'status': 'scheduled',
            'consultation_type': consultation_type,
            'duration_minutes': 30
        }
        
        result = supabase.table('appointments').insert(new_appointment).execute()
        
        if result.data:
            appointment_id = result.data[0]['id']
            
            # Create Google Calendar event
            calendar_result = create_calendar_event(
                doctor_info=doctor_info,
                appointment_details={
                    'appointment_date': appointment_date,
                    'appointment_time': appointment_time,
                    'reason': reason,
                    'consultation_type': consultation_type
                },
                patient_info={
                    'patient_name': patient_name,
                    'patient_email': patient_email,
                    'patient_phone': patient_phone
                }
            )
            
            if calendar_result['success']:
                supabase.table('appointments').update({
                    'calendar_event_id': calendar_result['event_id']
                }).eq('id', appointment_id).execute()
            
            return json.dumps({
                'status': 'success',
                'message': 'Appointment booked successfully!',
                'appointment_details': {
                    'appointment_id': appointment_id,
                    'doctor_name': doctor_info['name'],
                    'clinic_name': doctor_info['clinic_name'],
                    'patient_name': patient_name,
                    'appointment_date': appointment_date,
                    'appointment_time': appointment_time,
                    'consultation_type': consultation_type,
                    'reason': reason,
                    'doctor_phone': doctor_info['phone'],
                    'consultation_fee': f"₹{doctor_info['consultation_fee']}"
                },
                'calendar_integration': {
                    'calendar_event_created': calendar_result['success'],
                    'calendar_event_id': calendar_result.get('event_id', ''),
                    'calendar_link': calendar_result.get('event_link', ''),
                    'calendar_message': calendar_result['message']
                },
                'instructions': f"Please arrive 15 minutes early. Contact clinic at {doctor_info['phone']} for any changes."
            }, indent=2)
        else:
            return json.dumps({
                'status': 'error',
                'message': 'Failed to book appointment'
            })
            
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f'Error booking appointment: {str(e)}'
        })

@tool  
def get_appointment_details(appointment_id: int = 0, patient_name: str = "") -> str:
    """
    Get appointment details by ID or patient name.
    
    Args:
        appointment_id (int): Specific appointment ID (default: 0 means not provided)
        patient_name (str): Patient name to search appointments (default: empty string)
    
    Returns:
        str: JSON string with appointment details
    """
    try:
        supabase = init_supabase()
        
        if appointment_id > 0:
            result = supabase.table('appointments').select('''
                *, doctors(name, clinic_name, phone, specialty)
            ''').eq('id', appointment_id).execute()
        elif patient_name and patient_name.strip():
            result = supabase.table('appointments').select('''
                *, doctors(name, clinic_name, phone, specialty)
            ''').ilike('patient_name', f'%{patient_name}%').execute()
        else:
            return json.dumps({
                'status': 'error',
                'message': 'Please provide either appointment_id or patient_name'
            })
        
        if not result.data:
            return json.dumps({
                'status': 'error',
                'message': 'No appointments found'
            })
        
        appointments = []
        for apt in result.data:
            appointments.append({
                'appointment_id': apt['id'],
                'patient_name': apt['patient_name'],
                'doctor_name': apt['doctors']['name'],
                'clinic_name': apt['doctors']['clinic_name'],
                'specialty': apt['doctors']['specialty'],
                'appointment_date': apt['appointment_date'],
                'reason': apt['reason'],
                'status': apt['status'],
                'consultation_type': apt['consultation_type'],
                'doctor_phone': apt['doctors']['phone']
            })
        
        return json.dumps({
            'status': 'success',
            'appointments': appointments,
            'total_found': len(appointments)
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': f'Error retrieving appointments: {str(e)}'
        })

# Chatbot setup
tools = [find_available_doctors, book_appointment, get_appointment_details]

def llm_init_model():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key,
        temperature=0.7
    )
    return llm

llm = llm_init_model()
llm_with_tools = llm.bind_tools(tools)

# State for chatbot
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Memory for conversations
memory = MemorySaver()

# Build graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)

# Global variables
model = None
device = torch.device('cpu')
feedback_file = "/home/vulcan/Abhay/Projects/strideaide-ml/backend/data/feedback.txt"

# Image preprocessing functions
def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_bytes):
    """Preprocess uploaded image for inference"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_np = np.array(image)
    
    transform = get_inference_transform()
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor, image_np

def numpy_to_base64(image_np):
    """Convert numpy image to base64 string"""
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        model_path = "/home/vulcan/Abhay/Projects/strideaide-ml/Cancer_Classification/backend/models/best_model.pth"
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return
        
        model = EfficientNetB0SkinCancer(num_classes=2, pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from checkpoint with AUC: {checkpoint.get('best_auc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded from state dict")
        
        model.eval()
        print(f"Model loaded successfully on {device}!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.post("/predict")
async def predict_skin_cancer(file: UploadFile = File(...)):
    """Make prediction on uploaded skin lesion image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        with prediction_time.time():
            # Read and preprocess image
            image_bytes = await file.read()
            image_tensor, original_image = preprocess_image(image_bytes)
            image_tensor = image_tensor.to(device)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence = probabilities[0, 1].item()
                prediction = 1 if confidence > 0.5 else 0
            
            # Generate feature-based visualization
            try:
                feature_maps = []
                def hook_fn(module, input, output):
                    feature_maps.append(output.detach())
                
                last_conv_layer = None
                for name, module in model.backbone.features.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv_layer = module
                
                if last_conv_layer is not None:
                    hook = last_conv_layer.register_forward_hook(hook_fn)
                    
                    model.eval()
                    with torch.no_grad():
                        output = model(image_tensor)
                    
                    if feature_maps:
                        features = feature_maps[0]
                        cam = torch.mean(features, dim=1).squeeze().numpy()
                        
                        orig_h, orig_w = original_image.shape[:2]
                        cam_resized = cv2.resize(cam, (orig_w, orig_h))
                        
                        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                        
                        threshold = 0.5
                        cam_norm[cam_norm < threshold] = 0
                        
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        
                        original_norm = original_image.astype(np.float32) / 255.0
                        alpha = 0.3
                        overlayed = alpha * heatmap + (1 - alpha) * original_norm
                        overlayed = np.clip(overlayed, 0, 1)
                        
                        gradcam_b64 = numpy_to_base64(overlayed)
                        heatmap_b64 = numpy_to_base64(heatmap)
                        
                    else:
                        gradcam_b64 = None
                        heatmap_b64 = None
                    
                    hook.remove()
                    
                else:
                    gradcam_b64 = None
                    heatmap_b64 = None
                
            except Exception as e:
                print(f"Visualization error: {e}")
                gradcam_b64 = None
                heatmap_b64 = None
            
            prediction_counter.inc()
            
            # Generate unique prediction ID
            prediction_id = str(uuid.uuid4())
            
            result = {
                "prediction_id": prediction_id,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "label": "Cancerous" if prediction == 1 else "Non-cancerous",
                "risk_level": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
                "gradcam_overlay": gradcam_b64,
                "heatmap": heatmap_b64,
                "timestamp": datetime.now().isoformat(),
                "recommendation": {
                    "message": "We recommend consulting with a dermatologist for further evaluation." if prediction == 1 else "The prediction suggests non-cancerous, but we still recommend professional consultation for peace of mind.",
                    "urgency": "High" if prediction == 1 else "Medium",
                    "next_steps": [
                        "Consult with a dermatologist",
                        "Book an appointment through our system",
                        "Bring this analysis to your doctor"
                    ]
                }
            }
            
            return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/chat")
async def chat_with_assistant(
    request: Request,
    message: str = Form(...),
    thread_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    prediction_context: Optional[str] = Form(None)
):
    """Chat with the appointment booking assistant"""
    try:
        # Create enhanced message with context
        enhanced_message = message
        if prediction_context:
            try:
                context_data = json.loads(prediction_context)
                if context_data.get('prediction') == 1:  # Cancerous
                    enhanced_message = f"""
Based on my skin cancer prediction analysis that showed a {context_data.get('label', 'cancerous')} result with {context_data.get('confidence', 0):.2%} confidence, I need medical consultation.

{message}

Please prioritize dermatology specialists who can handle potential skin cancer cases.
"""
            except json.JSONDecodeError:
                pass
        
        # Configure thread
        config = {'configurable': {'thread_id': thread_id}}
        
        # Check if this is a new conversation by trying to get existing state
        try:
            existing_state = graph.get_state(config)
            is_new_conversation = len(existing_state.values.get('messages', [])) == 0
        except:
            is_new_conversation = True
        
        # Add system message for new conversations
        messages_to_send = []
        if is_new_conversation:
            from langchain_core.messages import SystemMessage
            system_message = SystemMessage(content="""You are a helpful medical appointment booking assistant for StrideAide ML. 
            
Your main functions are:
1. Find available doctors based on specialty, location, and date preferences
2. Book appointments with doctors
3. Retrieve appointment details
4. Provide information about doctors and their availability

When helping users:
- Always ask for necessary information (name, email, phone) before booking
- Provide clear information about available doctors and time slots
- Confirm appointment details before finalizing
- Be polite and professional
- Focus on dermatology specialists for skin cancer related consultations

You have access to tools to search for doctors, book appointments, and get appointment details. Use them to help users effectively.""")
            messages_to_send.append(system_message)
        
        # Create human message
        human_message = HumanMessage(content=enhanced_message)
        messages_to_send.append(human_message)
        
        # Invoke graph
        state = graph.invoke({"messages": messages_to_send}, config=config)
        
        # Get the last assistant message
        last_message = state["messages"][-1]
        
        # Extract content from the message
        if hasattr(last_message, 'content'):
            assistant_message = last_message.content
        else:
            assistant_message = str(last_message)
        
        return JSONResponse(content={
            "response": assistant_message,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    prediction_id: str = Form(...),
    feedback_type: str = Form(...),
    original_prediction: int = Form(...),
    confidence: float = Form(...)
):
    """Submit feedback for model predictions"""
    try:
        feedback_data = {
            "prediction_id": prediction_id,
            "feedback_type": feedback_type,
            "original_prediction": original_prediction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        feedback_counter.labels(type=feedback_type).inc()
        
        return JSONResponse(content={"status": "success", "message": "Feedback recorded"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "services": {
            "skin_cancer_prediction": model is not None,
            "appointment_booking": True,
            "google_calendar": get_google_calendar_service() is not None
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)