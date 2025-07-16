import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { z } from "zod";
import { createClient } from '@supabase/supabase-js'

interface Env {
    SUPABASE_URL: string;
    SUPABASE_ANON_KEY: string;
}

export class MyMCP extends McpAgent {
    server = new McpServer({
        name: "Appointify",
        version: "1.0.0",
    });

    constructor(ctx: DurableObjectState, env: Env) {
        super(ctx, env);
    }

    async init() {
        this.server.tool(
            "find_available_doctors",
            "Find available doctors based on specialty, location, and date preferences. Returns sorted list by rating and next available slot.",
            {
                specialty: z.string().optional().default("dermatology").describe("Doctor specialty (e.g., 'Cardiology', 'Dermatology')"),
                date: z.string().optional().describe("Preferred date in YYYY-MM-DD format. Leave empty for next 7 days"),
                location: z.string().optional().default("Bangalore").describe("Doctor location")
            },
            async (input: { specialty?: string; date?: string; location?: string }) => {
                try {
                    const env = this.env as Env;
                    const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_ANON_KEY);

                    let doctorsQuery = supabase.from('doctors').select('*').eq('is_available', true);
                    
                    if (input.specialty) {
                        doctorsQuery = doctorsQuery.ilike('specialty', `%${input.specialty}%`);
                    }
                    if (input.location) {
                        doctorsQuery = doctorsQuery.ilike('location', `%${input.location}%`);
                    }
                    
                    const { data: doctors, error: doctorsError } = await doctorsQuery;
                    
                    if (doctorsError) {
                        throw new Error(`Supabase query failed: ${doctorsError.message}`);
                    }
                    
                    if (!doctors || doctors.length === 0) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'No doctors found matching criteria'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    const availableDoctors = [];
                    const searchStartDate = input.date && input.date.trim() 
                        ? input.date 
                        : new Date().toISOString().split('T')[0];
                    
                    // Get all doctor IDs for batch query
                    const doctorIds = doctors.map((doc: any) => doc.id);
                    
                    // Fetch all appointments for all doctors in one query
                    const { data: allAppointments, error: appointmentsError } = await supabase
                        .from('appointments')
                        .select('doctor_id, appointment_date, status')
                        .in('doctor_id', doctorIds)
                        .gte('appointment_date', new Date(searchStartDate).toISOString());
                    
                    if (appointmentsError) {
                        throw new Error(`Failed to fetch appointments: ${appointmentsError.message}`);
                    }
                    
                    // Group appointments by doctor_id
                    const appointmentsByDoctor: { [key: number]: any[] } = {};
                    (allAppointments || []).forEach((appt: any) => {
                        if (!appointmentsByDoctor[appt.doctor_id]) {
                            appointmentsByDoctor[appt.doctor_id] = [];
                        }
                        appointmentsByDoctor[appt.doctor_id].push(appt);
                    });
                    
                    for (const doctor of doctors) {
                        const doctorAppointments = appointmentsByDoctor[doctor.id] || [];
                        const nextSlots = this.findNextAvailableSlots(doctor, doctorAppointments, searchStartDate);
                        
                        if (nextSlots.length > 0) {
                            availableDoctors.push({
                                doctor_id: doctor.id,
                                name: doctor.name,
                                specialty: doctor.specialty,
                                location: doctor.location,
                                clinic_name: doctor.clinic_name,
                                bio: doctor.bio,
                                years_experience: doctor.years_experience,
                                rating: parseFloat(String(doctor.rating)) || 0,
                                consultation_fee: parseFloat(String(doctor.consultation_fee)) || 0,
                                qualifications: doctor.qualifications,
                                languages: doctor.languages,
                                next_available_slots: nextSlots,
                                contact: {
                                    phone: doctor.phone,
                                    email: doctor.email
                                }
                            });
                        }
                    }
                    
                    // Sort by rating (desc) then by next available slot time (asc)
                    availableDoctors.sort((a, b) => {
                        if (b.rating !== a.rating) return b.rating - a.rating;
                        return a.next_available_slots[0].datetime.localeCompare(b.next_available_slots[0].datetime);
                    });
                    
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'success',
                                    available_doctors: availableDoctors,
                                    total_found: availableDoctors.length,
                                    search_criteria: {
                                        specialty: input.specialty || 'dermatology',
                                        date: searchStartDate,
                                        location: input.location || 'Bangalore'
                                    }
                                }, null, 2)
                            }
                        ]
                    };
                    
                } catch (error) {
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'error',
                                    message: `Error finding doctors: ${error instanceof Error ? error.message : String(error)}`
                                }, null, 2)
                            }
                        ]
                    };
                }
            }
        );

        this.server.tool(
            "book_appointment",
            "Book an appointment with a doctor and update the appointments table.",
            {
                doctor_id: z.number().describe("ID of the doctor"),
                patient_name: z.string().describe("Full name of the patient"),
                patient_email: z.string().email().describe("Patient's email address"),
                patient_phone: z.string().describe("Patient's phone number"),
                appointment_date: z.string().describe("Date in YYYY-MM-DD format"),
                appointment_time: z.string().describe("Time in HH:MM format"),
                reason: z.string().optional().default("General consultation").describe("Reason for the appointment"),
                consultation_type: z.string().optional().default("in_person").describe("Type of consultation - 'in_person', 'video_call', or 'phone'")
            },
            async (input: { 
                doctor_id: number; 
                patient_name: string; 
                patient_email: string; 
                patient_phone: string; 
                appointment_date: string; 
                appointment_time: string; 
                reason?: string; 
                consultation_type?: string; 
            }) => {
                try {
                    const env = this.env as Env;
                    const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_ANON_KEY);

                    // Fetch doctor details
                    const { data: doctor, error: doctorError } = await supabase
                        .from('doctors')
                        .select('*')
                        .eq('id', input.doctor_id)
                        .eq('is_available', true)
                        .single();
                    
                    if (doctorError || !doctor) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'Doctor not found or not available'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    // Parse appointment datetime
                    const appointmentDatetime = `${input.appointment_date} ${input.appointment_time}:00`;
                    const appointmentDt = new Date(appointmentDatetime);
                    
                    // Validate working hours
                    const workingHours = doctor.working_hours || {
                        start: 9,
                        end: 17,
                        days: [1, 2, 3, 4, 5]
                    };
                    
                    const hour = appointmentDt.getHours();
                    const weekday = appointmentDt.getDay() === 0 ? 7 : appointmentDt.getDay(); // Convert Sunday from 0 to 7
                    
                    if (hour < workingHours.start || 
                        hour >= workingHours.end || 
                        !workingHours.days.includes(weekday)) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'Appointment time is outside doctor\'s working hours'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    // Check if slot is already booked
                    const { data: existingAppointments, error: existingError } = await supabase
                        .from('appointments')
                        .select('*')
                        .eq('doctor_id', input.doctor_id)
                        .eq('appointment_date', appointmentDt.toISOString());
                    
                    if (existingError) {
                        throw new Error(`Failed to check existing appointments: ${existingError.message}`);
                    }
                    
                    if (existingAppointments && existingAppointments.length > 0) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'This time slot is already booked'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    // Create new appointment
                    const newAppointment = {
                        doctor_id: input.doctor_id,
                        patient_name: input.patient_name,
                        patient_email: input.patient_email,
                        patient_phone: input.patient_phone,
                        appointment_date: appointmentDt.toISOString(),
                        reason: input.reason || "General consultation",
                        status: 'scheduled',
                        consultation_type: input.consultation_type || "in_person",
                        duration_minutes: 30
                    };
                    
                    const { data: result, error: insertError } = await supabase
                        .from('appointments')
                        .insert(newAppointment)
                        .select()
                        .single();
                    
                    if (insertError || !result) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'Failed to book appointment'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    const appointmentId = result.id;
                    
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'success',
                                    message: 'Appointment booked successfully!',
                                    appointment_details: {
                                        appointment_id: appointmentId,
                                        doctor_name: doctor.name,
                                        clinic_name: doctor.clinic_name,
                                        patient_name: input.patient_name,
                                        appointment_date: input.appointment_date,
                                        appointment_time: input.appointment_time,
                                        consultation_type: input.consultation_type || "in_person",
                                        reason: input.reason || "General consultation",
                                        doctor_phone: doctor.phone,
                                        consultation_fee: `₹${doctor.consultation_fee}`
                                    },
                                    instructions: `Please arrive 15 minutes early. Contact clinic at ${doctor.phone} for any changes.`
                                }, null, 2)
                            }
                        ]
                    };
                    
                } catch (error) {
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'error',
                                    message: `Error booking appointment: ${error instanceof Error ? error.message : String(error)}`
                                }, null, 2)
                            }
                        ]
                    };
                }
            }
        );

        this.server.tool(
            "get_appointment_details",
            "Get appointment details by ID or patient name.",
            {
                appointment_id: z.number().optional().default(0).describe("Specific appointment ID (0 means not provided)"),
                patient_name: z.string().optional().default("").describe("Patient name to search appointments")
            },
            async (input: { appointment_id?: number; patient_name?: string }) => {
                try {
                    const env = this.env as Env;
                    const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_ANON_KEY);
                    
                    let query;
                    
                    if (input.appointment_id && input.appointment_id > 0) {
                        query = supabase
                            .from('appointments')
                            .select(`
                                *,
                                doctors (
                                    name,
                                    clinic_name,
                                    phone,
                                    specialty
                                )
                            `)
                            .eq('id', input.appointment_id);
                    } else if (input.patient_name && input.patient_name.trim()) {
                        query = supabase
                            .from('appointments')
                            .select(`
                                *,
                                doctors (
                                    name,
                                    clinic_name,
                                    phone,
                                    specialty
                                )
                            `)
                            .ilike('patient_name', `%${input.patient_name.trim()}%`);
                    } else {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'Please provide either appointment_id or patient_name'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    const { data: result, error } = await query;
                    
                    if (error) {
                        throw new Error(`Database query failed: ${error.message}`);
                    }
                    
                    if (!result || result.length === 0) {
                        return {
                            content: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        status: 'error',
                                        message: 'No appointments found'
                                    }, null, 2)
                                }
                            ]
                        };
                    }
                    
                    const appointments = result.map((apt: any) => ({
                        appointment_id: apt.id,
                        patient_name: apt.patient_name,
                        patient_email: apt.patient_email,
                        patient_phone: apt.patient_phone,
                        doctor_name: apt.doctors?.name || 'Unknown',
                        clinic_name: apt.doctors?.clinic_name || 'Unknown',
                        specialty: apt.doctors?.specialty || 'Unknown',
                        appointment_date: apt.appointment_date,
                        reason: apt.reason,
                        status: apt.status,
                        consultation_type: apt.consultation_type,
                        duration_minutes: apt.duration_minutes,
                        doctor_phone: apt.doctors?.phone || 'Unknown',
                        calendar_event_id: apt.calendar_event_id
                    }));
                    
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'success',
                                    appointments: appointments,
                                    total_found: appointments.length
                                }, null, 2)
                            }
                        ]
                    };
                    
                } catch (error) {
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    status: 'error',
                                    message: `Error retrieving appointments: ${error instanceof Error ? error.message : String(error)}`
                                }, null, 2)
                            }
                        ]
                    };
                }
            }
        );
    }

    private findNextAvailableSlots(doctor: any, existingAppointments: any[], startDate: string, numSlots: number = 3): any[] {
        const workingHours = doctor.working_hours || {
            start: 9,
            end: 17,
            slot_duration: 30,
            days: [1, 2, 3, 4, 5]
        };
        
        const bookedTimes: Date[] = [];
        for (const app of existingAppointments) {
            if (app.status === 'scheduled' || app.status === 'confirmed') {
                try {
                    let appDateStr = app.appointment_date;
                    if (appDateStr.includes('+')) {
                        appDateStr = appDateStr.split('+')[0];
                    }
                    if (appDateStr.endsWith('Z')) {
                        appDateStr = appDateStr.replace('Z', '');
                    }
                    
                    const appTime = new Date(appDateStr);
                    bookedTimes.push(appTime);
                } catch (error) {
                    console.warn(`Could not parse appointment date ${app.appointment_date}: ${error}`);
                }
            }
        }
        
        const availableSlots = [];
        const currentDate = new Date(startDate + 'T00:00:00');
        
        for (let dayOffset = 0; dayOffset < 14 && availableSlots.length < numSlots; dayOffset++) {
            const checkDate = new Date(currentDate);
            checkDate.setDate(currentDate.getDate() + dayOffset);
            
            const weekday = checkDate.getDay() === 0 ? 7 : checkDate.getDay(); // Convert Sunday from 0 to 7
            if (!workingHours.days.includes(weekday)) {
                continue;
            }
            
            const startHour = Math.max(0, Math.min(23, workingHours.start));
            const endHour = Math.max(0, Math.min(23, workingHours.end));
            
            let startTime: Date;
            if (checkDate.toDateString() === new Date().toDateString()) {
                const currentHour = Math.max(startHour, Math.min(23, new Date().getHours() + 1));
                startTime = new Date(checkDate);
                startTime.setHours(currentHour, 0, 0, 0);
            } else {
                startTime = new Date(checkDate);
                startTime.setHours(startHour, 0, 0, 0);
            }
            
            const endTime = new Date(checkDate);
            endTime.setHours(endHour, 0, 0, 0);
            
            if (startTime >= endTime) {
                continue;
            }
            
            const currentSlot = new Date(startTime);
            while (currentSlot < endTime && availableSlots.length < numSlots) {
                let isAvailable = true;
                for (const bookedTime of bookedTimes) {
                    const timeDiff = Math.abs(currentSlot.getTime() - bookedTime.getTime());
                    if (timeDiff < workingHours.slot_duration * 60 * 1000) {
                        isAvailable = false;
                        break;
                    }
                }
                
                if (isAvailable) {
                    availableSlots.push({
                        date: currentSlot.toISOString().split('T')[0],
                        time: currentSlot.toTimeString().split(' ')[0].substring(0, 5),
                        datetime: currentSlot.toISOString(),
                        day_of_week: currentSlot.toLocaleDateString('en-US', { weekday: 'long' }),
                        formatted: currentSlot.toLocaleDateString('en-US', { 
                            weekday: 'long', 
                            year: 'numeric', 
                            month: 'long', 
                            day: 'numeric' 
                        }) + ' at ' + currentSlot.toLocaleTimeString('en-US', { 
                            hour: 'numeric', 
                            minute: '2-digit', 
                            hour12: true 
                        }),
                        ist_time: new Date(currentSlot.getTime() + 5.5 * 60 * 60 * 1000)
                            .toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }) + ' IST'
                    });
                }
                
                currentSlot.setMinutes(currentSlot.getMinutes() + workingHours.slot_duration);
            }
        }
        
        return availableSlots;
    }
}

export default {
    fetch(request: Request, env: Env, ctx: ExecutionContext) {
        const url = new URL(request.url);

        if (url.pathname === "/sse") {
            return MyMCP.serveSSE("/sse").fetch(request, env, ctx);
        }

        if (url.pathname === "/mcp") {
            return MyMCP.serve("/mcp").fetch(request, env, ctx);
        }

        return new Response("Not found", { status: 404 });
    },
};