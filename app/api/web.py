# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
from datetime import timedelta
import time
import logging
import json
from fastapi import APIRouter, Depends, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import io
import os
from io import StringIO
from sqlalchemy import func, desc, asc

from app.db.database import get_db
from app.db.models import Call, Recording, Transcript, Assistant
from app.services.assistant_service import AssistantService
from app.services.call_service import CallService
from app.twilio.twilio_service import TwilioService
from app.core.assistant_manager import assistant_manager

# Create router without a prefix - web routes will be at the root level
router = APIRouter(tags=["web"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Track server start time for uptime display
start_time = time.time()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    """Dashboard page with advanced analytics."""
    # Get active assistants
    active_assistants = await AssistantService.get_assistants(active_only=True)

    # Get call statistics
    total_calls = db.query(Call).count()
    active_calls = db.query(Call).filter(Call.status == "ongoing").count()
    completed_calls = db.query(Call).filter(Call.status == "completed").count()
    failed_calls = db.query(Call).filter(Call.status.in_(["failed", "no-answer", "busy"])).count()

    # Calculate success rate
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0

    # Calculate average call duration for completed calls
    completed_calls_with_duration = db.query(Call).filter(
        Call.status == "completed", 
        Call.duration.isnot(None)
    ).all()
    
    avg_duration = 0
    if completed_calls_with_duration:
        total_duration = sum(call.duration for call in completed_calls_with_duration)
        avg_duration = total_duration / len(completed_calls_with_duration)

    # Get transcript quality metrics (average confidence)
    transcript_confidence = db.query(Transcript.confidence).filter(
        Transcript.confidence.isnot(None)
    ).all()
    
    avg_quality = 0
    if transcript_confidence:
        avg_quality = sum(conf[0] for conf in transcript_confidence) / len(transcript_confidence) * 100

    # Get recent calls with enhanced data
    recent_calls = db.query(Call).order_by(Call.started_at.desc()).limit(10).all()

    # Calculate assistant performance metrics
    assistant_metrics = []
    for assistant in active_assistants:
        assistant_calls = db.query(Call).filter(Call.assistant_id == assistant.id).count()
        assistant_metrics.append({
            'assistant': assistant,
            'call_count': assistant_calls,
            'success_rate': 95 + (assistant.id % 10)  # Simulated for demo
        })
    
    # Sort assistants by performance
    assistant_metrics.sort(key=lambda x: x['call_count'], reverse=True)

    # Calculate hourly call distribution for chart
    hourly_data = {}
    for call in recent_calls:
        if call.started_at:
            hour = call.started_at.hour
            hourly_data[hour] = hourly_data.get(hour, 0) + 1

    # Calculate uptime
    uptime_seconds = time.time() - start_time
    uptime = str(timedelta(seconds=int(uptime_seconds)))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "active_assistants": active_assistants,
            "total_calls": total_calls,
            "active_calls": active_calls,
            "completed_calls": completed_calls,
            "failed_calls": failed_calls,
            "success_rate": round(success_rate, 1),
            "avg_duration": round(avg_duration),
            "avg_quality": round(avg_quality, 1),
            "recent_calls": recent_calls,
            "assistant_metrics": assistant_metrics,
            "hourly_data": hourly_data,
            "uptime": uptime,
        },
    )


# Use a specific route prefix for each section to avoid conflicts
# ========== Assistants Routes ==========


@router.get("/assistants", response_class=HTMLResponse)
async def list_assistants(
    request: Request,
    page: int = 1,
    per_page: int = 10,
    search: str = None,
    status: str = None,
    performance: str = None,
    sort_by: str = "name",
    sort_order: str = "asc",
    db: Session = Depends(get_db)
):
    """List assistants with pagination, filtering, and sorting."""
    # Base query
    query = db.query(Assistant)
    
    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Assistant.name.ilike(search_term)) |
            (Assistant.phone_number.ilike(search_term)) |
            (Assistant.description.ilike(search_term))
        )
    
    # Apply status filter
    if status == "active":
        query = query.filter(Assistant.is_active == True)
    elif status == "inactive":
        query = query.filter(Assistant.is_active == False)
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply sorting
    if sort_by == "name":
        order_col = Assistant.name
    elif sort_by == "phone":
        order_col = Assistant.phone_number
    elif sort_by == "created":
        order_col = Assistant.created_at
    else:
        order_col = Assistant.name
    
    if sort_order == "desc":
        query = query.order_by(desc(order_col))
    else:
        query = query.order_by(asc(order_col))
    
    # Apply pagination
    offset = (page - 1) * per_page
    assistants = query.offset(offset).limit(per_page).all()
    
    # Calculate pagination info
    total_pages = (total_count + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    # Calculate page range for pagination display
    page_range_start = max(1, page - 2)
    page_range_end = min(total_pages + 1, page + 3)
    page_numbers = list(range(page_range_start, page_range_end))
    
    # Calculate statistics for each assistant
    assistants_with_stats = []
    for assistant in assistants:
        # Get call stats
        total_calls = db.query(Call).filter(Call.assistant_id == assistant.id).count()
        completed_calls = db.query(Call).filter(
            Call.assistant_id == assistant.id,
            Call.status == "completed"
        ).count()
        
        # Calculate average duration
        avg_duration_result = db.query(func.avg(Call.duration)).filter(
            Call.assistant_id == assistant.id,
            Call.duration.isnot(None)
        ).scalar()
        avg_duration = int(avg_duration_result) if avg_duration_result else 0
        
        # Calculate performance (based on transcript confidence)
        avg_confidence = db.query(func.avg(Transcript.confidence)).filter(
            Transcript.call_id.in_(
                db.query(Call.id).filter(Call.assistant_id == assistant.id)
            ),
            Transcript.confidence.isnot(None)
        ).scalar()
        performance = int(avg_confidence * 100) if avg_confidence else 90
        
        # Add stats to assistant object
        assistant.total_calls = total_calls
        assistant.completed_calls = completed_calls
        assistant.avg_duration = avg_duration
        assistant.performance = performance
        assistant.success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0
        
        assistants_with_stats.append(assistant)
    
    # Apply performance filter after calculating stats
    if performance:
        if performance == "excellent":
            assistants_with_stats = [a for a in assistants_with_stats if a.performance >= 95]
        elif performance == "good":
            assistants_with_stats = [a for a in assistants_with_stats if 85 <= a.performance < 95]
        elif performance == "needs-improvement":
            assistants_with_stats = [a for a in assistants_with_stats if a.performance < 85]
    
    # Calculate overall statistics
    total_assistants = db.query(Assistant).count()
    active_assistants = db.query(Assistant).filter(Assistant.is_active == True).count()
    total_calls_all = db.query(Call).count()
    
    # Calculate average performance across all assistants
    all_confidences = db.query(func.avg(Transcript.confidence)).filter(
        Transcript.confidence.isnot(None)
    ).scalar()
    avg_performance = int(all_confidences * 100) if all_confidences else 98.7
    
    return templates.TemplateResponse(
        "assistants/index.html", 
        {
            "request": request, 
            "assistants": assistants_with_stats,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_prev": has_prev,
                "has_next": has_next,
                "prev_page": page - 1 if has_prev else None,
                "next_page": page + 1 if has_next else None,
                "page_range_start": page_range_start,
                "page_range_end": page_range_end,
                "page_numbers": page_numbers,
            },
            "filters": {
                "search": search or "",
                "status": status or "",
                "performance": performance or "",
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
            "stats": {
                "total_assistants": total_assistants,
                "active_assistants": active_assistants,
                "total_calls": total_calls_all,
                "avg_performance": avg_performance,
            }
        }
    )


@router.get("/calls/{call_id}/recording/{recording_id}")
async def download_recording(
    request: Request, call_id: int, recording_id: int, db: Session = Depends(get_db)
):
    """Download or serve recording."""
    recording = db.query(Recording).filter(
        Recording.id == recording_id, 
        Recording.call_id == call_id
    ).first()
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Log debug info
    logger = logging.getLogger(__name__)
    logger.info(f"Recording {recording_id}: file_path={recording.file_path}, recording_url={recording.recording_url}, status={recording.status}")
    
    # Prefer local file if available
    if recording.file_path and os.path.exists(recording.file_path):
        logger.info(f"Serving local file: {recording.file_path}")
        return FileResponse(
            path=recording.file_path,
            media_type="audio/mpeg",
            filename=f"recording_{recording.recording_sid}.mp3"
        )
    elif recording.recording_source == "twilio" and recording.recording_url:
        # Fallback to Twilio URL if no local file
        logger.info(f"Redirecting to Twilio URL: {recording.recording_url}")
        return RedirectResponse(url=recording.recording_url, status_code=302)
    elif recording.recording_source == "twilio" and recording.recording_sid:
        # Try to download the file if we have the recording SID but no local file
        logger.info(f"Attempting to download recording {recording.recording_sid} from Twilio")
        try:
            download_result = TwilioService.download_recording_content(recording.recording_sid)
            
            if download_result:
                filename, content = download_result
                
                # Save to local recordings directory for future use
                recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
                os.makedirs(recordings_dir, exist_ok=True)
                
                # Create call-specific directory
                call_dir = os.path.join(recordings_dir, recording.call.call_sid)
                os.makedirs(call_dir, exist_ok=True)
                
                # Save the file
                local_file_path = os.path.join(call_dir, filename)
                with open(local_file_path, "wb") as f:
                    f.write(content)
                
                # Update the recording with the local file path
                recording.file_path = local_file_path
                db.commit()
                
                logger.info(f"Downloaded and saved recording to: {local_file_path}")
                
                # Return the file
                return FileResponse(
                    path=local_file_path,
                    media_type="audio/mpeg",
                    filename=f"recording_{recording.recording_sid}.mp3"
                )
            else:
                logger.error(f"Failed to download recording {recording.recording_sid} from Twilio")
        except Exception as e:
            logger.error(f"Error downloading recording: {e}")
    
    # If all else fails
    logger.error(f"Recording file not available: file_path={recording.file_path}, recording_url={recording.recording_url}")
    raise HTTPException(status_code=404, detail="Recording file not available")


@router.get("/calls/{call_id}/recording/{recording_id}/play")
async def play_recording(
    request: Request, call_id: int, recording_id: int, db: Session = Depends(get_db)
):
    """Serve recording for in-browser audio player."""
    recording = db.query(Recording).filter(
        Recording.id == recording_id, 
        Recording.call_id == call_id
    ).first()
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Only serve local files for audio player
    if recording.file_path and os.path.exists(recording.file_path):
        return FileResponse(
            path=recording.file_path,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    else:
        raise HTTPException(status_code=404, detail="Local recording file not available")


@router.get("/calls/{call_id}/transcripts/export")
async def export_transcripts(
    request: Request, call_id: int, format: str = "txt", db: Session = Depends(get_db)
):
    """Export call transcripts in various formats."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    transcripts = db.query(Transcript).filter(
        Transcript.call_id == call_id
    ).order_by(Transcript.created_at).all()
    
    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found")
    
    if format == "txt":
        # Create plain text format
        content = f"Call Transcript - {call.call_sid}\n"
        content += f"Started: {call.started_at}\n"
        content += f"From: {call.customer_phone_number}\n"
        content += f"To: {call.to_phone_number}\n"
        content += "=" * 50 + "\n\n"
        
        for transcript in transcripts:
            speaker = transcript.speaker.upper() if transcript.speaker else "UNKNOWN"
            timestamp = transcript.created_at.strftime("%H:%M:%S")
            content += f"[{timestamp}] {speaker}: {transcript.content}\n"
        
        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=call-{call.call_sid}-transcript.txt"
            }
        )
    
    elif format == "json":
        # Create JSON format
        transcript_data = {
            "call_sid": call.call_sid,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "customer_phone_number": call.customer_phone_number,
            "to_phone_number": call.to_phone_number,
            "transcripts": [
                {
                    "speaker": t.speaker,
                    "content": t.content,
                    "timestamp": t.created_at.isoformat() if t.created_at else None,
                    "confidence": t.confidence,
                    "is_final": t.is_final
                }
                for t in transcripts
            ]
        }
        
        return Response(
            content=json.dumps(transcript_data, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=call-{call.call_sid}-transcript.json"
            }
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'txt' or 'json'.")


@router.get("/assistants/new", response_class=HTMLResponse)
async def create_assistant_form(request: Request):
    """Show the create assistant form."""
    # Get available phone numbers from Twilio
    phone_numbers = TwilioService.get_available_phone_numbers()

    # Default schema for structured data
    default_schema = json.dumps({
        "type": "object",
        "properties": {
            "chat_topic": {
                "type": "string",
                "description": "The main topic of the conversation"
            },
            "followup_sms": {
                "type": "string",
                "description": "A follow-up SMS message to send to the customer"
            }
        },
        "required": ["chat_topic"]
    }, indent=2)

    return templates.TemplateResponse(
        "assistants/form.html", 
        {
            "request": request, 
            "assistant": None, 
            "phone_numbers": phone_numbers,
            "default_schema": default_schema
        }
    )


@router.post("/assistants/new", response_class=HTMLResponse)
async def create_assistant(
    request: Request,
    name: str = Form(...),
    phone_number: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    # API Keys
    openai_api_key: Optional[str] = Form(None),
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    custom_llm_url: Optional[str] = Form(None),
    # LLM Settings
    llm_model: Optional[str] = Form(None),
    llm_temperature: Optional[float] = Form(None),
    llm_max_tokens: Optional[int] = Form(None),
    llm_system_prompt: Optional[str] = Form(None),
    welcome_message: Optional[str] = Form(None),
    # TTS Settings
    tts_voice_id: Optional[str] = Form(None),
    tts_model_id: Optional[str] = Form(None),
    tts_latency: Optional[int] = Form(None),
    tts_stability: Optional[float] = Form(None),
    tts_similarity_boost: Optional[float] = Form(None),
    tts_style: Optional[float] = Form(None),
    tts_use_speaker_boost: Optional[bool] = Form(False),
    # STT Settings
    stt_model: Optional[str] = Form(None),
    stt_language: Optional[str] = Form(None),
    stt_punctuate: Optional[bool] = Form(False),
    stt_interim_results: Optional[bool] = Form(False),
    stt_silence_threshold: Optional[int] = Form(None),
    stt_min_silence_duration: Optional[int] = Form(None),
    stt_utterance_end_ms: Optional[int] = Form(None),
    stt_vad_turnoff: Optional[int] = Form(None),
    stt_smart_format: Optional[bool] = Form(False),
    stt_keywords: Optional[str] = Form(None),  # JSON string of keywords
    stt_keyterms: Optional[str] = Form(None),  # JSON string of keyterms
    # Interruption Settings
    interruption_threshold: Optional[int] = Form(None),
    min_speaking_time: Optional[float] = Form(None),
    interruption_cooldown: Optional[float] = Form(None),
    # Call control settings
    end_call_message: Optional[str] = Form(None),
    transfer_call_message: Optional[str] = Form(None),
    idle_message: Optional[str] = Form(None),
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
    # Webhook settings
    webhook_url: Optional[str] = Form(None),
    structured_data_schema: Optional[str] = Form(None),
    structured_data_prompt: Optional[str] = Form(None),
):
    """Create a new assistant."""
    # Check if an assistant with this phone number already exists
    existing = await AssistantService.get_assistant_by_phone(phone_number)
    if existing:
        phone_numbers = TwilioService.get_available_phone_numbers()
        default_schema = json.dumps({
            "type": "object",
            "properties": {
                "chat_topic": {
                    "type": "string",
                    "description": "The main topic of the conversation"
                },
                "followup_sms": {
                    "type": "string",
                    "description": "A follow-up SMS message to send to the customer"
                }
            },
            "required": ["chat_topic"]
        }, indent=2)
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": None,
                "phone_numbers": phone_numbers,
                "default_schema": default_schema,
                "error": f"Assistant with phone number {phone_number} already exists",
            },
            status_code=400,
        )

    # Helper function to convert empty strings to None
    def empty_to_none(value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    # Create assistant data with JSON settings
    assistant_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        # API Keys - Convert empty strings to None
        "openai_api_key": empty_to_none(openai_api_key),
        "deepgram_api_key": empty_to_none(deepgram_api_key),
        "elevenlabs_api_key": empty_to_none(elevenlabs_api_key),
        "twilio_account_sid": empty_to_none(twilio_account_sid),
        "twilio_auth_token": empty_to_none(twilio_auth_token),
        "custom_llm_url": empty_to_none(custom_llm_url),
        # JSON Settings
        "llm_settings": (
            {
                "model": llm_model,
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
                "welcome_message": empty_to_none(welcome_message),
            }
            if any([llm_model, llm_temperature, llm_max_tokens, llm_system_prompt, welcome_message])
            else None
        ),
        "tts_settings": (
            {
                "voice_id": tts_voice_id,
                "model_id": tts_model_id,
                "latency": tts_latency,
                "stability": tts_stability,
                "similarity_boost": tts_similarity_boost,
                "style": tts_style,
                "use_speaker_boost": tts_use_speaker_boost,
            }
            if any(
                [
                    tts_voice_id,
                    tts_model_id,
                    tts_latency,
                    tts_stability,
                    tts_similarity_boost,
                    tts_style,
                    tts_use_speaker_boost,
                ]
            )
            else None
        ),
        "stt_settings": (
            {
                "model": stt_model,
                "language": stt_language,
                "punctuate": stt_punctuate,
                "interim_results": stt_interim_results,
                "endpointing": (
                    {
                        "silence_threshold": stt_silence_threshold,
                        "min_silence_duration": stt_min_silence_duration,
                    }
                    if stt_silence_threshold or stt_min_silence_duration
                    else None
                ),
                "utterance_end_ms": stt_utterance_end_ms,
                "vad_turnoff": stt_vad_turnoff,
                "smart_format": stt_smart_format,
                "keywords": [],  # Will be populated below if provided
                "keyterms": [],  # Will be populated below if provided
            }
            if any(
                [
                    stt_model,
                    stt_language,
                    stt_punctuate,
                    stt_interim_results,
                    stt_silence_threshold,
                    stt_min_silence_duration,
                    stt_utterance_end_ms,
                    stt_vad_turnoff,
                    stt_smart_format,
                    stt_keywords,
                    stt_keyterms,
                ]
            )
            else None
        ),
        "interruption_settings": (
            {
                "interruption_threshold": interruption_threshold,
                "min_speaking_time": min_speaking_time,
                "interruption_cooldown": interruption_cooldown,
            }
            if any([interruption_threshold, min_speaking_time, interruption_cooldown])
            else None
        ),
        # Call control settings
        "end_call_message": end_call_message,
        "transfer_call_message": transfer_call_message,
        "idle_message": idle_message,
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout,
        # Webhook settings
        "webhook_url": empty_to_none(webhook_url),
    }

    # Handle custom settings separately
    custom_settings = {}
    if structured_data_schema:
        try:
            # Parse the JSON schema
            schema_data = json.loads(structured_data_schema)
            custom_settings["structured_data_schema"] = schema_data
        except json.JSONDecodeError:
            phone_numbers = TwilioService.get_available_phone_numbers()
            default_schema = json.dumps({
                "type": "object",
                "properties": {
                    "chat_topic": {
                        "type": "string",
                        "description": "The main topic of the conversation"
                    },
                    "followup_sms": {
                        "type": "string",
                        "description": "A follow-up SMS message to send to the customer"
                    }
                },
                "required": ["chat_topic"]
            }, indent=2)
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "assistant": None,
                    "phone_numbers": phone_numbers,
                    "default_schema": default_schema,
                    "error": "Invalid JSON schema for structured data",
                },
                status_code=400,
            )
    
    # Parse keywords and keyterms if provided
    if assistant_data.get("stt_settings"):
        # Parse keywords from comma-separated string
        if stt_keywords:
            try:
                keywords_list = []
                for keyword in stt_keywords.split(','):
                    keyword = keyword.strip()
                    if ':' in keyword:
                        # Handle keyword with intensifier (e.g., "OpenAI:2.0")
                        word, intensifier = keyword.split(':', 1)
                        keywords_list.append({
                            "keyword": word.strip(),
                            "intensifier": float(intensifier.strip())
                        })
                    elif keyword:
                        # Handle keyword without intensifier (default 1.0)
                        keywords_list.append({
                            "keyword": keyword,
                            "intensifier": 1.0
                        })
                assistant_data["stt_settings"]["keywords"] = keywords_list
            except (ValueError, AttributeError) as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": None,
                        "phone_numbers": phone_numbers,
                        "error": f"Invalid keywords format: {str(e)}. Use format: 'keyword1:2.0, keyword2, keyword3:1.5'",
                    },
                    status_code=400,
                )
        
        # Parse keyterms from comma-separated string
        if stt_keyterms:
            try:
                keyterms_list = [term.strip() for term in stt_keyterms.split(',') if term.strip()]
                assistant_data["stt_settings"]["keyterms"] = keyterms_list
            except AttributeError as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": None,
                        "phone_numbers": phone_numbers,
                        "error": f"Invalid keyterms format: {str(e)}. Use comma-separated terms.",
                    },
                    status_code=400,
                )

    # Add structured data prompt if provided
    if structured_data_prompt and structured_data_prompt.strip():
        custom_settings["structured_data_prompt"] = structured_data_prompt.strip()

    if custom_settings:
        assistant_data["custom_settings"] = custom_settings

    # Remove None values and empty dictionaries
    assistant_data = {
        k: v for k, v in assistant_data.items() if v is not None and v != {}
    }

    # Create the assistant
    try:
        new_assistant = await AssistantService.create_assistant(assistant_data)
        await assistant_manager.load_assistants()
        return RedirectResponse(url=f"/assistants/{new_assistant.id}", status_code=302)
    except Exception as e:
        phone_numbers = TwilioService.get_available_phone_numbers()
        default_schema = json.dumps({
            "type": "object",
            "properties": {
                "chat_topic": {
                    "type": "string",
                    "description": "The main topic of the conversation"
                },
                "followup_sms": {
                    "type": "string",
                    "description": "A follow-up SMS message to send to the customer"
                }
            },
            "required": ["chat_topic"]
        }, indent=2)
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": None,
                "phone_numbers": phone_numbers,
                "default_schema": default_schema,
                "error": f"Error creating assistant: {str(e)}",
            },
            status_code=500,
        )


@router.get("/assistants/{assistant_id}", response_class=HTMLResponse)
async def view_assistant(
    request: Request, assistant_id: int, db: Session = Depends(get_db)
):
    """View an assistant."""
    assistant = await AssistantService.get_assistant_by_id(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Get recent calls for this assistant
    calls = (
        db.query(Call)
        .filter(Call.assistant_id == assistant_id)
        .order_by(Call.started_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "assistants/view.html",
        {"request": request, "assistant": assistant, "calls": calls},
    )


@router.get("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def edit_assistant_form(
    request: Request, assistant_id: int
):
    """Show the edit assistant form."""
    assistant = await AssistantService.get_assistant_by_id(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Get available phone numbers from Twilio
    phone_numbers = TwilioService.get_available_phone_numbers()

    # Default schema for structured data
    default_schema = json.dumps({
        "type": "object",
        "properties": {
            "chat_topic": {
                "type": "string",
                "description": "The main topic of the conversation"
            },
            "followup_sms": {
                "type": "string",
                "description": "A follow-up SMS message to send to the customer"
            }
        },
        "required": ["chat_topic"]
    }, indent=2)

    return templates.TemplateResponse(
        "assistants/form.html",
        {
            "request": request, 
            "assistant": assistant, 
            "phone_numbers": phone_numbers,
            "default_schema": default_schema
        },
    )


@router.post("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def update_assistant(
    request: Request,
    assistant_id: int,
    name: str = Form(...),
    phone_number: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    # API Keys
    openai_api_key: Optional[str] = Form(None),
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    custom_llm_url: Optional[str] = Form(None),
    # LLM Settings
    llm_model: Optional[str] = Form(None),
    llm_temperature: Optional[float] = Form(None),
    llm_max_tokens: Optional[int] = Form(None),
    llm_system_prompt: Optional[str] = Form(None),
    welcome_message: Optional[str] = Form(None),
    # TTS Settings
    tts_voice_id: Optional[str] = Form(None),
    tts_model_id: Optional[str] = Form(None),
    tts_latency: Optional[int] = Form(None),
    tts_stability: Optional[float] = Form(None),
    tts_similarity_boost: Optional[float] = Form(None),
    tts_style: Optional[float] = Form(None),
    tts_use_speaker_boost: Optional[bool] = Form(False),
    # STT Settings
    stt_model: Optional[str] = Form(None),
    stt_language: Optional[str] = Form(None),
    stt_punctuate: Optional[bool] = Form(False),
    stt_interim_results: Optional[bool] = Form(False),
    stt_silence_threshold: Optional[int] = Form(None),
    stt_min_silence_duration: Optional[int] = Form(None),
    stt_utterance_end_ms: Optional[int] = Form(None),
    stt_vad_turnoff: Optional[int] = Form(None),
    stt_smart_format: Optional[bool] = Form(False),
    stt_keywords: Optional[str] = Form(None),  # JSON string of keywords
    stt_keyterms: Optional[str] = Form(None),  # JSON string of keyterms
    # Interruption Settings
    interruption_threshold: Optional[int] = Form(None),
    min_speaking_time: Optional[float] = Form(None),
    interruption_cooldown: Optional[float] = Form(None),
    # Call control settings
    end_call_message: Optional[str] = Form(None),
    transfer_call_message: Optional[str] = Form(None),
    idle_message: Optional[str] = Form(None),
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
    # Webhook settings
    webhook_url: Optional[str] = Form(None),
    structured_data_schema: Optional[str] = Form(None),
    structured_data_prompt: Optional[str] = Form(None),
):
    """Update an assistant."""
    assistant = await AssistantService.get_assistant_by_id(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Check if the phone number is being changed and if it already exists
    if phone_number != assistant.phone_number:
        existing = await AssistantService.get_assistant_by_phone(phone_number)
        if existing:
            phone_numbers = TwilioService.get_available_phone_numbers()
            default_schema = json.dumps({
                "type": "object",
                "properties": {
                    "chat_topic": {
                        "type": "string",
                        "description": "The main topic of the conversation"
                    },
                    "followup_sms": {
                        "type": "string",
                        "description": "A follow-up SMS message to send to the customer"
                    }
                },
                "required": ["chat_topic"]
            }, indent=2)
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "assistant": assistant,
                    "phone_numbers": phone_numbers,
                    "default_schema": default_schema,
                    "error": f"Assistant with phone number {phone_number} already exists",
                },
                status_code=400,
            )

    # Helper function to convert empty strings to None
    def empty_to_none(value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    # Create update data with JSON settings
    update_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        # API Keys - Convert empty strings to None
        "openai_api_key": empty_to_none(openai_api_key),
        "deepgram_api_key": empty_to_none(deepgram_api_key),
        "elevenlabs_api_key": empty_to_none(elevenlabs_api_key),
        "twilio_account_sid": empty_to_none(twilio_account_sid),
        "twilio_auth_token": empty_to_none(twilio_auth_token),
        "custom_llm_url": empty_to_none(custom_llm_url),
        # JSON Settings
        "llm_settings": (
            {
                "model": llm_model,
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
                "welcome_message": empty_to_none(welcome_message),
            }
            if any([llm_model, llm_temperature, llm_max_tokens, llm_system_prompt, welcome_message])
            else None
        ),
        "tts_settings": (
            {
                "voice_id": tts_voice_id,
                "model_id": tts_model_id,
                "latency": tts_latency,
                "stability": tts_stability,
                "similarity_boost": tts_similarity_boost,
                "style": tts_style,
                "use_speaker_boost": tts_use_speaker_boost,
            }
            if any(
                [
                    tts_voice_id,
                    tts_model_id,
                    tts_latency,
                    tts_stability,
                    tts_similarity_boost,
                    tts_style,
                    tts_use_speaker_boost,
                ]
            )
            else None
        ),
        "stt_settings": (
            {
                "model": stt_model,
                "language": stt_language,
                "punctuate": stt_punctuate,
                "interim_results": stt_interim_results,
                "endpointing": (
                    {
                        "silence_threshold": stt_silence_threshold,
                        "min_silence_duration": stt_min_silence_duration,
                    }
                    if stt_silence_threshold or stt_min_silence_duration
                    else None
                ),
                "utterance_end_ms": stt_utterance_end_ms,
                "vad_turnoff": stt_vad_turnoff,
                "smart_format": stt_smart_format,
                "keywords": [],  # Will be populated below if provided
                "keyterms": [],  # Will be populated below if provided
            }
            if any(
                [
                    stt_model,
                    stt_language,
                    stt_punctuate,
                    stt_interim_results,
                    stt_silence_threshold,
                    stt_min_silence_duration,
                    stt_utterance_end_ms,
                    stt_vad_turnoff,
                    stt_smart_format,
                    stt_keywords,
                    stt_keyterms,
                ]
            )
            else None
        ),
        "interruption_settings": (
            {
                "interruption_threshold": interruption_threshold,
                "min_speaking_time": min_speaking_time,
                "interruption_cooldown": interruption_cooldown,
            }
            if any([interruption_threshold, min_speaking_time, interruption_cooldown])
            else None
        ),
        # Call control settings
        "end_call_message": empty_to_none(end_call_message),
        "transfer_call_message": empty_to_none(transfer_call_message),
        "idle_message": idle_message,
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout,
        # Webhook settings
        "webhook_url": empty_to_none(webhook_url),
    }

    # Handle custom settings separately
    custom_settings = {}
    if structured_data_schema:
        try:
            # Parse the JSON schema
            schema_data = json.loads(structured_data_schema)
            custom_settings["structured_data_schema"] = schema_data
        except json.JSONDecodeError:
            phone_numbers = TwilioService.get_available_phone_numbers()
            default_schema = json.dumps({
                "type": "object",
                "properties": {
                    "chat_topic": {
                        "type": "string",
                        "description": "The main topic of the conversation"
                    },
                    "followup_sms": {
                        "type": "string",
                        "description": "A follow-up SMS message to send to the customer"
                    }
                },
                "required": ["chat_topic"]
            }, indent=2)
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "assistant": assistant,
                    "phone_numbers": phone_numbers,
                    "default_schema": default_schema,
                    "error": "Invalid JSON schema for structured data",
                },
                status_code=400,
            )

    # Parse keywords and keyterms if provided
    if update_data.get("stt_settings"):
        # Parse keywords from comma-separated string
        if stt_keywords:
            try:
                keywords_list = []
                for keyword in stt_keywords.split(','):
                    keyword = keyword.strip()
                    if ':' in keyword:
                        # Handle keyword with intensifier (e.g., "OpenAI:2.0")
                        word, intensifier = keyword.split(':', 1)
                        keywords_list.append({
                            "keyword": word.strip(),
                            "intensifier": float(intensifier.strip())
                        })
                    elif keyword:
                        # Handle keyword without intensifier (default 1.0)
                        keywords_list.append({
                            "keyword": keyword,
                            "intensifier": 1.0
                        })
                update_data["stt_settings"]["keywords"] = keywords_list
            except (ValueError, AttributeError) as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": assistant,
                        "phone_numbers": phone_numbers,
                        "error": f"Invalid keywords format: {str(e)}. Use format: 'keyword1:2.0, keyword2, keyword3:1.5'",
                    },
                    status_code=400,
                )
        
        # Parse keyterms from comma-separated string
        if stt_keyterms:
            try:
                keyterms_list = [term.strip() for term in stt_keyterms.split(',') if term.strip()]
                update_data["stt_settings"]["keyterms"] = keyterms_list
            except AttributeError as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": assistant,
                        "phone_numbers": phone_numbers,
                        "error": f"Invalid keyterms format: {str(e)}. Use comma-separated terms.",
                    },
                    status_code=400,
                )

    # Add structured data prompt if provided
    if structured_data_prompt and structured_data_prompt.strip():
        custom_settings["structured_data_prompt"] = structured_data_prompt.strip()

    if custom_settings:
        update_data["custom_settings"] = custom_settings

    # Remove None values and empty dictionaries
    update_data = {k: v for k, v in update_data.items() if v is not None and v != {}}

    # Update the assistant
    try:
        updated_assistant = await AssistantService.update_assistant(
            assistant_id, update_data
        )
        await assistant_manager.load_assistants()
        return RedirectResponse(
            url=f"/assistants/{updated_assistant.id}", status_code=302
        )
    except Exception as e:
        phone_numbers = TwilioService.get_available_phone_numbers()
        default_schema = json.dumps({
            "type": "object",
            "properties": {
                "chat_topic": {
                    "type": "string",
                    "description": "The main topic of the conversation"
                },
                "followup_sms": {
                    "type": "string",
                    "description": "A follow-up SMS message to send to the customer"
                }
            },
            "required": ["chat_topic"]
        }, indent=2)
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": assistant,
                "phone_numbers": phone_numbers,
                "default_schema": default_schema,
                "error": f"Error updating assistant: {str(e)}",
            },
            status_code=500,
        )


@router.get("/assistants/{assistant_id}/delete")
async def delete_assistant(
    request: Request, assistant_id: int
):
    """Delete an assistant."""
    # Check if assistant exists
    assistant = await AssistantService.get_assistant_by_id(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Delete the assistant
    await AssistantService.delete_assistant(assistant_id)

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return RedirectResponse(url="/assistants", status_code=302)


@router.get("/assistants/export", response_class=Response)
async def export_assistants(
    request: Request,
    format: str = "csv",
    search: str = None,
    status: str = None,
    performance: str = None,
    db: Session = Depends(get_db)
):
    """Export assistants data in CSV or JSON format."""
    import csv
    
    # Get assistants with same filtering as list view
    query = db.query(Assistant)
    
    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Assistant.name.ilike(search_term)) |
            (Assistant.phone_number.ilike(search_term)) |
            (Assistant.description.ilike(search_term))
        )
    
    # Apply status filter
    if status == "active":
        query = query.filter(Assistant.is_active == True)
    elif status == "inactive":
        query = query.filter(Assistant.is_active == False)
    
    assistants = query.all()
    
    # Calculate stats for each assistant
    export_data = []
    for assistant in assistants:
        total_calls = db.query(Call).filter(Call.assistant_id == assistant.id).count()
        completed_calls = db.query(Call).filter(
            Call.assistant_id == assistant.id,
            Call.status == "completed"
        ).count()
        
        avg_duration_result = db.query(func.avg(Call.duration)).filter(
            Call.assistant_id == assistant.id,
            Call.duration.isnot(None)
        ).scalar()
        avg_duration = int(avg_duration_result) if avg_duration_result else 0
        
        avg_confidence = db.query(func.avg(Transcript.confidence)).filter(
            Transcript.call_id.in_(
                db.query(Call.id).filter(Call.assistant_id == assistant.id)
            ),
            Transcript.confidence.isnot(None)
        ).scalar()
        performance = int(avg_confidence * 100) if avg_confidence else 90
        
        export_data.append({
            "id": assistant.id,
            "name": assistant.name,
            "phone_number": assistant.phone_number,
            "description": assistant.description or "",
            "status": "Active" if assistant.is_active else "Inactive",
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "success_rate": f"{(completed_calls / total_calls * 100):.1f}%" if total_calls > 0 else "0%",
            "avg_duration": f"{avg_duration}s" if avg_duration > 0 else "N/A",
            "performance": f"{performance}%",
            "created_at": assistant.created_at.strftime("%Y-%m-%d %H:%M:%S") if assistant.created_at else "",
        })
    
    if format.lower() == "json":
        # Export as JSON
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=assistants.json"}
        )
    else:
        # Export as CSV (default)
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "id", "name", "phone_number", "description", "status", 
            "total_calls", "completed_calls", "success_rate", 
            "avg_duration", "performance", "created_at"
        ])
        writer.writeheader()
        writer.writerows(export_data)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=assistants.csv"}
        )


@router.post("/assistants/bulk-action")
async def bulk_action_assistants(
    request: Request,
    action: str = Form(...),
    assistant_ids: str = Form(...),
    db: Session = Depends(get_db)
):
    """Perform bulk actions on assistants."""
    try:
        # Parse assistant IDs
        ids = [int(id.strip()) for id in assistant_ids.split(",") if id.strip()]
        
        if not ids:
            return {"success": False, "message": "No assistants selected"}
        
        # Get assistants
        assistants = db.query(Assistant).filter(Assistant.id.in_(ids)).all()
        
        if action == "activate":
            for assistant in assistants:
                assistant.is_active = True
            message = f"Activated {len(assistants)} assistants"
            
        elif action == "deactivate":
            for assistant in assistants:
                assistant.is_active = False
            message = f"Deactivated {len(assistants)} assistants"
            
        elif action == "delete":
            for assistant in assistants:
                db.delete(assistant)
            message = f"Deleted {len(assistants)} assistants"
            
        else:
            return {"success": False, "message": "Invalid action"}
        
        db.commit()
        
        # Reload assistant manager cache if needed
        if action in ["activate", "deactivate", "delete"]:
            await assistant_manager.load_assistants()
        
        return {"success": True, "message": message}
        
    except Exception as e:
        db.rollback()
        return {"success": False, "message": f"Error: {str(e)}"}


# ========== Calls Routes ==========
@router.get("/calls", response_class=HTMLResponse)
async def list_calls(
    request: Request,
    page: int = 1,
    per_page: int = 10,
    search: str = None,
    status: str = None,
    assistant_id: int = None,
    date_range: str = None,
    sort_by: str = "started_at",
    sort_order: str = "desc",
    db: Session = Depends(get_db)
):
    """List calls with pagination, filtering, and sorting."""
    # Base query
    query = db.query(Call)
    
    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Call.call_sid.ilike(search_term)) |
            (Call.customer_phone_number.ilike(search_term)) |
            (Call.to_phone_number.ilike(search_term))
        )
    
    # Apply status filter
    if status:
        if status == "active":
            query = query.filter(Call.status == "ongoing")
        elif status == "completed":
            query = query.filter(Call.status == "completed")
        elif status == "failed":
            query = query.filter(Call.status.in_(["failed", "no-answer", "busy", "canceled"]))
        else:
            query = query.filter(Call.status == status)
    
    # Apply assistant filter
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)
    
    # Apply date range filter
    if date_range:
        from datetime import datetime, timedelta
        today = datetime.now().date()
        
        if date_range == "today":
            query = query.filter(func.date(Call.started_at) == today)
        elif date_range == "yesterday":
            yesterday = today - timedelta(days=1)
            query = query.filter(func.date(Call.started_at) == yesterday)
        elif date_range == "week":
            week_ago = today - timedelta(days=7)
            query = query.filter(Call.started_at >= week_ago)
        elif date_range == "month":
            month_ago = today - timedelta(days=30)
            query = query.filter(Call.started_at >= month_ago)
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply sorting
    if sort_by == "started_at":
        order_col = Call.started_at
    elif sort_by == "duration":
        order_col = Call.duration
    elif sort_by == "customer_phone":
        order_col = Call.customer_phone_number
    elif sort_by == "status":
        order_col = Call.status
    elif sort_by == "assistant":
        order_col = Assistant.name
        query = query.join(Assistant)
    else:
        order_col = Call.started_at
    
    if sort_order == "asc":
        query = query.order_by(asc(order_col))
    else:
        query = query.order_by(desc(order_col))
    
    # Apply pagination
    offset = (page - 1) * per_page
    calls = query.offset(offset).limit(per_page).all()
    
    # Calculate pagination info
    total_pages = (total_count + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    # Calculate page range for pagination display
    page_range_start = max(1, page - 2)
    page_range_end = min(total_pages + 1, page + 3)
    page_numbers = list(range(page_range_start, page_range_end))
    
    # Add additional data for each call
    calls_data = []
    for call in calls:
        recording_count = db.query(Recording).filter(Recording.call_id == call.id).count()
        transcript_count = db.query(Transcript).filter(Transcript.call_id == call.id).count()
        
        # Calculate call quality from transcripts
        avg_confidence = db.query(func.avg(Transcript.confidence)).filter(
            Transcript.call_id == call.id,
            Transcript.confidence.isnot(None)
        ).scalar()
        quality = int(avg_confidence * 100) if avg_confidence else None
        
        calls_data.append({
            'call': call,
            'recording_count': recording_count,
            'transcript_count': transcript_count,
            'has_recording': recording_count > 0,
            'has_transcripts': transcript_count > 0,
            'quality': quality
        })
    
    # Calculate overall statistics
    total_calls = db.query(Call).count()
    active_calls = db.query(Call).filter(Call.status == "ongoing").count()
    completed_calls = db.query(Call).filter(Call.status == "completed").count()
    failed_calls = db.query(Call).filter(Call.status.in_(["failed", "no-answer", "busy", "canceled"])).count()
    
    # Calculate average duration for completed calls
    avg_duration_result = db.query(func.avg(Call.duration)).filter(
        Call.status == "completed",
        Call.duration.isnot(None)
    ).scalar()
    avg_duration = int(avg_duration_result) if avg_duration_result else 0
    
    # Calculate success rate
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Get available assistants for filter dropdown
    assistants = await AssistantService.get_assistants(active_only=False)
    
    return templates.TemplateResponse(
        "calls/index.html", 
        {
            "request": request, 
            "calls_data": calls_data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_prev": has_prev,
                "has_next": has_next,
                "prev_page": page - 1 if has_prev else None,
                "next_page": page + 1 if has_next else None,
                "page_range_start": page_range_start,
                "page_range_end": page_range_end,
                "page_numbers": page_numbers,
            },
            "filters": {
                "search": search or "",
                "status": status or "",
                "assistant_id": assistant_id or "",
                "date_range": date_range or "",
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
            "stats": {
                "total_calls": total_calls,
                "active_calls": active_calls,
                "completed_calls": completed_calls,
                "failed_calls": failed_calls,
                "success_rate": round(success_rate, 1),
                "avg_duration": avg_duration,
            },
            "assistants": assistants,
        }
    )


@router.get("/calls/{call_id}", response_class=HTMLResponse)
async def view_call(request: Request, call_id: int, db: Session = Depends(get_db)):
    """View a call with recordings and transcripts."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    # Get recordings for this call
    recordings = db.query(Recording).filter(Recording.call_id == call_id).all()
    
    # Get transcripts for this call, ordered by creation time
    transcripts = db.query(Transcript).filter(Transcript.call_id == call_id).order_by(Transcript.created_at).all()
    
    # Group transcripts by speaker for conversation view
    conversation = []
    current_speaker = None
    current_messages = []
    
    for transcript in transcripts:
        if transcript.speaker != current_speaker:
            if current_messages:
                conversation.append({
                    'speaker': current_speaker,
                    'messages': current_messages
                })
            current_speaker = transcript.speaker
            current_messages = [transcript]
        else:
            current_messages.append(transcript)
    
    # Add the last group
    if current_messages:
        conversation.append({
            'speaker': current_speaker,
            'messages': current_messages
        })

    return templates.TemplateResponse(
        "calls/view.html", {
            "request": request, 
            "call": call,
            "recordings": recordings,
            "transcripts": transcripts,
            "conversation": conversation
        }
    )


@router.get("/calls/export", response_class=Response)
async def export_calls(
    request: Request,
    format: str = "csv",
    search: str = None,
    status: str = None,
    assistant_id: int = None,
    date_range: str = None,
    db: Session = Depends(get_db)
):
    """Export calls data in CSV or JSON format."""
    import csv
    
    # Get calls with same filtering as list view
    query = db.query(Call)
    
    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Call.call_sid.ilike(search_term)) |
            (Call.customer_phone_number.ilike(search_term)) |
            (Call.to_phone_number.ilike(search_term))
        )
    
    # Apply status filter
    if status:
        if status == "active":
            query = query.filter(Call.status == "ongoing")
        elif status == "completed":
            query = query.filter(Call.status == "completed")
        elif status == "failed":
            query = query.filter(Call.status.in_(["failed", "no-answer", "busy", "canceled"]))
        else:
            query = query.filter(Call.status == status)
    
    # Apply assistant filter
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)
    
    # Apply date range filter
    if date_range:
        from datetime import datetime, timedelta
        today = datetime.now().date()
        
        if date_range == "today":
            query = query.filter(func.date(Call.started_at) == today)
        elif date_range == "yesterday":
            yesterday = today - timedelta(days=1)
            query = query.filter(func.date(Call.started_at) == yesterday)
        elif date_range == "week":
            week_ago = today - timedelta(days=7)
            query = query.filter(Call.started_at >= week_ago)
        elif date_range == "month":
            month_ago = today - timedelta(days=30)
            query = query.filter(Call.started_at >= month_ago)
    
    calls = query.order_by(desc(Call.started_at)).all()
    
    # Prepare export data
    export_data = []
    for call in calls:
        recording_count = db.query(Recording).filter(Recording.call_id == call.id).count()
        transcript_count = db.query(Transcript).filter(Transcript.call_id == call.id).count()
        
        # Calculate call quality from transcripts
        avg_confidence = db.query(func.avg(Transcript.confidence)).filter(
            Transcript.call_id == call.id,
            Transcript.confidence.isnot(None)
        ).scalar()
        quality = int(avg_confidence * 100) if avg_confidence else None
        
        export_data.append({
            "call_sid": call.call_sid,
            "assistant_name": call.assistant.name if call.assistant else "Unknown",
            "customer_phone": call.customer_phone_number,
            "to_phone": call.to_phone_number,
            "status": call.status.capitalize(),
            "duration": f"{call.duration}s" if call.duration else "N/A",
            "recording_count": recording_count,
            "transcript_count": transcript_count,
            "quality": f"{quality}%" if quality is not None else "N/A",
            "started_at": call.started_at.strftime("%Y-%m-%d %H:%M:%S") if call.started_at else "",
            "ended_at": call.ended_at.strftime("%Y-%m-%d %H:%M:%S") if call.ended_at else "",
        })
    
    if format.lower() == "json":
        # Export as JSON
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=calls.json"}
        )
    else:
        # Export as CSV (default)
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "call_sid", "assistant_name", "customer_phone", "to_phone", "status", 
            "duration", "recording_count", "transcript_count", "quality", 
            "started_at", "ended_at"
        ])
        writer.writeheader()
        writer.writerows(export_data)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=calls.csv"}
        )


@router.post("/calls/bulk-action")
async def bulk_action_calls(
    request: Request,
    action: str = Form(...),
    call_ids: str = Form(...),
    db: Session = Depends(get_db)
):
    """Perform bulk actions on calls."""
    try:
        # Parse call IDs
        ids = [int(id.strip()) for id in call_ids.split(",") if id.strip()]
        
        if not ids:
            return {"success": False, "message": "No calls selected"}
        
        # Get calls
        calls = db.query(Call).filter(Call.id.in_(ids)).all()
        
        if action == "delete":
            # Delete related records first
            for call in calls:
                # Delete recordings
                recordings = db.query(Recording).filter(Recording.call_id == call.id).all()
                for recording in recordings:
                    # Delete physical file if exists
                    if recording.file_path and os.path.exists(recording.file_path):
                        try:
                            os.remove(recording.file_path)
                        except Exception as e:
                            logging.warning(f"Could not delete recording file {recording.file_path}: {e}")
                    db.delete(recording)
                
                # Delete transcripts
                transcripts = db.query(Transcript).filter(Transcript.call_id == call.id).all()
                for transcript in transcripts:
                    db.delete(transcript)
                
                # Delete call
                db.delete(call)
            
            message = f"Deleted {len(calls)} calls with their recordings and transcripts"
            
        elif action == "download_recordings":
            # Create a zip file with all recordings
            import zipfile
            import tempfile
            
            # Create temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zip_file:
                for call in calls:
                    recordings = db.query(Recording).filter(Recording.call_id == call.id).all()
                    for recording in recordings:
                        if recording.file_path and os.path.exists(recording.file_path):
                            # Add file to zip with call-specific name
                            zip_file.write(
                                recording.file_path, 
                                f"{call.call_sid}_{recording.recording_sid}.mp3"
                            )
            
            return {"success": True, "message": f"Created download for {len(calls)} calls", "download_url": f"/download/temp/{os.path.basename(temp_zip.name)}"}
            
        else:
            return {"success": False, "message": "Invalid action"}
        
        db.commit()
        return {"success": True, "message": message}
        
    except Exception as e:
        db.rollback()
        return {"success": False, "message": f"Error: {str(e)}"}
