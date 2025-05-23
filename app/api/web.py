# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
from datetime import timedelta
import time
import logging
import json
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import io
import os

from app.db.database import get_db
from app.db.models import Call, Recording, Transcript
from app.services.assistant_service import AssistantService
from app.services.call_service import CallService
from app.core.assistant_manager import assistant_manager
from app.twilio.twilio_service import TwilioService

# Create router without a prefix - web routes will be at the root level
router = APIRouter(tags=["web"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Track server start time for uptime display
start_time = time.time()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    """Dashboard page."""
    # Get active assistants
    active_assistants = await AssistantService.get_assistants(active_only=True)

    # Get call statistics
    total_calls = db.query(Call).count()
    active_calls = db.query(Call).filter(Call.status == "ongoing").count()

    # Get recent calls
    recent_calls = db.query(Call).order_by(Call.started_at.desc()).limit(10).all()

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
            "recent_calls": recent_calls,
            "uptime": uptime,
        },
    )


# Use a specific route prefix for each section to avoid conflicts
# ========== Assistants Routes ==========


@router.get("/assistants", response_class=HTMLResponse)
async def list_assistants(request: Request):
    """List all assistants."""
    assistants = await AssistantService.get_assistants()
    return templates.TemplateResponse(
        "assistants/index.html", {"request": request, "assistants": assistants}
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
    # Interruption Settings
    interruption_threshold: Optional[int] = Form(None),
    min_speaking_time: Optional[float] = Form(None),
    interruption_cooldown: Optional[float] = Form(None),
    # Call control settings
    end_call_message: Optional[str] = Form(None),
    transfer_call_message: Optional[str] = Form(None),
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
    # Interruption Settings
    interruption_threshold: Optional[int] = Form(None),
    min_speaking_time: Optional[float] = Form(None),
    interruption_cooldown: Optional[float] = Form(None),
    # Call control settings
    end_call_message: Optional[str] = Form(None),
    transfer_call_message: Optional[str] = Form(None),
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


# ========== Calls Routes ==========
@router.get("/calls", response_class=HTMLResponse)
async def list_calls(request: Request, db: Session = Depends(get_db)):
    """List all calls."""
    # Get all calls with related data
    calls = db.query(Call).order_by(Call.started_at.desc()).all()
    
    # Add recording and transcript counts for each call
    calls_data = []
    for call in calls:
        recording_count = db.query(Recording).filter(Recording.call_id == call.id).count()
        transcript_count = db.query(Transcript).filter(Transcript.call_id == call.id).count()
        
        calls_data.append({
            'call': call,
            'recording_count': recording_count,
            'transcript_count': transcript_count,
            'has_recording': recording_count > 0,
            'has_transcripts': transcript_count > 0
        })
    
    return templates.TemplateResponse(
        "calls/index.html", {"request": request, "calls_data": calls_data}
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
