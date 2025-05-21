# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
from datetime import timedelta
import time
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Call
from app.services.assistant_service import AssistantService
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


@router.get("/assistants/new", response_class=HTMLResponse)
async def create_assistant_form(request: Request):
    """Show the create assistant form."""
    # Get available phone numbers from Twilio
    phone_numbers = TwilioService.get_available_phone_numbers()

    return templates.TemplateResponse(
        "assistants/form.html",
        {"request": request, "assistant": None, "phone_numbers": phone_numbers},
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
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
):
    """Create a new assistant."""
    # Check if an assistant with this phone number already exists
    existing = await AssistantService.get_assistant_by_phone(phone_number)
    if existing:
        phone_numbers = TwilioService.get_available_phone_numbers()
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": None,
                "phone_numbers": phone_numbers,
                "error": f"Assistant with phone number {phone_number} already exists",
            },
            status_code=400,
        )

    # Create assistant data with JSON settings
    assistant_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        # API Keys
        "openai_api_key": openai_api_key,
        "deepgram_api_key": deepgram_api_key,
        "elevenlabs_api_key": elevenlabs_api_key,
        "twilio_account_sid": twilio_account_sid,
        "twilio_auth_token": twilio_auth_token,
        "custom_llm_url": custom_llm_url,
        # JSON Settings
        "llm_settings": (
            {
                "model": llm_model,
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
            }
            if any([llm_model, llm_temperature, llm_max_tokens, llm_system_prompt])
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
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout,
    }

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
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": None,
                "phone_numbers": phone_numbers,
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

    return templates.TemplateResponse(
        "assistants/form.html",
        {"request": request, "assistant": assistant, "phone_numbers": phone_numbers},
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
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
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
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "assistant": assistant,
                    "phone_numbers": phone_numbers,
                    "error": f"Assistant with phone number {phone_number} already exists",
                },
                status_code=400,
            )

    # Create update data with JSON settings
    update_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        # API Keys
        "openai_api_key": openai_api_key or None,
        "deepgram_api_key": deepgram_api_key or None,
        "elevenlabs_api_key": elevenlabs_api_key or None,
        "twilio_account_sid": twilio_account_sid or None,
        "twilio_auth_token": twilio_auth_token or None,
        "custom_llm_url": custom_llm_url or None,
        # JSON Settings
        "llm_settings": (
            {
                "model": llm_model,
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
            }
            if any([llm_model, llm_temperature, llm_max_tokens, llm_system_prompt])
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
        "end_call_message": end_call_message or None,
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout,
    }

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
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": assistant,
                "phone_numbers": phone_numbers,
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
    calls = db.query(Call).order_by(Call.started_at.desc()).all()
    return templates.TemplateResponse(
        "calls/index.html", {"request": request, "calls": calls}
    )


@router.get("/calls/{call_id}", response_class=HTMLResponse)
async def view_call(request: Request, call_id: int, db: Session = Depends(get_db)):
    """View a call."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    return templates.TemplateResponse(
        "calls/view.html", {"request": request, "call": call}
    )
