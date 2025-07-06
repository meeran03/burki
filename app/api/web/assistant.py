"""
Assistant routes
"""

# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
import time
import json
import logging
from io import StringIO
import os
from fastapi import APIRouter, Depends, Request, Form, HTTPException, Response, UploadFile, File
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse
)
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, asc

from app.db.database import get_db, get_async_db_session
from app.db.models import (
    Call,
    Transcript,
    Assistant,
    User
)
from app.services.assistant_service import AssistantService
from app.services.auth_service import AuthService
from app.twilio.twilio_service import TwilioService
from app.core.assistant_manager import assistant_manager
from app.utils.url_utils import get_twiml_webhook_url

# Create router without a prefix - web routes will be at the root level
router = APIRouter(tags=["web"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Security
security = HTTPBearer(auto_error=False)
auth_service = AuthService()

# Track server start time for uptime display
start_time = time.time()

# Environment variables for configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# Initialize logger
logger = logging.getLogger(__name__)


# ========== Template Context Helpers ==========


def get_template_context(request: Request, **extra_context) -> dict:
    """Get template context with session data and any extra context."""
    context = {
        "request": request,
        "session": {
            "user_id": request.session.get("user_id"),
            "organization_id": request.session.get("organization_id"),
            "user_email": request.session.get("user_email", ""),
            "user_first_name": request.session.get("user_first_name", ""),
            "user_last_name": request.session.get("user_last_name", ""),
            "organization_name": request.session.get("organization_name", ""),
            "organization_slug": request.session.get("organization_slug", ""),
            "api_key_count": request.session.get("api_key_count", 0),
        },
    }
    context.update(extra_context)
    return context


async def get_current_user(
    request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """Get the current authenticated user from session."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        # Ensure first_name and last_name are populated
        user.split_full_name_if_needed()
    return user


async def require_auth(request: Request, db: Session = Depends(get_db)) -> User:
    """Require authentication and return the current user."""
    user = await get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Ensure first_name and last_name are populated
    user.split_full_name_if_needed()
    return user


@router.get("/assistants", response_class=HTMLResponse)
async def list_assistants(
    request: Request,
    current_user: User = Depends(require_auth),
    page: int = 1,
    per_page: int = 10,
    search: str = None,
    status: str = None,
    performance: str = None,
    sort_by: str = "name",
    sort_order: str = "asc",
    db: Session = Depends(get_db),
):
    """List assistants with pagination, filtering, and sorting."""
    # Base query - filter by organization and load phone numbers
    from sqlalchemy.orm import joinedload
    query = db.query(Assistant).options(joinedload(Assistant.phone_numbers)).filter(
        Assistant.organization_id == current_user.organization_id
    )

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Assistant.name.ilike(search_term))
            | (Assistant.description.ilike(search_term))
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
        completed_calls = (
            db.query(Call)
            .filter(Call.assistant_id == assistant.id, Call.status == "completed")
            .count()
        )

        # Calculate average duration
        avg_duration_result = (
            db.query(func.avg(Call.duration))
            .filter(Call.assistant_id == assistant.id, Call.duration.isnot(None))
            .scalar()
        )
        avg_duration = int(avg_duration_result) if avg_duration_result else 0

        # Calculate performance (based on transcript confidence)
        avg_confidence = (
            db.query(func.avg(Transcript.confidence))
            .filter(
                Transcript.call_id.in_(
                    db.query(Call.id).filter(Call.assistant_id == assistant.id)
                ),
                Transcript.confidence.isnot(None),
            )
            .scalar()
        )
        performance = int(avg_confidence * 100) if avg_confidence else 90

        # Add stats to assistant object
        assistant.total_calls = total_calls
        assistant.completed_calls = completed_calls
        assistant.avg_duration = avg_duration
        assistant.performance = performance
        assistant.success_rate = (
            (completed_calls / total_calls * 100) if total_calls > 0 else 0
        )

        assistants_with_stats.append(assistant)

    # Apply performance filter after calculating stats
    if performance:
        if performance == "excellent":
            assistants_with_stats = [
                a for a in assistants_with_stats if a.performance >= 95
            ]
        elif performance == "good":
            assistants_with_stats = [
                a for a in assistants_with_stats if 85 <= a.performance < 95
            ]
        elif performance == "needs-improvement":
            assistants_with_stats = [
                a for a in assistants_with_stats if a.performance < 85
            ]

    # Calculate overall statistics
    total_assistants = (
        db.query(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
        .count()
    )
    active_assistants = (
        db.query(Assistant)
        .filter(
            Assistant.organization_id == current_user.organization_id,
            Assistant.is_active == True,
        )
        .count()
    )
    total_calls_all = (
        db.query(Call)
        .filter(
            Call.assistant_id.in_(
                db.query(Assistant.id).filter(
                    Assistant.organization_id == current_user.organization_id
                )
            )
        )
        .count()
    )

    # Calculate average performance across all assistants
    all_confidences = (
        db.query(func.avg(Transcript.confidence))
        .filter(
            Transcript.call_id.in_(
                db.query(Call.id).filter(
                    Call.assistant_id.in_(
                        db.query(Assistant.id).filter(
                            Assistant.organization_id == current_user.organization_id
                        )
                    )
                )
            ),
            Transcript.confidence.isnot(None),
        )
        .scalar()
    )
    avg_performance = int(all_confidences * 100) if all_confidences else 98.7

    return templates.TemplateResponse(
        "assistants/index.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
            assistants=assistants_with_stats,
            pagination={
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
            filters={
                "search": search or "",
                "status": status or "",
                "performance": performance or "",
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
            stats={
                "total_assistants": total_assistants,
                "active_assistants": active_assistants,
                "total_calls": total_calls_all,
                "avg_performance": avg_performance,
            },
        ),
    )


@router.get("/assistants/new", response_class=HTMLResponse)
async def create_assistant_form(
    request: Request, current_user: User = Depends(require_auth)
):
    """Show the create assistant form."""
    # Default schema for structured data
    default_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "chat_topic": {
                    "type": "string",
                    "description": "The main topic of the conversation",
                },
                "followup_sms": {
                    "type": "string",
                    "description": "A follow-up SMS message to send to the customer",
                },
            },
            "required": ["chat_topic"],
        },
        indent=2,
    )

    return templates.TemplateResponse(
        "assistants/form.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
            assistant=None,
            default_schema=default_schema,
        ),
    )


@router.post("/assistants/new", response_class=HTMLResponse)
async def create_assistant(
    request: Request,
    current_user: User = Depends(require_auth),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    # LLM Provider Configuration
    llm_provider: str = Form(...),
    llm_provider_api_key: Optional[str] = Form(None),
    llm_provider_model: Optional[str] = Form(None),
    llm_provider_base_url: Optional[str] = Form(None),
    # Service API Keys
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    inworld_bearer_token: Optional[str] = Form(None),
    resemble_api_key: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    # LLM Settings
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
    tts_provider: Optional[str] = Form("elevenlabs"),
    # Deepgram TTS Settings
    tts_encoding: Optional[str] = Form("mulaw"),
    tts_sample_rate: Optional[int] = Form(8000),
    # Resemble AI TTS Settings
    tts_project_uuid: Optional[str] = Form(None),
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
    stt_keywords: Optional[str] = Form(None),
    stt_keyterms: Optional[str] = Form(None),
    stt_audio_denoising: Optional[bool] = Form(False),
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
    # RAG settings
    rag_enabled: bool = Form(False),
    rag_search_limit: Optional[int] = Form(None),
    rag_similarity_threshold: Optional[float] = Form(None),
    rag_chunk_size: Optional[int] = Form(None),
    # Tools configuration
    end_call_enabled: bool = Form(False),
    end_call_scenarios: Optional[str] = Form(None),
    end_call_custom_message: Optional[str] = Form(None),
    transfer_call_enabled: bool = Form(False),
    transfer_call_scenarios: Optional[str] = Form(None),
    transfer_call_numbers: Optional[str] = Form(None),
    transfer_call_custom_message: Optional[str] = Form(None),
    # Fallback providers configuration
    fallback_enabled: bool = Form(False),
    fallback_0_enabled: bool = Form(False),
    fallback_0_provider: Optional[str] = Form(None),
    fallback_0_model: Optional[str] = Form(None),
    fallback_0_api_key: Optional[str] = Form(None),
    fallback_0_base_url: Optional[str] = Form(None),
    fallback_1_enabled: bool = Form(False),
    fallback_1_provider: Optional[str] = Form(None),
    fallback_1_model: Optional[str] = Form(None),
    fallback_1_api_key: Optional[str] = Form(None),
    fallback_1_base_url: Optional[str] = Form(None),
    fallback_2_enabled: bool = Form(False),
    fallback_2_provider: Optional[str] = Form(None),
    fallback_2_model: Optional[str] = Form(None),
    fallback_2_api_key: Optional[str] = Form(None),
    fallback_2_base_url: Optional[str] = Form(None),
    # Inworld TTS Settings
    tts_language: Optional[str] = Form("en"),
    custom_voice_id: Optional[str] = Form(None),
    elevenlabs_language: Optional[str] = Form("en"),
):
    """Create a new assistant."""

    # Helper function to convert empty strings to None
    def empty_to_none(value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    # Build TTS settings
    tts_settings = {
        "provider": tts_provider or "elevenlabs",
        "voice_id": tts_voice_id or "rachel", 
        "model_id": tts_model_id or ("turbo" if tts_provider == "elevenlabs" else "aura"),
        "latency": tts_latency if tts_latency is not None else 1,
        "stability": tts_stability if tts_stability is not None else 0.5,
        "similarity_boost": tts_similarity_boost if tts_similarity_boost is not None else 0.75,
        "style": tts_style if tts_style is not None else 0.0,
        "use_speaker_boost": tts_use_speaker_boost if tts_use_speaker_boost is not None else True,
        "provider_config": {}
    }
    
    # Add provider-specific configurations
    if tts_provider == "deepgram":
        # Get additional Deepgram-specific fields
        tts_settings["provider_config"] = {
            "encoding": tts_encoding or "mulaw",
            "sample_rate": int(tts_sample_rate) if tts_sample_rate else 8000
        }
    elif tts_provider == "resemble":
        # Get additional Resemble-specific fields
        tts_settings["provider_config"] = {
            "project_uuid": tts_project_uuid or os.getenv("RESEMBLE_PROJECT_UUID"),
        }
    elif tts_provider == "inworld":
        # Inworld TTS specific configuration
        tts_settings["provider_config"] = {
            "language": tts_language or "en",
            "custom_voice_id": empty_to_none(custom_voice_id),
        }
    elif tts_provider == "elevenlabs":
        # ElevenLabs doesn't need additional config for now, but can be extended
        tts_settings["provider_config"] = {
            "language": elevenlabs_language or "en",
        }

    # Create assistant data with JSON settings
    assistant_data = {
        "name": name,
        "description": description,
        "is_active": is_active,
        # Note: organization_id and user_id are passed as separate parameters to create_assistant
        # LLM Provider Configuration
        "llm_provider": llm_provider,
        "llm_provider_config": {
            "api_key": empty_to_none(llm_provider_api_key),
            "model": empty_to_none(llm_provider_model),
            "base_url": empty_to_none(llm_provider_base_url),
        },
        # Clear legacy custom_llm_url when using a different provider
        "custom_llm_url": llm_provider_base_url if llm_provider == "custom" else None,
        # Service API Keys
        "deepgram_api_key": empty_to_none(deepgram_api_key),
        "elevenlabs_api_key": empty_to_none(elevenlabs_api_key),
        "inworld_bearer_token": empty_to_none(inworld_bearer_token),
        "resemble_api_key": empty_to_none(resemble_api_key),
        "twilio_account_sid": empty_to_none(twilio_account_sid),
        "twilio_auth_token": empty_to_none(twilio_auth_token),
        # JSON Settings
        "llm_settings": (
            {
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
                "welcome_message": empty_to_none(welcome_message),
            }
            if any(
                [llm_temperature, llm_max_tokens, llm_system_prompt, welcome_message]
            )
            else None
        ),
        "tts_settings": tts_settings,
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
                "audio_denoising": stt_audio_denoising,
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
                    stt_audio_denoising,
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
        # Tools configuration
        "tools_settings": {
            "enabled_tools": [],  # Will be populated below
            "end_call": {
                "enabled": end_call_enabled,
                "scenarios": [s.strip() for s in end_call_scenarios.split(",")] if end_call_scenarios and end_call_scenarios.strip() else [],
                "custom_message": empty_to_none(end_call_custom_message),
            },
            "transfer_call": {
                "enabled": transfer_call_enabled,
                "scenarios": [s.strip() for s in transfer_call_scenarios.split(",")] if transfer_call_scenarios and transfer_call_scenarios.strip() else [],
                "transfer_numbers": [n.strip() for n in transfer_call_numbers.split(",")] if transfer_call_numbers and transfer_call_numbers.strip() else [],
                "custom_message": empty_to_none(transfer_call_custom_message),
            },
            "custom_tools": [],  # For future custom tool definitions
        },
        # RAG settings
        "rag_settings": (
            {
                "enabled": rag_enabled,
                "search_limit": rag_search_limit if rag_search_limit is not None else 3,
                "similarity_threshold": rag_similarity_threshold if rag_similarity_threshold is not None else 0.7,
                "embedding_model": "text-embedding-3-small",
                "chunking_strategy": "recursive",
                "chunk_size": rag_chunk_size if rag_chunk_size is not None else 1000,
                "chunk_overlap": 200,
                "auto_process": True,
                "include_metadata": True,
                "context_window_tokens": 4000,
            }
            if rag_enabled or any([rag_search_limit, rag_similarity_threshold, rag_chunk_size])
            else None
        ),
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
            default_schema = json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "chat_topic": {
                            "type": "string",
                            "description": "The main topic of the conversation",
                        },
                        "followup_sms": {
                            "type": "string",
                            "description": "A follow-up SMS message to send to the customer",
                        },
                    },
                    "required": ["chat_topic"],
                },
                indent=2,
            )
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "current_user": current_user,
                    "organization": current_user.organization,
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
                for keyword in stt_keywords.split(","):
                    keyword = keyword.strip()
                    if ":" in keyword:
                        # Handle keyword with intensifier (e.g., "OpenAI:2.0")
                        word, intensifier = keyword.split(":", 1)
                        keywords_list.append(
                            {
                                "keyword": word.strip(),
                                "intensifier": float(intensifier.strip()),
                            }
                        )
                    elif keyword:
                        # Handle keyword without intensifier (default 1.0)
                        keywords_list.append({"keyword": keyword, "intensifier": 1.0})
                assistant_data["stt_settings"]["keywords"] = keywords_list
            except (ValueError, AttributeError) as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "current_user": current_user,
                        "organization": current_user.organization,
                        "assistant": None,
                        "phone_numbers": phone_numbers,
                        "error": f"Invalid keywords format: {str(e)}. Use format: 'keyword1:2.0, keyword2, keyword3:1.5'",
                    },
                    status_code=400,
                )

        # Parse keyterms from comma-separated string
        if stt_keyterms:
            try:
                keyterms_list = [
                    term.strip() for term in stt_keyterms.split(",") if term.strip()
                ]
                assistant_data["stt_settings"]["keyterms"] = keyterms_list
            except AttributeError as e:
                phone_numbers = TwilioService.get_available_phone_numbers()
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "current_user": current_user,
                        "organization": current_user.organization,
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

    # Populate enabled_tools list based on tool configurations
    if assistant_data.get("tools_settings"):
        enabled_tools = []
        if assistant_data["tools_settings"]["end_call"]["enabled"]:
            enabled_tools.append("endCall")
        if assistant_data["tools_settings"]["transfer_call"]["enabled"]:
            enabled_tools.append("transferCall")
        assistant_data["tools_settings"]["enabled_tools"] = enabled_tools

    # Process fallback providers configuration
    if fallback_enabled:
        fallbacks = []
        fallback_configs = [
            (fallback_0_enabled, fallback_0_provider, fallback_0_model, fallback_0_api_key, fallback_0_base_url),
            (fallback_1_enabled, fallback_1_provider, fallback_1_model, fallback_1_api_key, fallback_1_base_url),
            (fallback_2_enabled, fallback_2_provider, fallback_2_model, fallback_2_api_key, fallback_2_base_url),
        ]
        
        for enabled, provider, model, api_key, base_url in fallback_configs:
            if enabled and provider:
                fallback_config = {
                    "enabled": True,
                    "provider": provider,
                    "config": {
                        "api_key": empty_to_none(api_key),
                        "model": empty_to_none(model),
                        "base_url": empty_to_none(base_url),
                        "custom_config": {}
                    }
                }
                fallbacks.append(fallback_config)
        
        assistant_data["llm_fallback_providers"] = {
            "enabled": True,
            "fallbacks": fallbacks
        }
    else:
        assistant_data["llm_fallback_providers"] = {
            "enabled": False,
            "fallbacks": []
        }

    # Remove None values and empty dictionaries
    assistant_data = {k: v for k, v in assistant_data.items() if v is not None and v != {}}

    # Create the assistant
    try:
        new_assistant = await AssistantService.create_assistant(
            assistant_data, 
            current_user.id, 
            current_user.organization_id
        )
        await assistant_manager.load_assistants()
        return RedirectResponse(url=f"/assistants/{new_assistant.id}", status_code=302)
    except Exception as e:
        phone_numbers = TwilioService.get_available_phone_numbers()
        default_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "chat_topic": {
                        "type": "string",
                        "description": "The main topic of the conversation",
                    },
                    "followup_sms": {
                        "type": "string",
                        "description": "A follow-up SMS message to send to the customer",
                    },
                },
                "required": ["chat_topic"],
            },
            indent=2,
        )
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "current_user": current_user,
                "organization": current_user.organization,
                "assistant": None,
                "phone_numbers": phone_numbers,
                "default_schema": default_schema,
                "error": f"Error creating assistant: {str(e)}",
            },
            status_code=500,
        )


@router.get("/assistants/{assistant_id}", response_class=HTMLResponse)
async def view_assistant(
    request: Request, assistant_id: int, current_user: User = Depends(require_auth), db: Session = Depends(get_db)
):
    """View an assistant."""
    # Get assistant with phone numbers loaded
    async with await get_async_db_session() as async_db:
        from sqlalchemy.orm import joinedload
        from sqlalchemy import select
        result = await async_db.execute(
            select(Assistant)
            .options(joinedload(Assistant.phone_numbers))
            .where(
                Assistant.id == assistant_id,
                Assistant.organization_id == current_user.organization_id
            )
        )
        assistant = result.unique().scalar_one_or_none()
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
        get_template_context(request, assistant=assistant, calls=calls),
    )


@router.get("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def edit_assistant_form(request: Request, assistant_id: int, current_user: User = Depends(require_auth)):
    """Show the edit assistant form."""
    assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Load active documents using RAG service (filters out soft-deleted documents)
    documents = []
    try:
        # Check if RAG is available and load documents properly
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService.create_default_instance()
            # Get only active documents (is_active = True)
            documents = await rag_service.get_assistant_documents(
                assistant_id, 
                include_processing=True  # Include documents being processed
            )
        except ImportError:
            # RAG not available, just use empty list
            logger.warning("RAG service not available for loading documents")
            documents = []
    except Exception as e:
        # If there's an error loading documents, just log it and continue
        logger.warning(f"Could not load documents: {e}")
        documents = []

    # Default schema for structured data
    default_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "chat_topic": {
                    "type": "string",
                    "description": "The main topic of the conversation",
                },
                "followup_sms": {
                    "type": "string",
                    "description": "A follow-up SMS message to send to the customer",
                },
            },
            "required": ["chat_topic"],
        },
        indent=2,
    )

    return templates.TemplateResponse(
        "assistants/form.html",
        get_template_context(
            request,
            assistant=assistant,
            documents=documents,  # Pass documents separately
            default_schema=default_schema,
        ),
    )


@router.post("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def update_assistant(
    request: Request,
    assistant_id: int,
    current_user: User = Depends(require_auth),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    # LLM Provider Configuration
    llm_provider: str = Form(...),
    llm_provider_api_key: Optional[str] = Form(None),
    llm_provider_model: Optional[str] = Form(None),
    llm_provider_base_url: Optional[str] = Form(None),
    # Service API Keys
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    inworld_bearer_token: Optional[str] = Form(None),
    resemble_api_key: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    # LLM Settings
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
    tts_provider: Optional[str] = Form("elevenlabs"),
    # Deepgram TTS Settings
    tts_encoding: Optional[str] = Form("mulaw"),
    tts_sample_rate: Optional[int] = Form(8000),
    # Resemble AI TTS Settings
    tts_project_uuid: Optional[str] = Form(None),
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
    stt_keywords: Optional[str] = Form(None),
    stt_keyterms: Optional[str] = Form(None),
    stt_audio_denoising: Optional[bool] = Form(False),
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
    # RAG settings
    rag_enabled: bool = Form(False),
    rag_search_limit: Optional[int] = Form(None),
    rag_similarity_threshold: Optional[float] = Form(None),
    rag_chunk_size: Optional[int] = Form(None),
    # Tools configuration
    end_call_enabled: bool = Form(False),
    end_call_scenarios: Optional[str] = Form(None),
    end_call_custom_message: Optional[str] = Form(None),
    transfer_call_enabled: bool = Form(False),
    transfer_call_scenarios: Optional[str] = Form(None),
    transfer_call_numbers: Optional[str] = Form(None),
    transfer_call_custom_message: Optional[str] = Form(None),
    # Fallback providers configuration
    fallback_enabled: bool = Form(False),
    fallback_0_enabled: bool = Form(False),
    fallback_0_provider: Optional[str] = Form(None),
    fallback_0_model: Optional[str] = Form(None),
    fallback_0_api_key: Optional[str] = Form(None),
    fallback_0_base_url: Optional[str] = Form(None),
    fallback_1_enabled: bool = Form(False),
    fallback_1_provider: Optional[str] = Form(None),
    fallback_1_model: Optional[str] = Form(None),
    fallback_1_api_key: Optional[str] = Form(None),
    fallback_1_base_url: Optional[str] = Form(None),
    fallback_2_enabled: bool = Form(False),
    fallback_2_provider: Optional[str] = Form(None),
    fallback_2_model: Optional[str] = Form(None),
    fallback_2_api_key: Optional[str] = Form(None),
    fallback_2_base_url: Optional[str] = Form(None),
    # Inworld TTS Settings
    tts_language: Optional[str] = Form("en"),
    custom_voice_id: Optional[str] = Form(None),
    elevenlabs_language: Optional[str] = Form("en"),
):
    """Update an assistant."""
    assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Note: Phone number management is now handled separately through the PhoneNumber table

    # Helper function to convert empty strings to None
    def empty_to_none(value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    # Build TTS settings
    tts_settings = {
        "provider": tts_provider or "elevenlabs",
        "voice_id": tts_voice_id or "rachel", 
        "model_id": tts_model_id or ("turbo" if tts_provider == "elevenlabs" else "aura"),
        "latency": tts_latency if tts_latency is not None else 1,
        "stability": tts_stability if tts_stability is not None else 0.5,
        "similarity_boost": tts_similarity_boost if tts_similarity_boost is not None else 0.75,
        "style": tts_style if tts_style is not None else 0.0,
        "use_speaker_boost": tts_use_speaker_boost if tts_use_speaker_boost is not None else True,
        "provider_config": {}
    }
    
    # Add provider-specific configurations
    if tts_provider == "deepgram":
        # Get additional Deepgram-specific fields
        tts_settings["provider_config"] = {
            "encoding": tts_encoding or "mulaw",
            "sample_rate": int(tts_sample_rate) if tts_sample_rate else 8000
        }
    elif tts_provider == "resemble":
        # Get additional Resemble-specific fields
        tts_settings["provider_config"] = {
            "project_uuid": tts_project_uuid or os.getenv("RESEMBLE_PROJECT_UUID"),
        }
    elif tts_provider == "inworld":
        # Inworld TTS specific configuration
        tts_settings["provider_config"] = {
            "language": tts_language or "en",
            "custom_voice_id": empty_to_none(custom_voice_id),
        }
    elif tts_provider == "elevenlabs":
        # ElevenLabs doesn't need additional config for now, but can be extended
        tts_settings["provider_config"] = {
            "language": elevenlabs_language or "en",
        }

    # Create update data with JSON settings
    update_data = {
        "name": name,
        "description": description,
        "is_active": is_active,
        # LLM Provider Configuration
        "llm_provider": llm_provider,
        "llm_provider_config": {
            "api_key": empty_to_none(llm_provider_api_key),
            "model": empty_to_none(llm_provider_model),
            "base_url": empty_to_none(llm_provider_base_url) if llm_provider == "custom" else None,
        },
        # Clear legacy custom_llm_url when using a different provider
        "custom_llm_url": llm_provider_base_url if llm_provider == "custom" else None,
        # Service API Keys
        "deepgram_api_key": empty_to_none(deepgram_api_key),
        "elevenlabs_api_key": empty_to_none(elevenlabs_api_key),
        "resemble_api_key": empty_to_none(resemble_api_key),
        "inworld_bearer_token": empty_to_none(inworld_bearer_token),
        "twilio_account_sid": empty_to_none(twilio_account_sid),
        "twilio_auth_token": empty_to_none(twilio_auth_token),
        # JSON Settings
        "llm_settings": (
            {
                "temperature": llm_temperature,
                "max_tokens": llm_max_tokens,
                "system_prompt": llm_system_prompt,
                "welcome_message": empty_to_none(welcome_message),
            }
            if any(
                [llm_temperature, llm_max_tokens, llm_system_prompt, welcome_message]
            )
            else None
        ),
        "tts_settings": tts_settings,
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
                "audio_denoising": stt_audio_denoising,
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
                    stt_audio_denoising,
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
        # Tools configuration
        "tools_settings": {
            "enabled_tools": [],  # Will be populated below
            "end_call": {
                "enabled": end_call_enabled,
                "scenarios": [s.strip() for s in end_call_scenarios.split(",")] if end_call_scenarios and end_call_scenarios.strip() else [],
                "custom_message": empty_to_none(end_call_custom_message),
            },
            "transfer_call": {
                "enabled": transfer_call_enabled,
                "scenarios": [s.strip() for s in transfer_call_scenarios.split(",")] if transfer_call_scenarios and transfer_call_scenarios.strip() else [],
                "transfer_numbers": [n.strip() for n in transfer_call_numbers.split(",")] if transfer_call_numbers and transfer_call_numbers.strip() else [],
                "custom_message": empty_to_none(transfer_call_custom_message),
            },
            "custom_tools": [],  # For future custom tool definitions
        },
        # RAG settings
        "rag_settings": (
            {
                "enabled": rag_enabled,
                "search_limit": rag_search_limit if rag_search_limit is not None else 3,
                "similarity_threshold": rag_similarity_threshold if rag_similarity_threshold is not None else 0.7,
                "embedding_model": "text-embedding-3-small",
                "chunking_strategy": "recursive",
                "chunk_size": rag_chunk_size if rag_chunk_size is not None else 1000,
                "chunk_overlap": 200,
                "auto_process": True,
                "include_metadata": True,
                "context_window_tokens": 4000,
            }
            if rag_enabled or any([rag_search_limit, rag_similarity_threshold, rag_chunk_size])
            else None
        ),
    }

    # Handle custom settings separately
    custom_settings = {}
    if structured_data_schema:
        try:
            # Parse the JSON schema
            schema_data = json.loads(structured_data_schema)
            custom_settings["structured_data_schema"] = schema_data
        except json.JSONDecodeError:
            default_schema = json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "chat_topic": {
                            "type": "string",
                            "description": "The main topic of the conversation",
                        },
                        "followup_sms": {
                            "type": "string",
                            "description": "A follow-up SMS message to send to the customer",
                        },
                    },
                    "required": ["chat_topic"],
                },
                indent=2,
            )
            return templates.TemplateResponse(
                "assistants/form.html",
                {
                    "request": request,
                    "assistant": assistant,
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
                for keyword in stt_keywords.split(","):
                    keyword = keyword.strip()
                    if ":" in keyword:
                        # Handle keyword with intensifier (e.g., "OpenAI:2.0")
                        word, intensifier = keyword.split(":", 1)
                        keywords_list.append(
                            {
                                "keyword": word.strip(),
                                "intensifier": float(intensifier.strip()),
                            }
                        )
                    elif keyword:
                        # Handle keyword without intensifier (default 1.0)
                        keywords_list.append({"keyword": keyword, "intensifier": 1.0})
                update_data["stt_settings"]["keywords"] = keywords_list
            except (ValueError, AttributeError) as e:
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": assistant,
                        "error": f"Invalid keywords format: {str(e)}. Use format: 'keyword1:2.0, keyword2, keyword3:1.5'",
                    },
                    status_code=400,
                )

        # Parse keyterms from comma-separated string
        if stt_keyterms:
            try:
                keyterms_list = [
                    term.strip() for term in stt_keyterms.split(",") if term.strip()
                ]
                update_data["stt_settings"]["keyterms"] = keyterms_list
            except AttributeError as e:
                return templates.TemplateResponse(
                    "assistants/form.html",
                    {
                        "request": request,
                        "assistant": assistant,
                        "error": f"Invalid keyterms format: {str(e)}. Use comma-separated terms.",
                    },
                    status_code=400,
                )

    # Add structured data prompt if provided
    if structured_data_prompt and structured_data_prompt.strip():
        custom_settings["structured_data_prompt"] = structured_data_prompt.strip()

    if custom_settings:
        update_data["custom_settings"] = custom_settings

    # Populate enabled_tools list based on tool configurations
    if update_data.get("tools_settings"):
        enabled_tools = []
        if update_data["tools_settings"]["end_call"]["enabled"]:
            enabled_tools.append("endCall")
        if update_data["tools_settings"]["transfer_call"]["enabled"]:
            enabled_tools.append("transferCall")
        update_data["tools_settings"]["enabled_tools"] = enabled_tools

    # Process fallback providers configuration
    if fallback_enabled:
        fallbacks = []
        fallback_configs = [
            (fallback_0_enabled, fallback_0_provider, fallback_0_model, fallback_0_api_key, fallback_0_base_url),
            (fallback_1_enabled, fallback_1_provider, fallback_1_model, fallback_1_api_key, fallback_1_base_url),
            (fallback_2_enabled, fallback_2_provider, fallback_2_model, fallback_2_api_key, fallback_2_base_url),
        ]
        
        for enabled, provider, model, api_key, base_url in fallback_configs:
            if enabled and provider:
                fallback_config = {
                    "enabled": True,
                    "provider": provider,
                    "config": {
                        "api_key": empty_to_none(api_key),
                        "model": empty_to_none(model),
                        "base_url": empty_to_none(base_url),
                        "custom_config": {}
                    }
                }
                fallbacks.append(fallback_config)
        
        update_data["llm_fallback_providers"] = {
            "enabled": True,
            "fallbacks": fallbacks
        }
    else:
        update_data["llm_fallback_providers"] = {
            "enabled": False,
            "fallbacks": []
        }

    # Remove None values and empty dictionaries
    update_data = {k: v for k, v in update_data.items() if v is not None and v != {}}

    # Update the assistant
    try:
        updated_assistant = await AssistantService.update_assistant(
            assistant_id, update_data, current_user.organization_id
        )
        await assistant_manager.load_assistants()
        return RedirectResponse(
            url=f"/assistants/{updated_assistant.id}", status_code=302
        )
    except Exception as e:
        default_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "chat_topic": {
                        "type": "string",
                        "description": "The main topic of the conversation",
                    },
                    "followup_sms": {
                        "type": "string",
                        "description": "A follow-up SMS message to send to the customer",
                    },
                },
                "required": ["chat_topic"],
            },
            indent=2,
        )
        return templates.TemplateResponse(
            "assistants/form.html",
            {
                "request": request,
                "assistant": assistant,
                "default_schema": default_schema,
                "error": f"Error updating assistant: {str(e)}",
            },
            status_code=500,
        )


@router.get("/assistants/{assistant_id}/delete")
async def delete_assistant(request: Request, assistant_id: int, current_user: User = Depends(require_auth)):
    """Delete an assistant."""
    # Check if assistant exists
    assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Delete the assistant
    await AssistantService.delete_assistant(assistant_id, current_user.organization_id)

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
    db: Session = Depends(get_db),
):
    """Export assistants data in CSV or JSON format."""
    import csv

    # Get assistants with same filtering as list view
    query = db.query(Assistant)

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Assistant.name.ilike(search_term))
            | (Assistant.description.ilike(search_term))
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
        completed_calls = (
            db.query(Call)
            .filter(Call.assistant_id == assistant.id, Call.status == "completed")
            .count()
        )

        avg_duration_result = (
            db.query(func.avg(Call.duration))
            .filter(Call.assistant_id == assistant.id, Call.duration.isnot(None))
            .scalar()
        )
        avg_duration = int(avg_duration_result) if avg_duration_result else 0

        avg_confidence = (
            db.query(func.avg(Transcript.confidence))
            .filter(
                Transcript.call_id.in_(
                    db.query(Call.id).filter(Call.assistant_id == assistant.id)
                ),
                Transcript.confidence.isnot(None),
            )
            .scalar()
        )
        performance = int(avg_confidence * 100) if avg_confidence else 90

        export_data.append(
            {
                "id": assistant.id,
                "name": assistant.name,
                "description": assistant.description or "",
                "status": "Active" if assistant.is_active else "Inactive",
                "total_calls": total_calls,
                "completed_calls": completed_calls,
                "success_rate": (
                    f"{(completed_calls / total_calls * 100):.1f}%"
                    if total_calls > 0
                    else "0%"
                ),
                "avg_duration": f"{avg_duration}s" if avg_duration > 0 else "N/A",
                "performance": f"{performance}%",
                "created_at": (
                    assistant.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if assistant.created_at
                    else ""
                ),
            }
        )

    if format.lower() == "json":
        # Export as JSON
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=assistants.json"},
        )
    else:
        # Export as CSV (default)
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id",
                "name",
                "description",
                "status",
                "total_calls",
                "completed_calls",
                "success_rate",
                "avg_duration",
                "performance",
                "created_at",
            ],
        )
        writer.writeheader()
        writer.writerows(export_data)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=assistants.csv"},
        )


@router.post("/assistants/bulk-action")
async def bulk_action_assistants(
    request: Request,
    action: str = Form(...),
    assistant_ids: str = Form(...),
    db: Session = Depends(get_db),
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


@router.post("/assistants/fetch-phone-numbers", response_class=JSONResponse)
async def fetch_phone_numbers(
    request: Request,
    current_user: User = Depends(require_auth),
    twilio_account_sid: str = Form(...),
    twilio_auth_token: str = Form(...),
):
    """
    Fetch available phone numbers from user's Twilio account.
    This endpoint allows users to test their Twilio credentials and see available numbers.
    """
    try:
        # Validate credentials and fetch phone numbers
        phone_numbers = TwilioService.get_available_phone_numbers(
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token
        )
        
        if not phone_numbers:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "No phone numbers found or invalid Twilio credentials",
                    "phone_numbers": []
                }
            )
        
        return JSONResponse(
            content={
                "success": True,
                "phone_numbers": phone_numbers,
                "message": f"Found {len(phone_numbers)} phone numbers"
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching phone numbers for user {current_user.id}: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Failed to fetch phone numbers: {str(e)}",
                "phone_numbers": []
            }
        )
