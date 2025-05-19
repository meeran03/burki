from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Assistant, Call
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
    active_assistants = await AssistantService.get_assistants(db, active_only=True)
    
    # Get call statistics
    total_calls = db.query(Call).count()
    active_calls = db.query(Call).filter(Call.status == "ongoing").count()
    
    # Get recent calls
    recent_calls = db.query(Call).order_by(Call.started_at.desc()).limit(10).all()
    
    # Calculate uptime
    uptime_seconds = time.time() - start_time
    uptime = str(timedelta(seconds=int(uptime_seconds)))
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_assistants": active_assistants,
        "total_calls": total_calls,
        "active_calls": active_calls,
        "recent_calls": recent_calls,
        "uptime": uptime
    })

# Use a specific route prefix for each section to avoid conflicts
# ========== Assistants Routes ==========

@router.get("/assistants", response_class=HTMLResponse)
async def list_assistants(request: Request, db: Session = Depends(get_db)):
    """List all assistants."""
    assistants = await AssistantService.get_assistants(db)
    return templates.TemplateResponse("assistants/index.html", {
        "request": request,
        "assistants": assistants
    })

@router.get("/assistants/new", response_class=HTMLResponse)
async def create_assistant_form(request: Request):
    """Show the create assistant form."""
    # Get available phone numbers from Twilio
    phone_numbers = TwilioService.get_available_phone_numbers()
    
    return templates.TemplateResponse("assistants/form.html", {
        "request": request,
        "assistant": None,
        "phone_numbers": phone_numbers
    })

@router.post("/assistants/new", response_class=HTMLResponse)
async def create_assistant(
    request: Request,
    name: str = Form(...),
    phone_number: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    elevenlabs_voice_id: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    openai_model: Optional[str] = Form(None),
    custom_llm_url: Optional[str] = Form(None),
    openai_temperature: Optional[float] = Form(None),
    openai_max_tokens: Optional[int] = Form(None),
    system_prompt: Optional[str] = Form(None),
    end_call_message: Optional[str] = Form(None),
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new assistant."""
    # Check if an assistant with this phone number already exists
    existing = await AssistantService.get_assistant_by_phone(db, phone_number)
    if existing:
        # Get available phone numbers again to repopulate the form
        phone_numbers = TwilioService.get_available_phone_numbers()
        return templates.TemplateResponse("assistants/form.html", {
            "request": request,
            "assistant": None,
            "phone_numbers": phone_numbers,
            "error": f"Assistant with phone number {phone_number} already exists"
        }, status_code=400)
    
    # Create assistant data
    assistant_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        "openai_api_key": openai_api_key,
        "deepgram_api_key": deepgram_api_key,
        "elevenlabs_api_key": elevenlabs_api_key,
        "elevenlabs_voice_id": elevenlabs_voice_id,
        "twilio_account_sid": twilio_account_sid,
        "twilio_auth_token": twilio_auth_token,
        "openai_model": openai_model,
        "custom_llm_url": custom_llm_url,
        "openai_temperature": openai_temperature,
        "openai_max_tokens": openai_max_tokens,
        "system_prompt": system_prompt,
        "end_call_message": end_call_message,
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout
    }
    
    # Remove None values
    assistant_data = {k: v for k, v in assistant_data.items() if v is not None}
    
    # Create the assistant
    try:
        new_assistant = await AssistantService.create_assistant(db, assistant_data)
        
        # Reload the assistants cache
        await assistant_manager.load_assistants(db)
        
        return RedirectResponse(url=f"/assistants/{new_assistant.id}", status_code=302)
    except Exception as e:
        # Get available phone numbers again to repopulate the form
        phone_numbers = TwilioService.get_available_phone_numbers()
        return templates.TemplateResponse("assistants/form.html", {
            "request": request,
            "assistant": None,
            "phone_numbers": phone_numbers,
            "error": f"Error creating assistant: {str(e)}"
        }, status_code=500)

@router.get("/assistants/{assistant_id}", response_class=HTMLResponse)
async def view_assistant(request: Request, assistant_id: int, db: Session = Depends(get_db)):
    """View an assistant."""
    assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    # Get recent calls for this assistant
    calls = db.query(Call).filter(Call.assistant_id == assistant_id).order_by(Call.started_at.desc()).limit(10).all()
    
    return templates.TemplateResponse("assistants/view.html", {
        "request": request,
        "assistant": assistant,
        "calls": calls
    })

@router.get("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def edit_assistant_form(request: Request, assistant_id: int, db: Session = Depends(get_db)):
    """Show the edit assistant form."""
    assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    # Get available phone numbers from Twilio
    phone_numbers = TwilioService.get_available_phone_numbers()
    
    return templates.TemplateResponse("assistants/form.html", {
        "request": request,
        "assistant": assistant,
        "phone_numbers": phone_numbers
    })

@router.post("/assistants/{assistant_id}/edit", response_class=HTMLResponse)
async def update_assistant(
    request: Request,
    assistant_id: int,
    name: str = Form(...),
    phone_number: str = Form(...),
    description: Optional[str] = Form(None),
    is_active: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    deepgram_api_key: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    elevenlabs_voice_id: Optional[str] = Form(None),
    twilio_account_sid: Optional[str] = Form(None),
    twilio_auth_token: Optional[str] = Form(None),
    openai_model: Optional[str] = Form(None),
    custom_llm_url: Optional[str] = Form(None),
    openai_temperature: Optional[float] = Form(None),
    openai_max_tokens: Optional[int] = Form(None),
    system_prompt: Optional[str] = Form(None),
    end_call_message: Optional[str] = Form(None),
    max_idle_messages: Optional[int] = Form(None),
    idle_timeout: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Update an assistant."""
    assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    # Check if the phone number is being changed and if it already exists
    if phone_number != assistant.phone_number:
        existing = await AssistantService.get_assistant_by_phone(db, phone_number)
        if existing:
            # Get available phone numbers again to repopulate the form
            phone_numbers = TwilioService.get_available_phone_numbers()
            return templates.TemplateResponse("assistants/form.html", {
                "request": request,
                "assistant": assistant,
                "phone_numbers": phone_numbers,
                "error": f"Assistant with phone number {phone_number} already exists"
            }, status_code=400)
    
    # Create update data
    update_data = {
        "name": name,
        "phone_number": phone_number,
        "description": description,
        "is_active": is_active,
        "openai_api_key": openai_api_key or None,  # Convert empty string to None
        "deepgram_api_key": deepgram_api_key or None,
        "elevenlabs_api_key": elevenlabs_api_key or None,
        "elevenlabs_voice_id": elevenlabs_voice_id or None,
        "twilio_account_sid": twilio_account_sid or None,
        "twilio_auth_token": twilio_auth_token or None,
        "openai_model": openai_model or None,
        "custom_llm_url": custom_llm_url or None,
        "openai_temperature": openai_temperature,
        "openai_max_tokens": openai_max_tokens,
        "system_prompt": system_prompt or None,
        "end_call_message": end_call_message or None,
        "max_idle_messages": max_idle_messages,
        "idle_timeout": idle_timeout
    }
    
    # Update the assistant
    try:
        updated_assistant = await AssistantService.update_assistant(db, assistant_id, update_data)
        
        # Reload the assistants cache
        await assistant_manager.load_assistants(db)
        
        return RedirectResponse(url=f"/assistants/{updated_assistant.id}", status_code=302)
    except Exception as e:
        # Get available phone numbers again to repopulate the form
        phone_numbers = TwilioService.get_available_phone_numbers()
        return templates.TemplateResponse("assistants/form.html", {
            "request": request,
            "assistant": assistant,
            "phone_numbers": phone_numbers,
            "error": f"Error updating assistant: {str(e)}"
        }, status_code=500)

@router.get("/assistants/{assistant_id}/delete")
async def delete_assistant(request: Request, assistant_id: int, db: Session = Depends(get_db)):
    """Delete an assistant."""
    # Check if assistant exists
    assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    
    # Delete the assistant
    await AssistantService.delete_assistant(db, assistant_id)
    
    # Reload the assistants cache
    await assistant_manager.load_assistants(db)
    
    return RedirectResponse(url="/assistants", status_code=302)

# ========== Calls Routes ==========
@router.get("/calls", response_class=HTMLResponse)
async def list_calls(request: Request, db: Session = Depends(get_db)):
    """List all calls."""
    calls = db.query(Call).order_by(Call.started_at.desc()).all()
    return templates.TemplateResponse("calls/index.html", {
        "request": request,
        "calls": calls
    })

@router.get("/calls/{call_id}", response_class=HTMLResponse)
async def view_call(request: Request, call_id: int, db: Session = Depends(get_db)):
    """View a call."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return templates.TemplateResponse("calls/view.html", {
        "request": request,
        "call": call
    }) 