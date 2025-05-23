# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
import time
from datetime import timedelta
import os
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import (
    Call,
    Transcript,
    Assistant,
    User,
)
from app.services.auth_service import AuthService

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


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page showcasing Buraaq Voice AI."""
    return templates.TemplateResponse("landing.html", get_template_context(request))


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Dashboard page with advanced analytics."""
    # Get active assistants for this organization
    active_assistants = (
        db.query(Assistant)
        .filter(
            Assistant.organization_id == current_user.organization_id,
            Assistant.is_active == True,
        )
        .all()
    )

    # Get call statistics for this organization
    org_assistants = (
        db.query(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
        .all()
    )
    assistant_ids = [a.id for a in org_assistants]

    if assistant_ids:
        total_calls = (
            db.query(Call).filter(Call.assistant_id.in_(assistant_ids)).count()
        )
        active_calls = (
            db.query(Call)
            .filter(Call.assistant_id.in_(assistant_ids), Call.status == "ongoing")
            .count()
        )
        completed_calls = (
            db.query(Call)
            .filter(Call.assistant_id.in_(assistant_ids), Call.status == "completed")
            .count()
        )
        failed_calls = (
            db.query(Call)
            .filter(
                Call.assistant_id.in_(assistant_ids),
                Call.status.in_(["failed", "no-answer", "busy"]),
            )
            .count()
        )
    else:
        total_calls = active_calls = completed_calls = failed_calls = 0

    # Calculate success rate
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0

    # Calculate average call duration for completed calls
    if assistant_ids:
        completed_calls_with_duration = (
            db.query(Call)
            .filter(
                Call.assistant_id.in_(assistant_ids),
                Call.status == "completed",
                Call.duration.isnot(None),
            )
            .all()
        )
    else:
        completed_calls_with_duration = []

    avg_duration = 0
    if completed_calls_with_duration:
        total_duration = sum(call.duration for call in completed_calls_with_duration)
        avg_duration = total_duration / len(completed_calls_with_duration)

    # Get transcript quality metrics (average confidence)
    if assistant_ids:
        transcript_confidence = (
            db.query(Transcript.confidence)
            .filter(
                Transcript.call_id.in_(
                    db.query(Call.id).filter(Call.assistant_id.in_(assistant_ids))
                ),
                Transcript.confidence.isnot(None),
            )
            .all()
        )
    else:
        transcript_confidence = []

    avg_quality = 0
    if transcript_confidence:
        avg_quality = (
            sum(conf[0] for conf in transcript_confidence)
            / len(transcript_confidence)
            * 100
        )

    # Get recent calls with enhanced data
    if assistant_ids:
        recent_calls = (
            db.query(Call)
            .filter(Call.assistant_id.in_(assistant_ids))
            .order_by(Call.started_at.desc())
            .limit(10)
            .all()
        )
    else:
        recent_calls = []

    # Calculate assistant performance metrics
    assistant_metrics = []
    for assistant in active_assistants:
        assistant_calls = (
            db.query(Call).filter(Call.assistant_id == assistant.id).count()
        )
        assistant_metrics.append(
            {
                "assistant": assistant,
                "call_count": assistant_calls,
                "success_rate": 95 + (assistant.id % 10),  # Simulated for demo
            }
        )

    # Sort assistants by performance
    assistant_metrics.sort(key=lambda x: x["call_count"], reverse=True)

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
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
            active_assistants=active_assistants,
            total_calls=total_calls,
            active_calls=active_calls,
            completed_calls=completed_calls,
            failed_calls=failed_calls,
            success_rate=round(success_rate, 1),
            avg_duration=round(avg_duration),
            avg_quality=round(avg_quality, 1),
            recent_calls=recent_calls,
            assistant_metrics=assistant_metrics,
            hourly_data=hourly_data,
            uptime=uptime,
        ),
    )


@router.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """User profile page."""
    return templates.TemplateResponse(
        "profile.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
        ),
    )


@router.get("/organization", response_class=HTMLResponse)
async def organization_page(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Organization settings page."""
    # Get all users in the organization
    org_users = (
        db.query(User)
        .filter(User.organization_id == current_user.organization_id)
        .all()
    )
    
    return templates.TemplateResponse(
        "organization.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
            org_users=org_users,
        ),
    )
