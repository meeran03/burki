# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
import time
from datetime import timedelta
import datetime
import os
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import func
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
    """Landing page showcasing Burki Voice AI."""
    return templates.TemplateResponse("landing.html", get_template_context(request))


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Dashboard page with essential analytics."""
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
        # Basic call stats
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

        # Calculate success rate
        success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0

        # Calculate assistant availability
        total_assistants = len(org_assistants)
        active_assistants_count = len(active_assistants)
        assistant_availability = (active_assistants_count / total_assistants * 100) if total_assistants > 0 else 0

        # Get recent calls for the table
        recent_calls = (
            db.query(Call)
            .filter(Call.assistant_id.in_(assistant_ids))
            .order_by(Call.started_at.desc())
            .limit(10)
            .all()
        )

        # Simple daily call volume for the chart (last 7 days)
        now = datetime.datetime.utcnow()
        week_ago = now - datetime.timedelta(days=7)
        daily_calls = (
            db.query(
                func.date(Call.started_at).label('date'),
                func.count().label('count')
            )
            .filter(
                Call.assistant_id.in_(assistant_ids),
                Call.started_at >= week_ago
            )
            .group_by(func.date(Call.started_at))
            .order_by(func.date(Call.started_at))
            .all()
        )

        # Format daily call data
        daily_call_data = []
        current = week_ago.date()
        while current <= now.date():
            count = next((c.count for c in daily_calls if c.date == current), 0)
            daily_call_data.append({
                "date": current.strftime("%a"),
                "count": count
            })
            current += datetime.timedelta(days=1)

    else:
        # Set default values when no assistants exist
        total_calls = active_calls = completed_calls = 0
        success_rate = assistant_availability = 0
        daily_call_data = []
        recent_calls = []

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
            success_rate=round(success_rate, 1),
            assistant_availability=round(assistant_availability, 1),
            recent_calls=recent_calls,
            daily_call_data=daily_call_data,
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
        total_calls = completed_calls = failed_calls = 0

    # Calculate success rate
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0
    
    return templates.TemplateResponse(
        "organization.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
            org_users=org_users,
            active_assistants=active_assistants,
            total_calls=total_calls,
            success_rate=round(success_rate, 1),
        ),
    )
