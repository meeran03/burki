"""
Authentication routes
"""

# pylint: disable=logging-fstring-interpolation, broad-exception-caught
from typing import Optional
import time
import logging
import os
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.db.database import get_db
from app.db.models import (
    User,
)
from app.services.auth_service import AuthService, APIKeyService
from app.utils.config import config

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


# ========== Authentication Helpers ==========


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


def create_access_token(user_id: int) -> str:
    """Create a JWT access token for the user."""
    payload = {"user_id": user_id}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def verify_google_token(token: str) -> dict:
    """Verify Google OAuth token and return user info."""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return idinfo
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google token")


# ========== Authentication Routes ==========


@router.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Display login page."""
    # Check if user is already logged in
    user_id = request.session.get("user_id")
    if user_id:
        return RedirectResponse(url="/dashboard", status_code=302)
        
    return templates.TemplateResponse(
        "auth/login.html",
        get_template_context(request, error=error, google_client_id=GOOGLE_CLIENT_ID),
    )


@router.post("/auth/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    organization: str = Form(...),
    remember_me: bool = Form(False),
):
    """Handle manual login."""
    try:
        user = await auth_service.authenticate_user(email, password, organization)
        if not user:
            return templates.TemplateResponse(
                "auth/login.html",
                get_template_context(
                    request,
                    error="Invalid email, password, or organization",
                    google_client_id=GOOGLE_CLIENT_ID,
                ),
                status_code=400,
            )

        # Ensure names are properly split
        user.split_full_name_if_needed()

        # Get organization data (fallback in case eager loading didn't work)
        org_name = (
            getattr(user.organization, "name", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )
        org_slug = (
            getattr(user.organization, "slug", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )

        # If organization data is missing, fetch it separately
        if not org_name or not org_slug:
            organization = await AuthService.get_organization_by_id(
                user.organization_id
            )
            if organization:
                org_name = organization.name
                org_slug = organization.slug

        # Set session with complete user and organization data
        request.session["user_id"] = user.id
        request.session["organization_id"] = user.organization_id
        request.session["user_email"] = user.email
        request.session["user_first_name"] = user.first_name or ""
        request.session["user_last_name"] = user.last_name or ""
        request.session["organization_name"] = org_name
        request.session["organization_slug"] = org_slug

        # Redirect to dashboard
        return RedirectResponse(url="/dashboard", status_code=302)

    except Exception as e:
        logging.error(f"Login error: {e}")
        return templates.TemplateResponse(
            "auth/login.html",
            get_template_context(
                request,
                error="An error occurred during login",
                google_client_id=GOOGLE_CLIENT_ID,
            ),
            status_code=500,
        )


@router.post("/auth/google")
async def google_login(
    request: Request, credential: str = Form(...), organization: str = Form(...)
):
    """Handle Google OAuth login."""
    try:
        # Verify the Google token
        google_user = verify_google_token(credential)

        # Authenticate or create user
        user = await AuthService.authenticate_google_user(
            google_id=google_user["sub"],
            email=google_user["email"],
            full_name=google_user["name"],
            avatar_url=google_user.get("picture", ""),
            organization_slug=organization,
        )

        # Ensure names are properly split
        user.split_full_name_if_needed()

        # Get organization data (fallback in case eager loading didn't work)
        org_name = (
            getattr(user.organization, "name", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )
        org_slug = (
            getattr(user.organization, "slug", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )

        # If organization data is missing, fetch it separately
        if not org_name or not org_slug:
            organization = await AuthService.get_organization_by_id(
                user.organization_id
            )
            if organization:
                org_name = organization.name
                org_slug = organization.slug

        # Set session with complete user and organization data
        request.session["user_id"] = user.id
        request.session["organization_id"] = user.organization_id
        request.session["user_email"] = user.email
        request.session["user_first_name"] = user.first_name or ""
        request.session["user_last_name"] = user.last_name or ""
        request.session["organization_name"] = org_name
        request.session["organization_slug"] = org_slug

        # Redirect to dashboard
        return RedirectResponse(url="/dashboard", status_code=302)

    except HTTPException as e:
        return templates.TemplateResponse(
            "auth/login.html",
            get_template_context(
                request, error=e.detail, google_client_id=GOOGLE_CLIENT_ID
            ),
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Google login error: {e}")
        return templates.TemplateResponse(
            "auth/login.html",
            get_template_context(
                request,
                error="An error occurred during Google login",
                google_client_id=GOOGLE_CLIENT_ID,
            ),
            status_code=500,
        )


@router.get("/auth/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None, success: str = None):
    """Display registration page."""
    # Check if user is already logged in
    user_id = request.session.get("user_id")
    if user_id:
        return RedirectResponse(url="/dashboard", status_code=302)
        
    return templates.TemplateResponse(
        "auth/register.html",
        get_template_context(
            request, error=error, success=success, google_client_id=GOOGLE_CLIENT_ID
        ),
    )


@router.post("/auth/register")
async def register(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    organization: str = Form(...),
    organization_name: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    agree_terms: bool = Form(...),
):
    """Handle manual registration."""
    
    # Check if this is an AJAX request (expects JSON response)
    is_ajax = request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded") and \
              "XMLHttpRequest" in request.headers.get("x-requested-with", "") or \
              request.headers.get("accept", "").find("application/json") != -1
    
    try:
        # Validate passwords match
        if password != confirm_password:
            error_msg = "Passwords do not match"
            if is_ajax:
                return JSONResponse({"success": False, "error": error_msg}, status_code=400)
            return templates.TemplateResponse(
                "auth/register.html",
                get_template_context(
                    request,
                    error=error_msg,
                    google_client_id=GOOGLE_CLIENT_ID,
                ),
                status_code=400,
            )

        if not agree_terms:
            error_msg = "You must agree to the terms of service"
            if is_ajax:
                return JSONResponse({"success": False, "error": error_msg}, status_code=400)
            return templates.TemplateResponse(
                "auth/register.html",
                get_template_context(
                    request,
                    error=error_msg,
                    google_client_id=GOOGLE_CLIENT_ID,
                ),
                status_code=400,
            )

        # Create organization first
        org = await AuthService.create_organization(
            name=organization_name, slug=organization
        )

        # Create user
        user = await AuthService.create_user(
            organization_id=org.id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=password,
            role="admin",  # First user in org becomes admin
        )

        success_msg = "Account created successfully! You can now sign in."
        if is_ajax:
            return JSONResponse({
                "success": True, 
                "message": success_msg,
                "redirect": "/auth/login"
            })
        
        return templates.TemplateResponse(
            "auth/register.html",
            get_template_context(
                request,
                success=success_msg,
                google_client_id=GOOGLE_CLIENT_ID,
            ),
        )

    except HTTPException as e:
        if is_ajax:
            return JSONResponse({"success": False, "error": e.detail}, status_code=400)
        return templates.TemplateResponse(
            "auth/register.html",
            get_template_context(
                request, error=e.detail, google_client_id=GOOGLE_CLIENT_ID
            ),
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Registration error: {e}")
        error_msg = "An error occurred during registration"
        if is_ajax:
            return JSONResponse({"success": False, "error": error_msg}, status_code=500)
        return templates.TemplateResponse(
            "auth/register.html",
            get_template_context(
                request,
                error=error_msg,
                google_client_id=GOOGLE_CLIENT_ID,
            ),
            status_code=500,
        )


@router.post("/auth/google-register")
async def google_register(
    request: Request,
    credential: str = Form(...),
    organization: str = Form(...),
    organization_name: str = Form(...),
):
    """Handle Google OAuth registration."""
    try:
        # Verify the Google token
        google_user = verify_google_token(credential)

        # Create organization first
        org = await AuthService.create_organization(
            name=organization_name, slug=organization
        )

        # Create user
        user = await AuthService.create_user(
            organization_id=org.id,
            email=google_user["email"],
            full_name=google_user["name"],
            google_id=google_user["sub"],
            avatar_url=google_user.get("picture", ""),
            role="admin",  # First user in org becomes admin
            is_verified=True,  # Google users are auto-verified
        )

        # Ensure names are properly split
        user.split_full_name_if_needed()

        # Get organization data (fallback in case eager loading didn't work)
        org_name = (
            getattr(user.organization, "name", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )
        org_slug = (
            getattr(user.organization, "slug", "")
            if hasattr(user, "organization") and user.organization
            else ""
        )

        # If organization data is missing, fetch it separately
        if not org_name or not org_slug:

            organization = await AuthService.get_organization_by_id(
                user.organization_id
            )
            if organization:
                org_name = organization.name
                org_slug = organization.slug

        # Set session with complete user and organization data
        request.session["user_id"] = user.id
        request.session["organization_id"] = user.organization_id
        request.session["user_email"] = user.email
        request.session["user_first_name"] = user.first_name or ""
        request.session["user_last_name"] = user.last_name or ""
        request.session["organization_name"] = org_name
        request.session["organization_slug"] = org_slug

        # Redirect to dashboard
        return RedirectResponse(url="/dashboard", status_code=302)

    except HTTPException as e:
        return templates.TemplateResponse(
            "auth/register.html",
            get_template_context(
                request, error=e.detail, google_client_id=GOOGLE_CLIENT_ID
            ),
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Google registration error: {e}")
        return templates.TemplateResponse(
            "auth/register.html",
            get_template_context(
                request,
                error="An error occurred during registration",
                google_client_id=GOOGLE_CLIENT_ID,
            ),
            status_code=500,
        )


@router.get("/auth/logout")
async def logout(request: Request):
    """Handle logout."""
    request.session.clear()
    return RedirectResponse(url="/auth/login", status_code=302)


@router.get("/auth/api-keys", response_class=HTMLResponse)
async def api_keys_page(
    request: Request,
    current_user: User = Depends(require_auth),
    success: str = None,
    error: str = None,
    new_key: str = None,
    new_key_name: str = None,
):
    """Display API keys management page."""
    api_keys = await APIKeyService.get_user_api_keys(current_user.id)

    return templates.TemplateResponse(
        "auth/api_keys.html",
        get_template_context(
            request,
            api_keys=api_keys,
            success=success,
            error=error,
            new_key=new_key,
            new_key_name=new_key_name,
        ),
    )


@router.post("/auth/api-keys/create")
async def create_api_key(
    request: Request,
    current_user: User = Depends(require_auth),
    name: str = Form(...),
    permissions: list[str] = Form([]),
):
    """Create a new API key."""
    try:
        # Process permissions
        perms = {
            "read": "read" in permissions,
            "write": "write" in permissions,
            "admin": "admin" in permissions,
        }

        # Create the API key
        api_key_record, plain_key = await APIKeyService.create_api_key(
            current_user.id, name
        )

        # Update permissions
        api_key_record.permissions = perms

        # Redirect with the new key (this will show it once)
        return RedirectResponse(
            url=f"/auth/api-keys?new_key={plain_key}&new_key_name={name}",
            status_code=302,
        )

    except Exception as e:
        logging.error(f"API key creation error: {e}")
        return RedirectResponse(
            url="/auth/api-keys?error=Failed to create API key", status_code=302
        )


@router.post("/auth/api-keys/{key_id}/delete")
async def delete_api_key(
    request: Request, key_id: int, current_user: User = Depends(require_auth)
):
    """Delete an API key."""
    try:
        success = await APIKeyService.delete_api_key(current_user.id, key_id)
        if success:
            return RedirectResponse(
                url="/auth/api-keys?success=API key deleted successfully",
                status_code=302,
            )
        else:
            return RedirectResponse(
                url="/auth/api-keys?error=API key not found", status_code=302
            )
    except Exception as e:
        logging.error(f"API key deletion error: {e}")
        return RedirectResponse(
            url="/auth/api-keys?error=Failed to delete API key", status_code=302
        )
