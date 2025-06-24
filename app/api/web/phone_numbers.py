"""Phone Number Management Web API."""

import logging
from typing import Optional

from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.api.web.index import require_auth
from app.db.database import get_db
from app.db.models import User, PhoneNumber, Assistant
from app.services.phone_number_service import PhoneNumberService
from app.api.web.index import get_template_context

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/organization/phone-numbers", response_class=HTMLResponse)
async def phone_numbers_page(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Phone numbers management page."""
    try:
        from sqlalchemy.orm import joinedload
        
        # Get phone numbers for the organization (sync version)
        phone_numbers = (
            db.query(PhoneNumber)
            .options(joinedload(PhoneNumber.assistant))
            .filter(
                PhoneNumber.organization_id == current_user.organization_id,
                PhoneNumber.is_active == True
            )
            .order_by(PhoneNumber.phone_number)
            .all()
        )
        
        # Get available assistants for assignment
        available_assistants = (
            db.query(Assistant)
            .filter(
                Assistant.organization_id == current_user.organization_id,
                Assistant.is_active == True
            )
            .all()
        )
        
        # Get available (unassigned) phone numbers
        available_phone_numbers = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.organization_id == current_user.organization_id,
                PhoneNumber.is_active == True,
                PhoneNumber.assistant_id.is_(None)
            )
            .order_by(PhoneNumber.phone_number)
            .all()
        )
        
        return templates.TemplateResponse(
            "phone_numbers/index.html",
            get_template_context(
                request,
                current_user=current_user,
                organization=current_user.organization,
                phone_numbers=phone_numbers,
                available_assistants=available_assistants,
                available_phone_numbers=available_phone_numbers,
                has_twilio_creds=bool(
                    current_user.organization.twilio_account_sid and 
                    current_user.organization.twilio_auth_token
                ),
            ),
        )
    except Exception as e:
        logger.error(f"Error loading phone numbers page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load phone numbers")


@router.post("/organization/phone-numbers/sync", response_class=JSONResponse)
async def sync_phone_numbers(
    request: Request,
    current_user: User = Depends(require_auth),
):
    """Sync phone numbers from Twilio account."""
    try:
        result = await PhoneNumberService.sync_twilio_phone_numbers(
            current_user.organization_id
        )
        
        if result["success"]:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Successfully synced {result['synced_count']} phone numbers",
                    "errors": result.get("errors")
                }
            )
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "error": result["error"]
                },
                status_code=400
            )
            
    except Exception as e:
        logger.error(f"Error syncing phone numbers: {e}")
        return JSONResponse(
            content={"success": False, "error": "Failed to sync phone numbers"},
            status_code=500
        )


@router.post("/organization/phone-numbers/{phone_number_id}/assign")
async def assign_phone_number(
    phone_number_id: int,
    request: Request,
    current_user: User = Depends(require_auth),
    assistant_id: Optional[int] = Form(None),
):
    """Assign or unassign a phone number to/from an assistant."""
    try:
        result = await PhoneNumberService.assign_phone_to_assistant(
            phone_number_id=phone_number_id,
            assistant_id=assistant_id if assistant_id else None,
            organization_id=current_user.organization_id
        )
        
        if result["success"]:
            return RedirectResponse(
                url="/organization/phone-numbers?success=Phone number assignment updated",
                status_code=303
            )
        else:
            return RedirectResponse(
                url=f"/organization/phone-numbers?error={result['error']}",
                status_code=303
            )
            
    except Exception as e:
        logger.error(f"Error assigning phone number: {e}")
        return RedirectResponse(
            url="/organization/phone-numbers?error=Failed to assign phone number",
            status_code=303
        )


@router.post("/organization/phone-numbers/{phone_number_id}/delete")
async def delete_phone_number(
    phone_number_id: int,
    request: Request,
    current_user: User = Depends(require_auth),
):
    """Delete/deactivate a phone number."""
    try:
        result = await PhoneNumberService.delete_phone_number(
            phone_number_id=phone_number_id,
            organization_id=current_user.organization_id
        )
        
        if result["success"]:
            return RedirectResponse(
                url="/organization/phone-numbers?success=Phone number deleted",
                status_code=303
            )
        else:
            return RedirectResponse(
                url=f"/organization/phone-numbers?error={result['error']}",
                status_code=303
            )
            
    except Exception as e:
        logger.error(f"Error deleting phone number: {e}")
        return RedirectResponse(
            url="/organization/phone-numbers?error=Failed to delete phone number",
            status_code=303
        )


@router.get("/organization/settings", response_class=HTMLResponse)
async def organization_settings_page(
    request: Request,
    current_user: User = Depends(require_auth),
):
    """Organization settings page including Twilio configuration."""
    return templates.TemplateResponse(
        "organization/settings.html",
        get_template_context(
            request,
            current_user=current_user,
            organization=current_user.organization,
        ),
    )


@router.post("/organization/settings/twilio")
async def update_twilio_settings(
    request: Request,
    current_user: User = Depends(require_auth),
    twilio_account_sid: str = Form(...),
    twilio_auth_token: str = Form(...),
):
    """Update organization's Twilio credentials."""
    try:
        result = await PhoneNumberService.update_organization_twilio_credentials(
            organization_id=current_user.organization_id,
            account_sid=twilio_account_sid.strip(),
            auth_token=twilio_auth_token.strip()
        )
        
        if result["success"]:
            return RedirectResponse(
                url="/organization/settings?success=Twilio credentials updated successfully",
                status_code=303
            )
        else:
            return RedirectResponse(
                url=f"/organization/settings?error={result['error']}",
                status_code=303
            )
            
    except Exception as e:
        logger.error(f"Error updating Twilio credentials: {e}")
        return RedirectResponse(
            url="/organization/settings?error=Failed to update Twilio credentials",
            status_code=303
        ) 