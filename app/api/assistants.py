from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session

from app.core.auth import get_current_user_flexible, require_api_key
from app.db.database import get_db
from app.db.models import User, UserAPIKey
from app.services.assistant_service import AssistantService
from app.services.phone_number_service import PhoneNumberService
from app.core.assistant_manager import assistant_manager
from app.api.schemas import (
    AssistantCreate, 
    AssistantUpdate, 
    AssistantResponse, 
    APIResponse,
    PaginatedResponse
)

router = APIRouter(prefix="/api/v1/assistants", tags=["assistants"])


@router.post("/", response_model=AssistantResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant(
    assistant: AssistantCreate,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Create a new assistant.
    
    Creates a new voice assistant with the specified configuration.
    The assistant will be associated with the authenticated user's organization.
    """
    # Note: Phone number validation is now handled separately through the PhoneNumber table
    # This allows for multiple phone numbers per assistant and better management

    # Create the assistant
    assistant_data = assistant.dict(exclude_unset=True)
    new_assistant = await AssistantService.create_assistant(
        assistant_data, 
        current_user.id, 
        current_user.organization_id
    )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return new_assistant


@router.get("/", response_model=List[AssistantResponse])
async def get_assistants(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    active_only: bool = Query(False, description="Only return active assistants"),
    my_assistants_only: bool = Query(False, description="Only return assistants created by me"),
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get a list of assistants for your organization.
    
    Returns all assistants that belong to your organization,
    with optional filtering by status and creator.
    """
    user_id = current_user.id if my_assistants_only else None
    
    assistants = await AssistantService.get_assistants(
        organization_id=current_user.organization_id,
        skip=skip, 
        limit=limit, 
        active_only=active_only,
        user_id=user_id
    )
    return assistants


@router.get("/count", response_model=dict)
async def get_assistants_count(
    active_only: bool = Query(False, description="Only count active assistants"),
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get the count of assistants in your organization.
    """
    count = await AssistantService.count_assistants(
        organization_id=current_user.organization_id,
        active_only=active_only
    )
    return {"count": count}


@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get a specific assistant by ID.
    
    Returns the assistant details if it belongs to your organization.
    """
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )
    return assistant


@router.get("/by-phone/{phone_number}", response_model=AssistantResponse)
async def get_assistant_by_phone(
    phone_number: str,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get a specific assistant by phone number.
    
    Returns the assistant details if it belongs to your organization.
    Uses the new PhoneNumber table for lookups.
    """
    assistant = await AssistantService.get_assistant_by_phone_number(
        phone_number, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant assigned to phone number {phone_number} not found in your organization",
        )
    return assistant


@router.put("/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(
    assistant_id: int, 
    assistant_update: AssistantUpdate,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Update an assistant.
    
    Updates the assistant configuration if it belongs to your organization.
    Only the fields provided in the request will be updated.
    """
    # Check if assistant exists in the organization
    existing = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Note: Phone number management is now handled separately through the PhoneNumber table
    # Phone numbers are assigned to assistants through the /organization/phone-numbers interface

    # Update the assistant
    update_data = {k: v for k, v in assistant_update.dict().items() if v is not None}
    updated_assistant = await AssistantService.update_assistant(
        assistant_id, 
        update_data,
        current_user.organization_id
    )

    if not updated_assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return updated_assistant


@router.patch("/{assistant_id}/status", response_model=AssistantResponse)
async def update_assistant_status(
    assistant_id: int,
    is_active: bool = Query(..., description="Whether to activate or deactivate the assistant"),
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Update the active status of an assistant.
    
    Quick endpoint to activate or deactivate an assistant.
    """
    updated_assistant = await AssistantService.update_assistant(
        assistant_id,
        {"is_active": is_active},
        current_user.organization_id
    )

    if not updated_assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return updated_assistant


@router.delete("/{assistant_id}", response_model=APIResponse)
async def delete_assistant(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Delete an assistant.
    
    Permanently deletes the assistant if it belongs to your organization.
    This action cannot be undone.
    """
    # Check if assistant exists in the organization before deletion
    existing = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Delete the assistant
    result = await AssistantService.delete_assistant(
        assistant_id, 
        current_user.organization_id
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete assistant",
        )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return APIResponse(
        success=True,
        message=f"Assistant {existing.name} deleted successfully"
    )


# Additional endpoints for API key management and organization info

@router.get("/me/organization", response_model=dict)
async def get_my_organization_info(
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get information about your organization.
    
    Returns basic organization information for the authenticated user.
    """
    return {
        "organization_id": current_user.organization_id,
        "organization_name": current_user.organization.name if current_user.organization else None,
        "user_id": current_user.id,
        "user_email": current_user.email,
        "user_role": current_user.role,
    }


@router.get("/providers", response_model=dict)
async def get_supported_llm_providers():
    """
    Get list of supported LLM providers.
    
    Returns the available LLM providers and their default configurations.
    """
    return {
        "providers": {
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "default_model": "gpt-4o-mini"
            },
            "anthropic": {
                "name": "Anthropic",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "default_model": "claude-3-5-sonnet-20241022"
            },
            "gemini": {
                "name": "Google Gemini",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                "default_model": "gemini-1.5-flash"
            },
            "xai": {
                "name": "xAI",
                "models": ["grok-beta"],
                "default_model": "grok-beta"
            },
            "groq": {
                "name": "Groq",
                "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                "default_model": "llama-3.1-70b-versatile"
            }
        }
    }


# ========== Phone Number Management Endpoints ==========

@router.get("/{assistant_id}/phone-numbers", response_model=List[dict])
async def get_assistant_phone_numbers(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get all phone numbers assigned to a specific assistant.
    
    Returns a list of phone numbers currently assigned to the assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Get phone numbers for this assistant
    phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
        current_user.organization_id
    )
    
    # Filter to only assigned phone numbers for this assistant
    assigned_numbers = [
        {
            "id": pn.id,
            "phone_number": pn.phone_number,
            "friendly_name": pn.friendly_name,
            "twilio_sid": pn.twilio_sid,
            "is_active": pn.is_active,
            "capabilities": pn.capabilities,
            "assigned_at": pn.updated_at.isoformat() if pn.updated_at else None
        }
        for pn in phone_numbers 
        if pn.assistant_id == assistant_id
    ]
    
    return assigned_numbers


@router.post("/{assistant_id}/phone-numbers/{phone_number_id}/assign", response_model=APIResponse)
async def assign_phone_number_to_assistant(
    assistant_id: int,
    phone_number_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Assign a phone number to an assistant.
    
    Assigns the specified phone number to the assistant.
    The phone number must belong to your organization and not be assigned to another assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    try:
        # Assign the phone number
        result = await PhoneNumberService.assign_phone_to_assistant(
            phone_number_id, 
            assistant_id, 
            current_user.organization_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number successfully assigned to assistant '{assistant.name}'"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign phone number: {str(e)}"
        )


@router.post("/{assistant_id}/phone-numbers/{phone_number_id}/unassign", response_model=APIResponse)
async def unassign_phone_number_from_assistant(
    assistant_id: int,
    phone_number_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Unassign a phone number from an assistant.
    
    Removes the phone number assignment from the assistant.
    The phone number will remain in your organization but won't route calls to any assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    try:
        # Unassign the phone number (assign to None)
        result = await PhoneNumberService.assign_phone_to_assistant(
            phone_number_id, 
            None,  # Unassign by setting assistant_id to None
            current_user.organization_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number successfully unassigned from assistant '{assistant.name}'"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign phone number: {str(e)}"
        )


@router.get("/{assistant_id}/available-phone-numbers", response_model=List[dict])
async def get_available_phone_numbers_for_assistant(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get all unassigned phone numbers that can be assigned to an assistant.
    
    Returns phone numbers in your organization that are not currently assigned to any assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Get available phone numbers
    available_numbers = await PhoneNumberService.get_available_phone_numbers(
        current_user.organization_id
    )
    
    return [
        {
            "id": pn.id,
            "phone_number": pn.phone_number,
            "friendly_name": pn.friendly_name,
            "twilio_sid": pn.twilio_sid,
            "is_active": pn.is_active,
            "capabilities": pn.capabilities
        }
        for pn in available_numbers
    ]


# ========== Bulk Phone Number Assignment ==========

@router.post("/{assistant_id}/phone-numbers/bulk-assign", response_model=APIResponse)
async def bulk_assign_phone_numbers(
    assistant_id: int,
    phone_number_ids: List[int],
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Assign multiple phone numbers to an assistant at once.
    
    Assigns all specified phone numbers to the assistant.
    Returns success if all assignments succeed, or details about failures.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    if not phone_number_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No phone number IDs provided"
        )

    successes = []
    failures = []

    for phone_number_id in phone_number_ids:
        try:
            result = await PhoneNumberService.assign_phone_to_assistant(
                phone_number_id, 
                assistant_id, 
                current_user.organization_id
            )
            
            if result["success"]:
                successes.append(phone_number_id)
            else:
                failures.append({"phone_number_id": phone_number_id, "error": result["message"]})
        
        except Exception as e:
            failures.append({"phone_number_id": phone_number_id, "error": str(e)})

    # Reload assistants cache to update routing
    await assistant_manager.load_assistants()

    if failures:
        return APIResponse(
            success=len(successes) > 0,
            message=f"Assigned {len(successes)} phone numbers successfully. {len(failures)} failed.",
            data={
                "successes": successes,
                "failures": failures,
                "total_requested": len(phone_number_ids),
                "successful_count": len(successes),
                "failed_count": len(failures)
            }
        )
    else:
        return APIResponse(
            success=True,
            message=f"Successfully assigned all {len(successes)} phone numbers to assistant '{assistant.name}'",
            data={
                "successes": successes,
                "total_assigned": len(successes)
            }
        )
