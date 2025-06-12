from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query, Response
from sqlalchemy.orm import Session

from app.core.auth import get_current_user_flexible, require_api_key
from app.db.database import get_db
from app.db.models import User, UserAPIKey
from app.services.assistant_service import AssistantService
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
    # Check if an assistant with this phone number already exists in the organization
    existing = await AssistantService.get_assistant_by_phone(
        assistant.phone_number, 
        current_user.organization_id
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Assistant with phone number {assistant.phone_number} already exists in your organization",
        )

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
    search: Optional[str] = Query(None, description="Search by name/phone/description"),
    llm_provider: Optional[str] = Query(None, description="Filter by LLM provider"),
    sort_by: str = Query("created", regex="^(name|phone|created)$", description="Sort by field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
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
        user_id=user_id,
        llm_provider=llm_provider,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order
    )
    return assistants


@router.get("/count", response_model=dict)
async def get_assistants_count(
    active_only: bool = Query(False, description="Only count active assistants"),
    llm_provider: Optional[str] = Query(None, description="Filter by LLM provider"),
    search: Optional[str] = Query(None, description="Search term to match"),
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Get the count of assistants in your organization.
    """
    count = await AssistantService.count_assistants(
        organization_id=current_user.organization_id,
        active_only=active_only,
        llm_provider=llm_provider,
        search=search
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
    """
    assistant = await AssistantService.get_assistant_by_phone(
        phone_number, 
        current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with phone number {phone_number} not found in your organization",
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

    # If changing phone number, check if it's already in use within the organization
    if (
        assistant_update.phone_number
        and assistant_update.phone_number != existing.phone_number
    ):
        phone_exists = await AssistantService.get_assistant_by_phone(
            assistant_update.phone_number,
            current_user.organization_id
        )
        if phone_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assistant with phone number {assistant_update.phone_number} already exists in your organization",
            )

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


@router.get("/export", response_class=Response)
async def export_assistants_api(
    format: str = Query("csv", regex="^(csv|json)$", description="Export format"),
    search: Optional[str] = Query(None, description="Search term"),
    llm_provider: Optional[str] = Query(None, description="Filter by LLM provider"),
    active_only: bool = Query(False, description="Only include active assistants"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """Export assistants similar to web route."""
    # Reuse service to fetch all (no pagination)
    assistants = await AssistantService.get_assistants(
        organization_id=current_user.organization_id,
        skip=0,
        limit=10000,
        active_only=active_only,
        llm_provider=llm_provider,
        search=search,
        sort_by="name",
        sort_order="asc",
    )

    if format == "json":
        import json as _json
        content = _json.dumps([{
            "id": a.id,
            "name": a.name,
            "phone_number": a.phone_number,
            "description": a.description,
            "is_active": a.is_active,
            "llm_provider": a.llm_provider,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        } for a in assistants], indent=2)
        return Response(content=content, media_type="application/json", headers={"Content-Disposition": "attachment; filename=assistants.json"})

    # CSV
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "phone_number", "description", "status", "llm_provider", "created_at"])
    for a in assistants:
        writer.writerow([
            a.id,
            a.name,
            a.phone_number,
            a.description or "",
            "active" if a.is_active else "inactive",
            a.llm_provider,
            a.created_at.isoformat() if a.created_at else "",
        ])
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=assistants.csv"})
