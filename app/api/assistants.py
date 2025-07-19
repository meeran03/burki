"""
Assistant endpoints
"""

# pylint: disable=logging-fstring-interpolation,bare-except,broad-exception-caught,raise-missing-from
from typing import List, Optional
import logging
from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Depends,
    Query,
    Request,
    File,
    UploadFile,
    Form,
)
from sqlalchemy.orm import Session
from app.core.auth import get_current_user_flexible
from app.db.database import get_db
from app.db.models import User
from app.services.assistant_service import AssistantService
from app.services.phone_number_service import PhoneNumberService
from app.core.assistant_manager import assistant_manager
from app.api.schemas import (
    AssistantCreate,
    AssistantUpdate,
    AssistantResponse,
    APIResponse,
    PhoneNumberAssignRequest,
    PhoneNumberUnassignRequest,
    SyncPhoneNumbersResponse,
    OrganizationPhoneNumberResponse,
    OrganizationAssistantInfo
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/assistants", tags=["assistants"])


@router.post("", response_model=AssistantResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant(
    assistant: AssistantCreate, current_user: User = Depends(get_current_user_flexible)
):
    """
    Create a new assistant.

    Creates a new voice assistant with the specified configuration.
    The assistant will be associated with the authenticated user's organization.
    """
    # Extract user details early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    user_id = current_user.id

    # Note: Phone number validation is now handled separately through the PhoneNumber table
    # This allows for multiple phone numbers per assistant and better management

    assistant_data = assistant.dict(exclude_unset=True)
    if assistant_data["phone_number"]:
        assistant = await AssistantService.get_assistant_by_phone_number(
            phone_number=assistant_data["phone_number"], organization_id=organization_id
        )
        if assistant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assistant with phone number {assistant_data['phone_number']} already exists",
            )

    # Create the assistant
    new_assistant = await AssistantService.create_assistant(
        assistant_data, user_id, organization_id
    )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return new_assistant


@router.get("", response_model=List[AssistantResponse])
async def get_assistants(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of items to return"
    ),
    active_only: bool = Query(False, description="Only return active assistants"),
    my_assistants_only: bool = Query(
        False, description="Only return assistants created by me"
    ),
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Get a list of assistants for your organization.

    Returns all assistants that belong to your organization,
    with optional filtering by status and creator.
    """
    # Extract user details early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    user_id = current_user.id if my_assistants_only else None

    assistants = await AssistantService.get_assistants(
        organization_id=organization_id,
        skip=skip,
        limit=limit,
        active_only=active_only,
        user_id=user_id,
    )
    return assistants


@router.get("/count", response_model=dict)
async def get_assistants_count(
    active_only: bool = Query(False, description="Only count active assistants"),
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Get the count of assistants in your organization.
    """
    # Extract organization_id early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    
    count = await AssistantService.count_assistants(
        organization_id=organization_id, active_only=active_only
    )
    return {"count": count}


@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    assistant_id: int, current_user: User = Depends(get_current_user_flexible)
):
    """
    Get a specific assistant by ID.

    Returns the assistant details if it belongs to your organization.
    """
    # Extract organization_id early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )
    return assistant


@router.get("/by-phone/{phone_number}", response_model=AssistantResponse)
async def get_assistant_by_phone(
    phone_number: str, current_user: User = Depends(get_current_user_flexible)
):
    """
    Get a specific assistant by phone number.

    Returns the assistant details if it belongs to your organization.
    Uses the new PhoneNumber table for lookups.
    """
    # Extract organization_id early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    
    assistant = await AssistantService.get_assistant_by_phone_number(
        phone_number, organization_id
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
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Update an assistant.

    Updates the assistant configuration if it belongs to your organization.
    Only the fields provided in the request will be updated.
    """
    # Extract organization_id early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    user_id = current_user.id
    
    # Check if assistant exists in the organization
    existing = await AssistantService.get_assistant_by_id(
        assistant_id, organization_id
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
        assistant_id, update_data, organization_id
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
    is_active: bool = Query(
        ..., description="Whether to activate or deactivate the assistant"
    ),
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Update the active status of an assistant.

    Quick endpoint to activate or deactivate an assistant.
    """
    updated_assistant = await AssistantService.update_assistant(
        assistant_id, {"is_active": is_active}, current_user.organization_id
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
    assistant_id: int, current_user: User = Depends(get_current_user_flexible)
):
    """
    Delete an assistant.

    Permanently deletes the assistant if it belongs to your organization.
    This action cannot be undone.
    """
    # Check if assistant exists in the organization before deletion
    existing = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    # Delete the assistant
    result = await AssistantService.delete_assistant(
        assistant_id, current_user.organization_id
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete assistant",
        )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return APIResponse(
        success=True, message=f"Assistant {existing.name} deleted successfully"
    )


# Additional endpoints for API key management and organization info


@router.get("/me/organization", response_model=dict)
async def get_my_organization_info(
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Get information about your organization.

    Returns basic organization information for the authenticated user.
    """
    # Extract user details early to avoid DetachedInstanceError
    organization_id = current_user.organization_id
    organization_name = (
        current_user.organization.name if current_user.organization else None
    )
    user_id = current_user.id
    user_email = current_user.email
    user_role = current_user.role
    
    return {
        "organization_id": organization_id,
        "organization_name": organization_name,
        "user_id": user_id,
        "user_email": user_email,
        "user_role": user_role,
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
                "default_model": "gpt-4o-mini",
            },
            "anthropic": {
                "name": "Anthropic",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "default_model": "claude-3-5-sonnet-20241022",
            },
            "gemini": {
                "name": "Google Gemini",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                "default_model": "gemini-1.5-flash",
            },
            "xai": {
                "name": "xAI",
                "models": ["grok-beta"],
                "default_model": "grok-beta",
            },
            "groq": {
                "name": "Groq",
                "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                "default_model": "llama-3.1-70b-versatile",
            },
        }
    }


# ========== Phone Number Management Endpoints ==========


@router.get("/{assistant_id}/phone-numbers", response_model=List[dict])
async def get_assistant_phone_numbers(
    assistant_id: int, current_user: User = Depends(get_current_user_flexible)
):
    """
    Get all phone numbers assigned to a specific assistant.

    Returns a list of phone numbers currently assigned to the assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
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
            "assigned_at": pn.updated_at.isoformat() if pn.updated_at else None,
        }
        for pn in phone_numbers
        if pn.assistant_id == assistant_id
    ]

    return assigned_numbers


@router.post(
    "/{assistant_id}/phone-numbers/{phone_number_id}/assign", response_model=APIResponse
)
async def assign_phone_number_to_assistant(
    assistant_id: int,
    phone_number_id: int,
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Assign a phone number to an assistant.

    Assigns the specified phone number to the assistant.
    The phone number must belong to your organization and not be assigned to another assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    try:
        # Assign the phone number
        result = await PhoneNumberService.assign_phone_to_assistant(
            phone_number_id, assistant_id, current_user.organization_id
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number successfully assigned to assistant '{assistant.name}'",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign phone number: {str(e)}",
        )


@router.post(
    "/{assistant_id}/phone-numbers/{phone_number_id}/unassign",
    response_model=APIResponse,
)
async def unassign_phone_number_from_assistant(
    assistant_id: int,
    phone_number_id: int,
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Unassign a phone number from an assistant.

    Removes the phone number assignment from the assistant.
    The phone number will remain in your organization but won't route calls to any assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
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
            current_user.organization_id,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number successfully unassigned from assistant '{assistant.name}'",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign phone number: {str(e)}",
        )


@router.get("/{assistant_id}/available-phone-numbers", response_model=List[dict])
async def get_available_phone_numbers_for_assistant(
    assistant_id: int, current_user: User = Depends(get_current_user_flexible)
):
    """
    Get all unassigned phone numbers that can be assigned to an assistant.

    Returns phone numbers in your organization that are not currently assigned to any assistant.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
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
            "capabilities": pn.capabilities,
        }
        for pn in available_numbers
    ]


# ========== Bulk Phone Number Assignment ==========


@router.post("/{assistant_id}/phone-numbers/bulk-assign", response_model=APIResponse)
async def bulk_assign_phone_numbers(
    assistant_id: int,
    phone_number_ids: List[int],
    current_user: User = Depends(get_current_user_flexible),
):
    """
    Assign multiple phone numbers to an assistant at once.

    Assigns all specified phone numbers to the assistant.
    Returns success if all assignments succeed, or details about failures.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    if not phone_number_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No phone number IDs provided",
        )

    successes = []
    failures = []

    for phone_number_id in phone_number_ids:
        try:
            result = await PhoneNumberService.assign_phone_to_assistant(
                phone_number_id, assistant_id, current_user.organization_id
            )

            if result["success"]:
                successes.append(phone_number_id)
            else:
                failures.append(
                    {"phone_number_id": phone_number_id, "error": result["message"]}
                )

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
                "failed_count": len(failures),
            },
        )
    else:
        return APIResponse(
            success=True,
            message=f"Successfully assigned all {len(successes)} phone numbers to assistant '{assistant.name}'",
            data={"successes": successes, "total_assigned": len(successes)},
        )


# ========== RAG (Document Management) Routes ==========

@router.post("/{assistant_id}/documents/upload")
async def upload_document(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible),
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """Upload a document to an assistant's knowledge base."""
    try:
        # Verify assistant ownership
        assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check if RAG is available
        try:
            from app.services.rag_service import RAGService
        except ImportError:
            raise HTTPException(status_code=500, detail="RAG functionality not available")

        # Validate file
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Read file data
        file_data = await file.read()

        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Initialize RAG service
        rag_service = RAGService.create_default_instance()

        # Upload and process document
        document = await rag_service.upload_and_process_document(
            file_data=file_data,
            filename=file.filename,
            content_type=file.content_type,
            assistant_id=assistant_id,
            organization_id=current_user.organization_id,
            name=name or file.filename,
            category=category,
            tags=tag_list,
        )

        return {
            "success": True,
            "document": {
                "id": document.id,
                "name": document.name,
                "filename": document.original_filename,
                "status": document.processing_status,
                "created_at": document.created_at.isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Error uploading document")


@router.get("/{assistant_id}/documents")
async def list_documents(
    assistant_id: int,
    current_user: User = Depends(get_current_user_flexible),
):
    """List documents for an assistant."""
    try:
        # Verify assistant ownership
        assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check if RAG is available
        try:
            from app.services.rag_service import RAGService
        except ImportError:
            return {"documents": []}

        # Initialize RAG service
        rag_service = RAGService.create_default_instance()

        # Get documents
        documents = await rag_service.get_assistant_documents(assistant_id, include_processing=True)

        return {
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "filename": doc.original_filename,
                    "content_type": doc.content_type,
                    "file_size": doc.file_size,
                    "processing_status": doc.processing_status,
                    "processing_error": doc.processing_error,
                    "total_chunks": doc.total_chunks,
                    "processed_chunks": doc.processed_chunks,
                    "progress_percentage": doc.get_processing_progress(),
                    "category": doc.category,
                    "tags": doc.tags,
                    "created_at": doc.created_at.isoformat(),
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                }
                for doc in documents
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")


@router.delete("/{assistant_id}/documents/{document_id}")
async def delete_document(
    assistant_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user_flexible),
):
    """Delete a document from an assistant's knowledge base."""
    try:
        # Verify assistant ownership
        assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check if RAG is available
        try:
            from app.services.rag_service import RAGService
        except ImportError:
            raise HTTPException(status_code=500, detail="RAG functionality not available")

        # Initialize RAG service
        rag_service = RAGService.create_default_instance()

        # Delete document
        success = await rag_service.delete_document(document_id, assistant_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"success": True, "message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Error deleting document")


@router.get("/{assistant_id}/documents/{document_id}/status")
async def get_document_status(
    assistant_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user_flexible),
):
    """Get processing status of a document."""
    try:
        # Verify assistant ownership
        assistant = await AssistantService.get_assistant_by_id(assistant_id, current_user.organization_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check if RAG is available
        try:
            from app.services.rag_service import RAGService
        except ImportError:
            raise HTTPException(status_code=500, detail="RAG functionality not available")

        # Initialize RAG service
        rag_service = RAGService.create_default_instance()

        # Get document status
        status = await rag_service.get_document_status(document_id)

        if not status:
            raise HTTPException(status_code=404, detail="Document not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(status_code=500, detail="Error getting document status")


# ========== Smart Phone Number Assignment ==========

@router.post("/{assistant_id}/phone-numbers/assign-by-number", response_model=APIResponse)
async def assign_phone_number_by_string(
    assistant_id: int,
    request: PhoneNumberAssignRequest,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Assign a phone number to an assistant by phone number string.
    
    Auto-syncs from Twilio if the number doesn't exist locally and auto_sync is True.
    This is the recommended way to assign newly purchased Twilio numbers.
    
    Args:
        assistant_id: ID of the assistant to assign the phone number to
        request: JSON request containing phone_number, friendly_name (optional), and auto_sync
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    try:
        # Extract values from request
        phone_number = request.phone_number
        friendly_name = request.friendly_name
        auto_sync = request.auto_sync
        
        # Normalize phone number format (remove spaces, ensure + prefix)
        normalized_number = phone_number.strip()
        if not normalized_number.startswith('+'):
            # Try to add + if it looks like an international number
            if normalized_number.startswith('1') and len(normalized_number) == 11:
                normalized_number = '+' + normalized_number
            elif len(normalized_number) == 10:
                normalized_number = '+1' + normalized_number
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Phone number must be in E.164 format (e.g., +1234567890)"
                )

        # Try to find the phone number in our database first
        from app.services.phone_number_service import PhoneNumberService
        phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
            current_user.organization_id
        )
        
        existing_phone = None
        for pn in phone_numbers:
            if pn.phone_number == normalized_number:
                existing_phone = pn
                break

        # If not found locally and auto_sync is enabled, sync from Twilio
        if not existing_phone and auto_sync:
            # Sync all phone numbers from Twilio
            sync_result = await PhoneNumberService.sync_twilio_phone_numbers(
                current_user.organization_id
            )
            
            if not sync_result.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to sync from Twilio: {sync_result.get('error', 'Unknown error')}"
                )
            
            # Try to find the phone number again after sync
            phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
                current_user.organization_id
            )
            
            for pn in phone_numbers:
                if pn.phone_number == normalized_number:
                    existing_phone = pn
                    break

        # If still not found, return error
        if not existing_phone:
            error_msg = f"Phone number {normalized_number} not found"
            if auto_sync:
                error_msg += " in your Twilio account or local database after sync attempt"
            else:
                error_msg += " in local database. Try setting auto_sync=true or sync manually first"
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )

        # Check if phone number is already assigned to another assistant
        if existing_phone.assistant_id and existing_phone.assistant_id != assistant_id:
            # Get the other assistant's name for a better error message
            other_assistant = await AssistantService.get_assistant_by_id(
                existing_phone.assistant_id, current_user.organization_id
            )
            other_name = other_assistant.name if other_assistant else f"Assistant ID {existing_phone.assistant_id}"
            
            return APIResponse(
                success=False,
                message=f"Phone number {normalized_number} is already assigned to {other_name}",
                data={
                    "phone_number": normalized_number,
                    "phone_number_id": existing_phone.id,
                    "assistant_id": assistant_id,
                    "assistant_name": assistant.name,
                    "friendly_name": friendly_name,
                    "was_synced": not existing_phone or auto_sync
                }
            )


        # Assign the phone number
        result = await PhoneNumberService.assign_phone_to_assistant(
            existing_phone.id, assistant_id, current_user.organization_id, friendly_name
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=result.get("error", "Failed to assign phone number")
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number {normalized_number} successfully assigned to assistant '{assistant.name}'",
            data={
                "phone_number": normalized_number,
                "phone_number_id": existing_phone.id,
                "assistant_id": assistant_id,
                "assistant_name": assistant.name,
                "friendly_name": friendly_name,
                "was_synced": not existing_phone or auto_sync
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign phone number: {str(e)}"
        )


@router.post("/{assistant_id}/phone-numbers/unassign-by-number", response_model=APIResponse)
async def unassign_phone_number_by_string(
    assistant_id: int,
    request: PhoneNumberUnassignRequest,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Unassign a phone number from an assistant by phone number string.
    """
    # Verify assistant exists and belongs to organization
    assistant = await AssistantService.get_assistant_by_id(
        assistant_id, current_user.organization_id
    )
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found in your organization",
        )

    try:
        # Extract phone number from request
        phone_number = request.phone_number
        
        # Normalize phone number format
        normalized_number = phone_number.strip()
        if not normalized_number.startswith('+'):
            if normalized_number.startswith('1') and len(normalized_number) == 11:
                normalized_number = '+' + normalized_number
            elif len(normalized_number) == 10:
                normalized_number = '+1' + normalized_number

        # Find the phone number in our database
        from app.services.phone_number_service import PhoneNumberService
        phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
            current_user.organization_id
        )
        
        existing_phone = None
        for pn in phone_numbers:
            if pn.phone_number == normalized_number:
                existing_phone = pn
                break

        if not existing_phone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Phone number {normalized_number} not found in your organization"
            )

        # Check if it's actually assigned to this assistant
        if existing_phone.assistant_id != assistant_id:
            if existing_phone.assistant_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Phone number {normalized_number} is not assigned to this assistant"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Phone number {normalized_number} is not assigned to any assistant"
                )

        # Unassign the phone number
        result = await PhoneNumberService.assign_phone_to_assistant(
            existing_phone.id, None, current_user.organization_id
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=result.get("error", "Failed to unassign phone number")
            )

        # Reload assistants cache to update routing
        await assistant_manager.load_assistants()

        return APIResponse(
            success=True,
            message=f"Phone number {normalized_number} successfully unassigned from assistant '{assistant.name}'",
            data={
                "phone_number": normalized_number,
                "phone_number_id": existing_phone.id,
                "assistant_id": assistant_id,
                "assistant_name": assistant.name
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign phone number: {str(e)}"
        )


# ========== Organization Phone Number Management ==========

@router.post("/organization/phone-numbers/sync", response_model=SyncPhoneNumbersResponse)
async def sync_organization_phone_numbers(
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Sync phone numbers from Twilio to local database.
    
    This will import any new phone numbers from your Twilio account
    that aren't already in the local database.
    """
    try:
        from app.services.phone_number_service import PhoneNumberService
        
        result = await PhoneNumberService.sync_twilio_phone_numbers(
            current_user.organization_id
        )

        if result["success"]:
            return SyncPhoneNumbersResponse(
                success=True,
                message=f"Successfully synced {result.get('synced_count', 0)} phone numbers from Twilio",
                synced_count=result.get('synced_count', 0)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to sync phone numbers")
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync phone numbers: {str(e)}"
        )


@router.get("/organization/phone-numbers", response_model=List[OrganizationPhoneNumberResponse])
async def list_organization_phone_numbers(
    include_assigned: bool = Query(True, description="Include phone numbers assigned to assistants"),
    include_unassigned: bool = Query(True, description="Include unassigned phone numbers"),
    current_user: User = Depends(get_current_user_flexible)
):
    """
    List all phone numbers in your organization.
    
    Returns phone numbers with their assignment status and assistant information.
    """
    try:
        from app.services.phone_number_service import PhoneNumberService
        
        # Get all phone numbers for the organization
        phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
            current_user.organization_id
        )

        # Filter based on assignment status
        filtered_numbers = []
        for pn in phone_numbers:
            is_assigned = pn.assistant_id is not None
            
            if (is_assigned and include_assigned) or (not is_assigned and include_unassigned):
                # Get assistant info if assigned
                assistant_info = None
                if pn.assistant_id:
                    assistant = await AssistantService.get_assistant_by_id(
                        pn.assistant_id, current_user.organization_id
                    )
                    if assistant:
                        assistant_info = OrganizationAssistantInfo(
                            id=assistant.id,
                            name=assistant.name,
                            is_active=assistant.is_active
                        )

                filtered_numbers.append(OrganizationPhoneNumberResponse(
                    id=pn.id,
                    phone_number=pn.phone_number,
                    friendly_name=pn.friendly_name,
                    twilio_sid=pn.twilio_sid,
                    is_active=pn.is_active,
                    capabilities=pn.capabilities,
                    assistant=assistant_info,
                    created_at=pn.created_at,
                    updated_at=pn.updated_at
                ))

        return filtered_numbers

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list phone numbers: {str(e)}"
        )


@router.get("/organization/phone-numbers/available", response_model=List[OrganizationPhoneNumberResponse])
async def list_available_phone_numbers(
    current_user: User = Depends(get_current_user_flexible)
):
    """
    List phone numbers available for assignment (not assigned to any assistant).
    """
    try:
        from app.services.phone_number_service import PhoneNumberService
        
        available_numbers = await PhoneNumberService.get_available_phone_numbers(
            current_user.organization_id
        )

        return [
            OrganizationPhoneNumberResponse(
                id=pn.id,
                phone_number=pn.phone_number,
                friendly_name=pn.friendly_name,
                twilio_sid=pn.twilio_sid,
                is_active=pn.is_active,
                capabilities=pn.capabilities,
                assistant=None,
                created_at=pn.created_at,
                updated_at=pn.updated_at
            )
            for pn in available_numbers
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list available phone numbers: {str(e)}"
        )
