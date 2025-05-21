from typing import List
from fastapi import APIRouter, HTTPException, status
from app.services.assistant_service import AssistantService
from app.core.assistant_manager import assistant_manager
from app.api.schemas import AssistantCreate, AssistantUpdate, AssistantResponse

router = APIRouter(prefix="/assistants", tags=["assistants"])


@router.post("/", response_model=AssistantResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant(assistant: AssistantCreate):
    """
    Create a new assistant.
    """
    # Check if an assistant with this phone number already exists
    existing = await AssistantService.get_assistant_by_phone(assistant.phone_number)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Assistant with phone number {assistant.phone_number} already exists",
        )

    # Create the assistant
    new_assistant = await AssistantService.create_assistant(assistant.dict())

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return new_assistant


@router.get("/", response_model=List[AssistantResponse])
async def get_assistants(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = False,
):
    """
    Get a list of assistants.
    """
    assistants = await AssistantService.get_assistants(
        skip=skip, limit=limit, active_only=active_only
    )
    return assistants


@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(assistant_id: int):
    """
    Get a specific assistant by ID.
    """
    assistant = await AssistantService.get_assistant_by_id(assistant_id)
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found",
        )
    return assistant


@router.get("/by-phone/{phone_number}", response_model=AssistantResponse)
async def get_assistant_by_phone(phone_number: str):
    """
    Get a specific assistant by phone number.
    """
    assistant = await AssistantService.get_assistant_by_phone(phone_number)
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with phone number {phone_number} not found",
        )
    return assistant


@router.put("/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(assistant_id: int, assistant_update: AssistantUpdate):
    """
    Update an assistant.
    """
    # Check if assistant exists
    existing = await AssistantService.get_assistant_by_id(assistant_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found",
        )

    # If changing phone number, check if it's already in use
    if (
        assistant_update.phone_number
        and assistant_update.phone_number != existing.phone_number
    ):
        phone_exists = await AssistantService.get_assistant_by_phone(
            assistant_update.phone_number
        )
        if phone_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assistant with phone number {assistant_update.phone_number} already exists",
            )

    # Update the assistant
    update_data = {k: v for k, v in assistant_update.dict().items() if v is not None}
    updated_assistant = await AssistantService.update_assistant(
        assistant_id, update_data
    )

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return updated_assistant


@router.delete("/{assistant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assistant(assistant_id: int):
    """
    Delete an assistant.
    """
    # Check if assistant exists
    existing = await AssistantService.get_assistant_by_id(assistant_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assistant with ID {assistant_id} not found",
        )

    # Delete the assistant
    result = await AssistantService.delete_assistant(assistant_id)

    # Reload the assistants cache
    await assistant_manager.load_assistants()

    return None
