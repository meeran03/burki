# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
from typing import Optional

from app.db.models import Assistant
from app.services.assistant_service import AssistantService
from app.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class AssistantManager(metaclass=Singleton):
    """
    Manages multiple virtual assistants and their configurations.
    """

    def __init__(self):
        """Initialize the assistant manager."""
        self.assistant_cache = {}  # Cache assistants by phone number for quick lookup
        self.assistant_id_cache = {}  # Cache assistants by ID for quick lookup

    async def load_assistants(self):
        """
        Load all active assistants from the database and cache by assigned phone numbers.
        """
        try:
            # Clear the cache first
            self.assistant_cache = {}
            self.assistant_id_cache = {}
            
            # Load all active assistants with their phone numbers
            from app.services.phone_number_service import PhoneNumberService
            from app.db.models import PhoneNumber
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            from app.db.database import get_async_db_session
            
            async with await get_async_db_session() as db:
                # Get all active phone numbers with their assigned assistants
                query = select(PhoneNumber).options(
                    selectinload(PhoneNumber.assistant)
                ).where(
                    PhoneNumber.is_active == True,
                    PhoneNumber.assistant_id.is_not(None)
                )
                
                result = await db.execute(query)
                phone_numbers = result.scalars().all()
                
                # Cache assistants by their assigned phone numbers
                cached_assistants = set()
                for phone_number_obj in phone_numbers:
                    if phone_number_obj.assistant and phone_number_obj.assistant.is_active:
                        # Cache by phone number
                        self.assistant_cache[phone_number_obj.phone_number] = phone_number_obj.assistant
                        
                        # Cache by ID (only once per assistant)
                        if phone_number_obj.assistant.id not in cached_assistants:
                            self.assistant_id_cache[phone_number_obj.assistant.id] = phone_number_obj.assistant
                            cached_assistants.add(phone_number_obj.assistant.id)

            logger.info(f"Loaded assistants for {len(self.assistant_cache)} phone numbers and {len(cached_assistants)} unique assistants")
        except Exception as e:
            logger.error(f"Error loading assistants: {e}")

    async def get_assistant_by_phone(self, phone_number: str) -> Optional[Assistant]:
        """
        Get an assistant by phone number.
        Uses the new PhoneNumber table to find assigned assistant.

        Args:
            phone_number: Phone number to look up

        Returns:
            Optional[Assistant]: Assistant assigned to the phone number or None
        """
        # Check the cache first
        if phone_number in self.assistant_cache:
            return self.assistant_cache[phone_number]

        # Not found in cache, check the database using new phone number table
        assistant = await AssistantService.get_assistant_by_phone_number(phone_number)

        # If found and active, add to cache
        if assistant and assistant.is_active:
            self.assistant_cache[phone_number] = assistant
            self.assistant_id_cache[assistant.id] = assistant

        return assistant

    async def get_assistant_by_id(self, assistant_id: int) -> Optional[Assistant]:
        """
        Get an assistant by ID.
        First checks the cache, then the database.

        Args:
            assistant_id: Assistant ID to look up

        Returns:
            Optional[Assistant]: Assistant or None
        """
        # Check the cache first
        if assistant_id in self.assistant_id_cache:
            return self.assistant_id_cache[assistant_id]

        # Not found in cache, check the database
        assistant = await AssistantService.get_assistant_by_id(assistant_id)

        # If found and active, add to ID cache (phone numbers will be cached when looked up by phone)
        if assistant and assistant.is_active:
            self.assistant_id_cache[assistant.id] = assistant

        return assistant

# Create singleton instance
assistant_manager = AssistantManager()
