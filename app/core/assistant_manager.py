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
        Load all active assistants from the database.

        Args:
        """
        try:
            # Clear the cache first
            self.assistant_cache = {}
            self.assistant_id_cache = {}
            # Load all active assistants
            assistants = await AssistantService.get_active_assistants()

            # Cache them by phone number and ID
            for assistant in assistants:
                self.assistant_cache[assistant.phone_number] = assistant
                self.assistant_id_cache[assistant.id] = assistant

            logger.info(f"Loaded {len(assistants)} active assistants")
        except Exception as e:
            logger.error(f"Error loading assistants: {e}")

    async def get_assistant_by_phone(self, phone_number: str) -> Optional[Assistant]:
        """
        Get an assistant by phone number.
        First checks the cache, then the database.

        Args:
            phone_number: Phone number to look up

        Returns:
            Optional[Assistant]: Assistant or None
        """
        # Check the cache first
        if phone_number in self.assistant_cache:
            return self.assistant_cache[phone_number]

        # Not found in cache, check the database
        assistant = await AssistantService.get_assistant_by_phone(phone_number)

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

        # If found and active, add to cache
        if assistant and assistant.is_active:
            self.assistant_cache[assistant.phone_number] = assistant
            self.assistant_id_cache[assistant.id] = assistant

        return assistant

# Create singleton instance
assistant_manager = AssistantManager()
