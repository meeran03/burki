# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Assistant
from app.db.database import get_async_db_session
from app.twilio.twilio_service import TwilioService
from app.utils.url_utils import get_twiml_webhook_url

logger = logging.getLogger(__name__)


class AssistantService:
    """
    Service class for handling Assistant model operations.
    """

    @staticmethod
    async def create_assistant(assistant_data: Dict[str, Any], user_id: int, organization_id: int) -> Assistant:
        """
        Create a new assistant.

        Args:
            assistant_data: Assistant data
            user_id: ID of the user creating the assistant
            organization_id: ID of the organization

        Returns:
            Assistant: Created assistant
        """
        try:
            async with await get_async_db_session() as db:
                # Add user_id and organization_id to the assistant data
                assistant_data["user_id"] = user_id
                assistant_data["organization_id"] = organization_id
                
                assistant = Assistant(**assistant_data)
                db.add(assistant)
                await db.commit()
                await db.refresh(assistant)
                logger.info(f"Created assistant with ID: {assistant.id} for organization: {organization_id}")

            return assistant
        except SQLAlchemyError as e:
            logger.error(f"Error creating assistant: {e}")
            raise

    @staticmethod
    async def get_assistant_by_id(assistant_id: int, organization_id: int = None) -> Optional[Assistant]:
        """
        Get assistant by ID.

        Args:
            assistant_id: Assistant ID
            organization_id: Organization ID for filtering (optional)

        Returns:
            Optional[Assistant]: Found assistant or None
        """
        async with await get_async_db_session() as db:
            query = select(Assistant).where(Assistant.id == assistant_id)
            
            if organization_id:
                query = query.where(Assistant.organization_id == organization_id)
            
            result = await db.execute(query)
            return result.scalar_one_or_none()

    # This method has been replaced by get_assistant_by_phone_number() 
    # which uses the new PhoneNumber table for lookups

    @staticmethod
    async def get_assistants(
        organization_id: int,
        skip: int = 0, 
        limit: int = 100, 
        active_only: bool = False,
        user_id: int = None
    ) -> List[Assistant]:
        """
        Get a list of assistants for an organization.

        Args:
            organization_id: Organization ID
            skip: Number of assistants to skip
            limit: Maximum number of assistants to return
            active_only: Only return active assistants
            user_id: Filter by user ID (optional)

        Returns:
            List[Assistant]: List of assistants
        """
        async with await get_async_db_session() as db:
            query = select(Assistant).where(Assistant.organization_id == organization_id)

            if active_only:
                query = query.filter(Assistant.is_active == True)
            
            if user_id:
                query = query.filter(Assistant.user_id == user_id)

            query = query.offset(skip).limit(limit).order_by(Assistant.created_at.desc())
            result = await db.execute(query)
            return result.scalars().all()

    @staticmethod
    async def update_assistant(
        assistant_id: int, 
        update_data: Dict[str, Any], 
        organization_id: int
    ) -> Optional[Assistant]:
        """
        Update an assistant.

        Args:
            assistant_id: Assistant ID
            update_data: Data to update
            organization_id: Organization ID for security

        Returns:
            Optional[Assistant]: Updated assistant or None
        """
        try:
            async with await get_async_db_session() as db:
                # Get assistant within the same session, ensuring it belongs to the organization
                query = select(Assistant).where(
                    Assistant.id == assistant_id,
                    Assistant.organization_id == organization_id
                )
                result = await db.execute(query)
                assistant = result.scalar_one_or_none()
                
                if not assistant:
                    return None

                # Update assistant attributes
                for key, value in update_data.items():
                    if hasattr(assistant, key):
                        setattr(assistant, key, value)

                await db.commit()
                await db.refresh(assistant)
                logger.info(f"Updated assistant with ID: {assistant.id} for organization: {organization_id}")

                # Note: Webhook configuration is now handled at the phone number assignment level

                return assistant
        except SQLAlchemyError as e:
            logger.error(f"Error updating assistant: {e}")
            raise

    @staticmethod
    async def delete_assistant(assistant_id: int, organization_id: int) -> bool:
        """
        Delete an assistant.

        Args:
            assistant_id: Assistant ID
            organization_id: Organization ID for security

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            async with await get_async_db_session() as db:
                # Get assistant ensuring it belongs to the organization
                query = select(Assistant).where(
                    Assistant.id == assistant_id,
                    Assistant.organization_id == organization_id
                )
                result = await db.execute(query)
                assistant = result.scalar_one_or_none()
                
                if not assistant:
                    return False

                await db.delete(assistant)
                await db.commit()
                logger.info(f"Deleted assistant with ID: {assistant_id} for organization: {organization_id}")
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error deleting assistant: {e}")
            raise

    @staticmethod
    async def get_active_assistants(organization_id: int = None) -> List[Assistant]:
        """
        Get all active assistants.

        Args:
            organization_id: Organization ID for filtering (optional)

        Returns:
            List[Assistant]: List of active assistants
        """
        async with await get_async_db_session() as db:
            query = select(Assistant).where(Assistant.is_active == True)
            
            if organization_id:
                query = query.where(Assistant.organization_id == organization_id)
            
            result = await db.execute(query)
            return result.scalars().all()

    @staticmethod
    async def count_assistants(organization_id: int, active_only: bool = False) -> int:
        """
        Count assistants for an organization.

        Args:
            organization_id: Organization ID
            active_only: Only count active assistants

        Returns:
            int: Number of assistants
        """
        async with await get_async_db_session() as db:
            query = select(Assistant).where(Assistant.organization_id == organization_id)
            
            if active_only:
                query = query.filter(Assistant.is_active == True)
            
            result = await db.execute(query)
            assistants = result.scalars().all()
            return len(assistants)

    # Note: Webhook configuration is now handled at the phone number assignment level
    # in the PhoneNumberService when assigning phone numbers to assistants

    @staticmethod
    async def get_assistant_by_phone_number(phone_number: str, organization_id: int = None) -> Optional[Assistant]:
        """
        Get assistant assigned to a specific phone number.
        Uses the new PhoneNumber table to find the assigned assistant.
        
        Args:
            phone_number: Phone number to look up
            organization_id: Organization ID for filtering (optional)
            
        Returns:
            Optional[Assistant]: Assistant assigned to the phone number or None
        """
        async with get_async_db_session() as db:
            from sqlalchemy.orm import selectinload
            from app.db.models import PhoneNumber
            
            # Query phone number with assistant relationship
            query = select(PhoneNumber).options(
                selectinload(PhoneNumber.assistant)
            ).where(
                PhoneNumber.phone_number == phone_number,
                PhoneNumber.is_active == True,
                PhoneNumber.assistant_id.is_not(None)  # Must be assigned to an assistant
            )
            
            if organization_id:
                query = query.where(PhoneNumber.organization_id == organization_id)
            
            result = await db.execute(query)
            phone_number_obj = result.scalar_one_or_none()
            
            if phone_number_obj and phone_number_obj.assistant:
                return phone_number_obj.assistant
            
            return None
