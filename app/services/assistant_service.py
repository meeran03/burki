# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Assistant
from app.db.database import get_async_db_session
from app.twilio.twilio_service import TwilioService
from app.utils.url_utils import get_twiml_webhook_url, get_sms_webhook_url

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

            # Configure Twilio webhook for the phone number
            await AssistantService._configure_twilio_webhook(assistant)

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

    @staticmethod
    async def get_assistant_by_phone(phone_number: str, organization_id: int = None) -> Optional[Assistant]:
        """
        Get assistant by phone number.

        Args:
            phone_number: Assistant phone number
            organization_id: Organization ID for filtering (optional)

        Returns:
            Optional[Assistant]: Found assistant or None
        """
        async with await get_async_db_session() as db:
            query = select(Assistant).where(Assistant.phone_number == phone_number)
            
            if organization_id:
                query = query.where(Assistant.organization_id == organization_id)
            
            result = await db.execute(query)
            return result.scalar_one_or_none()

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

                # Check if phone number is being updated
                phone_number_changed = (
                    "phone_number" in update_data
                    and update_data["phone_number"] != assistant.phone_number
                )

                # Update assistant attributes
                for key, value in update_data.items():
                    if hasattr(assistant, key):
                        setattr(assistant, key, value)

                await db.commit()
                await db.refresh(assistant)
                logger.info(f"Updated assistant with ID: {assistant.id} for organization: {organization_id}")

                # If phone number changed or status changed to active, update Twilio webhook
                if phone_number_changed or (
                    "is_active" in update_data and update_data["is_active"]
                ):
                    await AssistantService._configure_twilio_webhook(assistant)

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

    @staticmethod
    async def _configure_twilio_webhook(assistant: Assistant) -> bool:
        """
        Configure Twilio webhook for an assistant's phone number.

        Args:
            assistant: Assistant model

        Returns:
            bool: True if successful, False otherwise
        """
        if not assistant or not assistant.phone_number or not assistant.is_active:
            return False

        # Get the webhook URLs
        voice_webhook_url = get_twiml_webhook_url()
        sms_webhook_url = get_sms_webhook_url()
        
        logger.info(
            f"Configuring Twilio webhooks for {assistant.phone_number} - "
            f"Voice: {voice_webhook_url}, SMS: {sms_webhook_url}"
        )

        # Use assistant-specific Twil`io credentials if available
        account_sid = assistant.twilio_account_sid
        auth_token = assistant.twilio_auth_token

        # Update both voice and SMS webhooks
        success = TwilioService.update_all_webhooks(
            phone_number=assistant.phone_number,
            voice_webhook_url=voice_webhook_url,
            sms_webhook_url=sms_webhook_url,
            account_sid=account_sid,
            auth_token=auth_token,
        )

        if success:
            logger.info(
                f"Successfully configured Twilio voice and SMS webhooks for {assistant.phone_number}"
            )
        else:
            logger.warning(
                f"Failed to configure Twilio webhooks for {assistant.phone_number}"
            )

        return success
