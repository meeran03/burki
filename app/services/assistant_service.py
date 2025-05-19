import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Assistant
from app.twilio.twilio_service import TwilioService
from app.utils.url_utils import get_twiml_webhook_url

logger = logging.getLogger(__name__)

class AssistantService:
    """
    Service class for handling Assistant model operations.
    """
    
    @staticmethod
    async def create_assistant(db: Session, assistant_data: Dict[str, Any]) -> Assistant:
        """
        Create a new assistant.
        
        Args:
            db: Database session
            assistant_data: Assistant data
            
        Returns:
            Assistant: Created assistant
        """
        try:
            assistant = Assistant(**assistant_data)
            db.add(assistant)
            db.commit()
            db.refresh(assistant)
            logger.info(f"Created assistant with ID: {assistant.id}")
            
            # Configure Twilio webhook for the phone number
            await AssistantService._configure_twilio_webhook(assistant)
            
            return assistant
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating assistant: {e}")
            raise
    
    @staticmethod
    async def get_assistant_by_id(db: Session, assistant_id: int) -> Optional[Assistant]:
        """
        Get assistant by ID.
        
        Args:
            db: Database session
            assistant_id: Assistant ID
            
        Returns:
            Optional[Assistant]: Found assistant or None
        """
        return db.query(Assistant).filter(Assistant.id == assistant_id).first()
    
    @staticmethod
    async def get_assistant_by_phone(db: Session, phone_number: str) -> Optional[Assistant]:
        """
        Get assistant by phone number.
        
        Args:
            db: Database session
            phone_number: Assistant phone number
            
        Returns:
            Optional[Assistant]: Found assistant or None
        """
        return db.query(Assistant).filter(Assistant.phone_number == phone_number).first()
    
    @staticmethod
    async def get_assistants(db: Session, skip: int = 0, limit: int = 100, active_only: bool = False) -> List[Assistant]:
        """
        Get a list of assistants.
        
        Args:
            db: Database session
            skip: Number of assistants to skip
            limit: Maximum number of assistants to return
            active_only: Only return active assistants
            
        Returns:
            List[Assistant]: List of assistants
        """
        query = db.query(Assistant)
        
        if active_only:
            query = query.filter(Assistant.is_active == True)
            
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    async def update_assistant(db: Session, assistant_id: int, update_data: Dict[str, Any]) -> Optional[Assistant]:
        """
        Update an assistant.
        
        Args:
            db: Database session
            assistant_id: Assistant ID
            update_data: Data to update
            
        Returns:
            Optional[Assistant]: Updated assistant or None
        """
        try:
            assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
            if not assistant:
                return None
            
            # Check if phone number is being updated
            phone_number_changed = "phone_number" in update_data and update_data["phone_number"] != assistant.phone_number
                
            # Update assistant attributes
            for key, value in update_data.items():
                if hasattr(assistant, key):
                    setattr(assistant, key, value)
            
            db.commit()
            db.refresh(assistant)
            logger.info(f"Updated assistant with ID: {assistant.id}")
            
            # If phone number changed or status changed to active, update Twilio webhook
            if phone_number_changed or ("is_active" in update_data and update_data["is_active"]):
                await AssistantService._configure_twilio_webhook(assistant)
            
            return assistant
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating assistant: {e}")
            raise
    
    @staticmethod
    async def delete_assistant(db: Session, assistant_id: int) -> bool:
        """
        Delete an assistant.
        
        Args:
            db: Database session
            assistant_id: Assistant ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            assistant = await AssistantService.get_assistant_by_id(db, assistant_id)
            if not assistant:
                return False
            
            db.delete(assistant)
            db.commit()
            logger.info(f"Deleted assistant with ID: {assistant_id}")
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error deleting assistant: {e}")
            raise
    
    @staticmethod
    async def get_active_assistants(db: Session) -> List[Assistant]:
        """
        Get all active assistants.
        
        Args:
            db: Database session
            
        Returns:
            List[Assistant]: List of active assistants
        """
        return await AssistantService.get_assistants(db, active_only=True)
    
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
        
        # Get the webhook URL
        webhook_url = get_twiml_webhook_url()
        logger.info(f"Configuring Twilio webhook for {assistant.phone_number} to {webhook_url}")
        
        # Use assistant-specific Twilio credentials if available
        account_sid = assistant.twilio_account_sid
        auth_token = assistant.twilio_auth_token
        
        # Update the webhook
        success = TwilioService.update_phone_webhook(
            phone_number=assistant.phone_number,
            webhook_url=webhook_url,
            account_sid=account_sid, 
            auth_token=auth_token
        )
        
        if success:
            logger.info(f"Successfully configured Twilio webhook for {assistant.phone_number}")
        else:
            logger.warning(f"Failed to configure Twilio webhook for {assistant.phone_number}")
        
        return success 