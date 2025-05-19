import logging
import os
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Assistant
from app.services.assistant_service import AssistantService

logger = logging.getLogger(__name__)

class AssistantManager:
    """
    Manages multiple virtual assistants and their configurations.
    """
    
    def __init__(self):
        """Initialize the assistant manager."""
        self.assistant_cache = {}  # Cache assistants by phone number for quick lookup
        
    async def load_assistants(self, db: Session):
        """
        Load all active assistants from the database.
        
        Args:
            db: Database session
        """
        try:
            # Clear the cache first
            self.assistant_cache = {}
            
            # Load all active assistants
            assistants = await AssistantService.get_active_assistants(db)
            
            # Cache them by phone number
            for assistant in assistants:
                self.assistant_cache[assistant.phone_number] = assistant
                
            logger.info(f"Loaded {len(assistants)} active assistants")
        except Exception as e:
            logger.error(f"Error loading assistants: {e}")
    
    async def get_assistant_by_phone(self, phone_number: str, db: Session) -> Optional[Assistant]:
        """
        Get an assistant by phone number.
        First checks the cache, then the database.
        
        Args:
            phone_number: Phone number to look up
            db: Database session
            
        Returns:
            Optional[Assistant]: Assistant or None
        """
        # Check the cache first
        if phone_number in self.assistant_cache:
            return self.assistant_cache[phone_number]
        
        # Not found in cache, check the database
        assistant = await AssistantService.get_assistant_by_phone(db, phone_number)
        
        # If found and active, add to cache
        if assistant and assistant.is_active:
            self.assistant_cache[phone_number] = assistant
            
        return assistant
    
    def get_assistant_config(self, assistant: Assistant) -> Dict[str, Any]:
        """
        Get the configuration for an assistant.
        
        Args:
            assistant: Assistant model
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Build configuration from assistant model
        config = {
            "id": assistant.id,
            "name": assistant.name,
            "phone_number": assistant.phone_number,
            
            # API Keys (override environment variables)
            "openai_api_key": assistant.openai_api_key,
            "custom_llm_url": assistant.custom_llm_url,
            "deepgram_api_key": assistant.deepgram_api_key,
            "elevenlabs_api_key": assistant.elevenlabs_api_key,
            "twilio_account_sid": assistant.twilio_account_sid,
            "twilio_auth_token": assistant.twilio_auth_token,
            
            # Configuration
            "system_prompt": assistant.system_prompt,
            "elevenlabs_voice_id": assistant.elevenlabs_voice_id,
            "openai_model": assistant.openai_model,
            "openai_temperature": assistant.openai_temperature,
            "openai_max_tokens": assistant.openai_max_tokens,
            
            # Endpointing settings
            "silence_min_duration_ms": assistant.silence_min_duration_ms,
            "energy_threshold": assistant.energy_threshold,
            "wait_after_speech_ms": assistant.wait_after_speech_ms,
            "no_punctuation_wait_ms": assistant.no_punctuation_wait_ms,
            
            # Interruption settings
            "voice_seconds_threshold": assistant.voice_seconds_threshold,
            "word_count_threshold": assistant.word_count_threshold,
            
            # Call control settings
            "end_call_message": assistant.end_call_message,
            "max_idle_messages": assistant.max_idle_messages,
            "idle_timeout": assistant.idle_timeout,
            
            # Custom settings
            "custom_settings": assistant.custom_settings or {}
        }
        
        # Remove None values to use environment defaults
        return {k: v for k, v in config.items() if v is not None}
    
    def apply_assistant_config(self, assistant: Assistant):
        """
        Apply an assistant's configuration to the environment.
        
        Args:
            assistant: Assistant model
        """
        # Apply configuration to environment variables
        config = self.get_assistant_config(assistant)
        
        # Set environment variables for API keys if they exist
        for key in ["openai_api_key", "custom_llm_url", "deepgram_api_key", "elevenlabs_api_key",
                    "twilio_account_sid", "twilio_auth_token"]:
            if key in config and config[key]:
                os.environ[key.upper()] = config[key]
        
        # Set other environment variables
        if "openai_model" in config:
            os.environ["OPENAI_MODEL"] = config["openai_model"]
        if "openai_temperature" in config:
            os.environ["OPENAI_TEMPERATURE"] = str(config["openai_temperature"])
        if "openai_max_tokens" in config:
            os.environ["OPENAI_MAX_TOKENS"] = str(config["openai_max_tokens"])
        if "elevenlabs_voice_id" in config:
            os.environ["ELEVENLABS_VOICE_ID"] = config["elevenlabs_voice_id"]
        
        logger.info(f"Applied configuration for assistant: {assistant.name}")

# Create singleton instance
assistant_manager = AssistantManager() 