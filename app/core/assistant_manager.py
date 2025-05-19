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
            
            # LLM Settings
            "llm_settings": assistant.llm_settings or {},
            
            # Interruption Settings
            "interruption_settings": assistant.interruption_settings or {},
            
            # TTS Settings
            "tts_settings": assistant.tts_settings or {},
            
            # STT Settings
            "stt_settings": assistant.stt_settings or {},
            
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
        
        # Set LLM settings
        if "llm_settings" in config:
            llm_settings = config["llm_settings"]
            if "model" in llm_settings:
                os.environ["OPENAI_MODEL"] = llm_settings["model"]
            if "temperature" in llm_settings:
                os.environ["OPENAI_TEMPERATURE"] = str(llm_settings["temperature"])
            if "max_tokens" in llm_settings:
                os.environ["OPENAI_MAX_TOKENS"] = str(llm_settings["max_tokens"])
            if "system_prompt" in llm_settings:
                os.environ["SYSTEM_PROMPT"] = llm_settings["system_prompt"]
        
        # Set TTS settings
        if "tts_settings" in config:
            tts_settings = config["tts_settings"]
            if "voice_id" in tts_settings:
                os.environ["ELEVENLABS_VOICE_ID"] = tts_settings["voice_id"]
        
        logger.info(f"Applied configuration for assistant: {assistant.name}")

# Create singleton instance
assistant_manager = AssistantManager() 