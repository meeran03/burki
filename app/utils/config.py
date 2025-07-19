import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for the application.
    Loads settings from environment variables with defaults.
    
    Note: Assistant-specific settings should be stored in the Assistant model.
    These settings are used as fallbacks when no assistant is specified.
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Server settings
        self.port = int(os.getenv("PORT", 5000))
        self.host = os.getenv("HOST", "0.0.0.0")
        
        # API Keys (used as fallbacks only)
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.custom_llm_url = os.getenv("CUSTOM_LLM_URL")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Default LLM settings (used when no assistant-specific settings exist)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
        
        # Call control settings
        self.idle_timeout = int(os.getenv("IDLE_TIMEOUT", "30"))  # Time in seconds before considering the call idle
        self.max_idle_messages = int(os.getenv("MAX_IDLE_MESSAGES", "3"))  # Maximum number of idle messages to send before ending the call
        self.end_call_message = os.getenv("END_CALL_MESSAGE", "Thank you for calling. Goodbye!")    

        # Google Analytics
        self.GOOGLE_ANALYTICS_ID = os.getenv("GOOGLE_ANALYTICS_ID")
        self.GOOGLE_SITE_VERIFICATION = os.getenv("GOOGLE_SITE_VERIFICATION")
        
        # Validate required config
        self._validate()
    
    def _validate(self):
        """Validate required configuration values."""
        warnings = []
        
        if not self.twilio_account_sid:
            warnings.append("TWILIO_ACCOUNT_SID not set - call control features will be limited")
        
        if not self.twilio_auth_token:
            warnings.append("TWILIO_AUTH_TOKEN not set - call control features will be limited")
        
        if not self.deepgram_api_key:
            warnings.append("DEEPGRAM_API_KEY not set - speech recognition will be simulated")
        
        if not self.openai_api_key and not self.custom_llm_url:
            warnings.append("Neither OPENAI_API_KEY nor CUSTOM_LLM_URL set - response generation will be simulated")
        
        if not self.elevenlabs_api_key:
            warnings.append("ELEVENLABS_API_KEY not set - speech synthesis will be simulated")
        
        for warning in warnings:
            logger.warning(warning)
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the OpenAI conversation.
        
        Returns:
            str: System prompt
        """
        return os.getenv("SYSTEM_PROMPT", (
            "You are a helpful voice assistant having a phone conversation. "
            "Keep your responses concise but natural, as if speaking on the phone. "
            "Speak in a conversational tone, and provide helpful information. "
            "If you don't know something, simply say so rather than making up information."
        ))
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "server": {
                "port": self.port,
                "host": self.host
            },
            "twilio": {
                "account_sid": self.twilio_account_sid,
                "phone_number": self.twilio_phone_number
            },
            "deepgram": {
                "api_key_set": bool(self.deepgram_api_key)
            },
            "llm": {
                "openai_api_key_set": bool(self.openai_api_key),
                "custom_llm_url_set": bool(self.custom_llm_url),
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens
            },
            "elevenlabs": {
                "api_key_set": bool(self.elevenlabs_api_key),
                "voice_id": self.elevenlabs_voice_id
            },
            "call_control": {
                "idle_timeout": self.idle_timeout,
                "max_idle_messages": self.max_idle_messages,
                "end_call_message": self.end_call_message
            }
        }


# Create a singleton instance
config = Config() 