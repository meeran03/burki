"""
Database initialization script.
This script creates a default assistant if none exists.
Run with: python -m app.db.init_db
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.db.database import SessionLocal
from app.services.assistant_service import AssistantService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def create_default_assistant(db: Session):
    """Create a default assistant if none exists."""
    try:
        # Check if any assistants exist - use get_active_assistants since get_assistants requires organization_id
        assistants = await AssistantService.get_active_assistants()
        if assistants:
            logger.info(f"Found {len(assistants)} existing assistants, skipping default creation")
            return
        
        # Create a default assistant
        default_assistant = {
            "name": "Default Assistant",
            "phone_number": os.getenv("TWILIO_PHONE_NUMBER", "+18005551234"),
            "description": "Default voice assistant created during initialization",
            
            # Use environment variables for API keys
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "custom_llm_url": os.getenv("CUSTOM_LLM_URL"),
            "deepgram_api_key": os.getenv("DEEPGRAM_API_KEY"),
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
            
            # Configuration
            "system_prompt": "You are a helpful voice assistant having a phone conversation. Keep your responses concise and natural.",
            "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID"),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            "openai_temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            "openai_max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            
            # Endpointing settings
            "silence_min_duration_ms": int(os.getenv("SILENCE_MIN_DURATION_MS", "500")),
            "energy_threshold": int(os.getenv("ENERGY_THRESHOLD", "50")),
            "wait_after_speech_ms": int(os.getenv("WAIT_AFTER_SPEECH_MS", "700")),
            "no_punctuation_wait_ms": int(os.getenv("NO_PUNCTUATION_WAIT_MS", "300")),
            
            # Interruption settings
            "voice_seconds_threshold": int(os.getenv("VOICE_SECONDS_THRESHOLD", "2")),
            "word_count_threshold": int(os.getenv("WORD_COUNT_THRESHOLD", "5")),
            
            # Call control settings
            "end_call_message": "Thank you for calling. Goodbye!",
            "max_idle_messages": int(os.getenv("MAX_IDLE_MESSAGES", "3")),
            "idle_timeout": int(os.getenv("IDLE_TIMEOUT", "10")),
            
            # Active
            "is_active": True
        }
        
        # Create the assistant with default user_id and organization_id (1 for admin/default)
        # Note: This assumes there's a default organization and user with ID 1
        assistant = await AssistantService.create_assistant(
            default_assistant, 
            user_id=1,  # Default user ID - adjust as needed
            organization_id=1  # Default organization ID - adjust as needed
        )
        logger.info(f"Created default assistant with ID: {assistant.id}")
        
    except Exception as e:
        logger.error(f"Error creating default assistant: {e}", exc_info=True)
        raise

async def init():
    """Initialize the database with sample data."""
    db = SessionLocal()
    try:
        await create_default_assistant(db)
        logger.info("Database initialization complete")
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Initializing database with sample data...")
    asyncio.run(init())