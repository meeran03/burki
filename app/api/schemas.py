from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import datetime

# Assistant schemas
class AssistantBase(BaseModel):
    """Base schema for assistant data."""
    name: str
    phone_number: str
    description: Optional[str] = None
    
    # API keys - these are optional in requests
    openai_api_key: Optional[str] = None
    custom_llm_url: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    
    # LLM Settings
    llm_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant that can answer questions and help with tasks."
        }
    )
    
    # Interruption Settings
    interruption_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "interruption_threshold": 3,
            "min_speaking_time": 0.5,
            "interruption_cooldown": 2.0
        }
    )
    
    # TTS Settings
    tts_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "voice_id": "rachel",
            "model_id": "turbo",
            "latency": 1,
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    )
    
    # STT Settings
    stt_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "model": "nova-2",
            "language": "en-US",
            "punctuate": True,
            "interim_results": True,
            "endpointing": {
                "silence_threshold": 500,
                "min_silence_duration": 500
            },
            "utterance_end_ms": 1000,
            "vad_turnoff": 500,
            "smart_format": True
        }
    )
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Call control settings
    end_call_message: Optional[str] = None
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = True

class AssistantCreate(AssistantBase):
    """Schema for creating a new assistant."""
    pass

class AssistantUpdate(BaseModel):
    """Schema for updating an assistant."""
    name: Optional[str] = None
    phone_number: Optional[str] = None
    description: Optional[str] = None
    
    # API keys
    openai_api_key: Optional[str] = None
    custom_llm_url: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    
    # LLM Settings
    llm_settings: Optional[Dict[str, Any]] = None
    
    # Interruption Settings
    interruption_settings: Optional[Dict[str, Any]] = None
    
    # TTS Settings
    tts_settings: Optional[Dict[str, Any]] = None
    
    # STT Settings
    stt_settings: Optional[Dict[str, Any]] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Call control settings
    end_call_message: Optional[str] = None
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class AssistantResponse(AssistantBase):
    """Schema for assistant response."""
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    class Config:
        orm_mode = True

# Call schemas
class CallBase(BaseModel):
    """Base schema for call data."""
    call_sid: str
    to_phone_number: str
    customer_phone_number: str
    metadata: Optional[Dict[str, Any]] = None

class CallCreate(CallBase):
    """Schema for creating a new call."""
    assistant_id: int

class CallResponse(CallBase):
    """Schema for call response."""
    id: int
    assistant_id: int
    status: str
    duration: Optional[int] = None
    started_at: datetime.datetime
    ended_at: Optional[datetime.datetime] = None
    
    class Config:
        orm_mode = True

# Transcript schemas
class TranscriptBase(BaseModel):
    """Base schema for transcript data."""
    content: str
    is_final: bool = True
    speaker: Optional[str] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    confidence: Optional[float] = None

class TranscriptCreate(TranscriptBase):
    """Schema for creating a new transcript."""
    call_id: int

class TranscriptResponse(TranscriptBase):
    """Schema for transcript response."""
    id: int
    call_id: int
    created_at: datetime.datetime
    
    class Config:
        orm_mode = True

# Recording schemas
class RecordingBase(BaseModel):
    """Base schema for recording data."""
    file_path: str
    format: Optional[str] = "wav"
    recording_type: Optional[str] = "full"
    duration: Optional[float] = None

class RecordingCreate(RecordingBase):
    """Schema for creating a new recording."""
    call_id: int

class RecordingResponse(RecordingBase):
    """Schema for recording response."""
    id: int
    call_id: int
    created_at: datetime.datetime
    
    class Config:
        orm_mode = True 