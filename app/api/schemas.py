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
    
    # Configuration
    system_prompt: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    openai_model: Optional[str] = None
    openai_temperature: Optional[float] = None
    openai_max_tokens: Optional[int] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Endpointing settings
    silence_min_duration_ms: Optional[int] = None
    energy_threshold: Optional[int] = None
    wait_after_speech_ms: Optional[int] = None
    no_punctuation_wait_ms: Optional[int] = None
    
    # Interruption settings
    voice_seconds_threshold: Optional[int] = None
    word_count_threshold: Optional[int] = None
    
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
    
    # Configuration
    system_prompt: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    openai_model: Optional[str] = None
    openai_temperature: Optional[float] = None
    openai_max_tokens: Optional[int] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Endpointing settings
    silence_min_duration_ms: Optional[int] = None
    energy_threshold: Optional[int] = None
    wait_after_speech_ms: Optional[int] = None
    no_punctuation_wait_ms: Optional[int] = None
    
    # Interruption settings
    voice_seconds_threshold: Optional[int] = None
    word_count_threshold: Optional[int] = None
    
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