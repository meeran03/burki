# pylint: disable=logging-fstring-interpolation,bare-except,broad-exception-caught
import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# Assistant schemas
class AssistantBase(BaseModel):
    """Base schema for assistant data."""

    name: str
    # Note: phone_number is now managed separately through PhoneNumber table
    # phone_number: str  # Removed - handled via separate phone number assignment
    description: Optional[str] = None
    
    # LLM Provider Configuration
    llm_provider: Optional[str] = Field(
        default="openai",
        description="LLM provider: openai, anthropic, gemini, xai, groq, custom",
    )
    llm_provider_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "api_key": None,
            "base_url": None,
            "model": "gpt-4o-mini",
            "custom_config": {},
        }
    )
    
    # Legacy API keys (for backward compatibility)
    openai_api_key: Optional[str] = None
    custom_llm_url: Optional[str] = None
    
    # Other service API keys
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    
    # LLM Settings
    llm_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "temperature": 0.5,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant that can answer questions and help with tasks.",
            "welcome_message": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop_sequences": [],
        }
    )
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Interruption Settings
    interruption_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "interruption_threshold": 3,
            "min_speaking_time": 0.5,
            "interruption_cooldown": 2.0,
        }
    )
    
    # TTS Settings
    tts_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "provider": "elevenlabs",
            "voice_id": "rachel",
            "model_id": "turbo",
            "latency": 1,
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "provider_config": {},
        }
    )
    
    # STT Settings
    stt_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "model": "nova-2",
            "language": "en-US",
            "punctuate": True,
            "interim_results": True,
            "endpointing": {"silence_threshold": 500, "min_silence_duration": 500},
            "utterance_end_ms": 1000,
            "vad_turnoff": 500,
            "smart_format": True,
            "keywords": [],
            "keyterms": [],
            "audio_denoising": False,
        }
    )
    
    # Call control settings
    end_call_message: Optional[str] = None
    transfer_call_message: Optional[str] = None
    idle_message: Optional[str] = Field(
        default="Are you still there? I'm here to help if you need anything."
    )
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None

    # Tools configuration
    tools_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled_tools": [],
            "end_call": {"enabled": False, "scenarios": [], "custom_message": None},
            "transfer_call": {
                "enabled": False,
                "scenarios": [],
                "transfer_numbers": [],
                "custom_message": None,
            },
            "custom_tools": [],
        }
    )

    # RAG settings
    rag_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": False,
            "search_limit": 3,
            "similarity_threshold": 0.7,
            "embedding_model": "text-embedding-3-small",
            "chunking_strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "auto_process": True,
            "include_metadata": True,
            "context_window_tokens": 4000,
        }
    )

    # Fallback providers configuration
    llm_fallback_providers: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"enabled": False, "fallbacks": []}
    )
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = True

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"

    @field_validator("llm_provider_config", mode="before")
    @classmethod
    def validate_llm_provider_config(cls, v):
        """Ensure llm_provider_config has required structure."""
        if v is None:
            return {
                "api_key": None,
                "base_url": None,
                "model": "gpt-4o-mini",
                "custom_config": {},
            }
        return v

    @field_validator("tts_settings", mode="before")
    @classmethod
    def validate_tts_settings(cls, v):
        """Ensure TTS settings have proper structure."""
        if v is None:
            return {
                "provider": "elevenlabs",
                "voice_id": "rachel",
                "model_id": "turbo",
                "latency": 1,
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
                "provider_config": {},
            }

        # Ensure provider_config exists
        if "provider_config" not in v:
            v["provider_config"] = {}

        return v

    @field_validator("stt_settings", mode="before")
    @classmethod
    def validate_stt_settings(cls, v):
        """Ensure STT settings have proper structure."""
        if v is None:
            return {
                "model": "nova-2",
                "language": "en-US",
                "punctuate": True,
                "interim_results": True,
                "endpointing": {"silence_threshold": 500, "min_silence_duration": 500},
                "utterance_end_ms": 1000,
                "vad_turnoff": 500,
                "smart_format": True,
                "keywords": [],
                "keyterms": [],
                "audio_denoising": False,
            }

        # Ensure keywords and keyterms are lists
        if "keywords" not in v:
            v["keywords"] = []
        if "keyterms" not in v:
            v["keyterms"] = []

        return v

    @field_validator("tools_settings", mode="before")
    @classmethod
    def validate_tools_settings(cls, v):
        """Ensure tools settings have proper structure."""
        if v is None:
            return {
                "enabled_tools": [],
                "end_call": {"enabled": False, "scenarios": [], "custom_message": None},
                "transfer_call": {
                    "enabled": False,
                    "scenarios": [],
                    "transfer_numbers": [],
                    "custom_message": None,
                },
                "custom_tools": [],
            }

        # Update enabled_tools based on individual tool settings
        enabled_tools = []
        if v.get("end_call", {}).get("enabled", False):
            enabled_tools.append("endCall")
        if v.get("transfer_call", {}).get("enabled", False):
            enabled_tools.append("transferCall")
        v["enabled_tools"] = enabled_tools

        return v

    @field_validator("llm_fallback_providers", mode="before")
    @classmethod
    def validate_fallback_providers(cls, v):
        """Ensure fallback providers have proper structure."""
        if v is None:
            return {"enabled": False, "fallbacks": []}
        return v

    @field_validator("custom_settings", mode="before")
    @classmethod
    def validate_custom_settings(cls, v):
        """Validate custom settings, especially structured data schema."""
        if v is None:
            return None

        # If structured_data_schema is provided, validate it's valid JSON
        if "structured_data_schema" in v:
            schema_data = v["structured_data_schema"]
            if isinstance(schema_data, str):
                try:
                    import json

                    # Parse JSON string to validate it
                    parsed_schema = json.loads(schema_data)
                    v["structured_data_schema"] = parsed_schema
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON schema for structured_data_schema: {str(e)}"
                    )

        return v

    @model_validator(mode="before")
    @classmethod
    def convert_empty_strings_to_none(cls, values):
        """Convert empty strings to None for optional fields."""
        # Handle both dict and ORM object inputs
        if hasattr(values, 'items'):  # Dictionary-like object
            for key, value in values.items():
                if isinstance(value, str) and value.strip() == "":
                    values[key] = None
        elif hasattr(values, '__dict__'):  # ORM object or similar
            # For ORM objects, just return as-is since Pydantic will handle the conversion
            pass
        return values

    @classmethod
    def parse_keywords_string(cls, keywords_str: str) -> List[Dict[str, Any]]:
        """Parse comma-separated keywords string into list of keyword objects."""
        if not keywords_str or not keywords_str.strip():
            return []

        keywords_list = []
        for keyword in keywords_str.split(","):
            keyword = keyword.strip()
            if ":" in keyword:
                # Handle keyword with intensifier (e.g., "OpenAI:2.0")
                word, intensifier = keyword.split(":", 1)
                try:
                    keywords_list.append(
                        {
                            "keyword": word.strip(),
                            "intensifier": float(intensifier.strip()),
                        }
                    )
                except ValueError:
                    raise ValueError(f"Invalid intensifier value in keyword: {keyword}")
            elif keyword:
                # Handle keyword without intensifier (default 1.0)
                keywords_list.append({"keyword": keyword, "intensifier": 1.0})

        return keywords_list

    @classmethod
    def parse_keyterms_string(cls, keyterms_str: str) -> List[str]:
        """Parse comma-separated keyterms string into list."""
        if not keyterms_str or not keyterms_str.strip():
            return []

        return [term.strip() for term in keyterms_str.split(",") if term.strip()]

    @classmethod
    def parse_scenarios_string(cls, scenarios_str: str) -> List[str]:
        """Parse comma-separated scenarios string into list."""
        if not scenarios_str or not scenarios_str.strip():
            return []

        return [s.strip() for s in scenarios_str.split(",") if s.strip()]

    @classmethod
    def parse_numbers_string(cls, numbers_str: str) -> List[str]:
        """Parse comma-separated phone numbers string into list."""
        if not numbers_str or not numbers_str.strip():
            return []

        return [n.strip() for n in numbers_str.split(",") if n.strip()]


class AssistantCreate(AssistantBase):
    """Schema for creating a new assistant."""

    # organization_id and user_id will be set from the authenticated user
    pass


class AssistantUpdate(AssistantBase):
    """Schema for updating an assistant."""

    # Override all fields from AssistantBase to make them optional
    name: Optional[str] = None
    description: Optional[str] = None
    
    # LLM Provider Configuration
    llm_provider: Optional[str] = None
    llm_provider_config: Optional[Dict[str, Any]] = None
    
    # Legacy API keys
    openai_api_key: Optional[str] = None
    custom_llm_url: Optional[str] = None
    
    # Other service API keys
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    
    # LLM Settings
    llm_settings: Optional[Dict[str, Any]] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Interruption Settings
    interruption_settings: Optional[Dict[str, Any]] = None
    
    # TTS Settings
    tts_settings: Optional[Dict[str, Any]] = None
    
    # STT Settings
    stt_settings: Optional[Dict[str, Any]] = None
    
    # Call control settings
    end_call_message: Optional[str] = None
    transfer_call_message: Optional[str] = None
    idle_message: Optional[str] = None
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None

    # Tools configuration
    tools_settings: Optional[Dict[str, Any]] = None

    # RAG settings
    rag_settings: Optional[Dict[str, Any]] = None

    # Fallback providers configuration
    llm_fallback_providers: Optional[Dict[str, Any]] = None
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class AssistantResponse(AssistantBase):
    """Schema for assistant response."""

    id: int
    organization_id: int
    user_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    class Config:
        from_attributes = True


# Call schemas
class CallBase(BaseModel):
    """Base schema for call data."""

    call_sid: str
    to_phone_number: str
    customer_phone_number: str
    call_meta: Optional[Dict[str, Any]] = None  # Changed from 'metadata' to match model


class CallCreate(CallBase):
    """Schema for creating a new call."""

    assistant_id: int


class CallResponse(CallBase):
    """Schema for call response."""

    id: int
    assistant_id: int
    status: str
    duration: Optional[int] = None
    started_at: Optional[datetime.datetime] = None
    ended_at: Optional[datetime.datetime] = None
    
    class Config:
        from_attributes = True


# Transcript schemas
class TranscriptBase(BaseModel):
    """Base schema for transcript data."""

    content: str
    is_final: Optional[bool] = True
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
    created_at: Optional[datetime.datetime] = None
    
    class Config:
        from_attributes = True


# Recording schemas
class RecordingBase(BaseModel):
    """Base schema for recording data."""

    recording_sid: Optional[str] = None
    file_path: Optional[str] = None
    recording_url: Optional[str] = None
    duration: Optional[float] = None
    format: Optional[str] = "wav"
    recording_type: Optional[str] = "full"
    recording_source: Optional[str] = "twilio"
    status: Optional[str] = "recording"


class RecordingCreate(RecordingBase):
    """Schema for creating a new recording."""

    call_id: int


class RecordingResponse(RecordingBase):
    """Schema for recording response."""

    id: int
    call_id: int
    created_at: Optional[datetime.datetime] = None
    
    class Config:
        from_attributes = True


# API Response schemas
class APIResponse(BaseModel):
    """Standard API response format."""

    success: bool
    message: str
    data: Optional[Any] = None


class PaginatedResponse(BaseModel):
    """Paginated response format."""

    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int


# User and Organization schemas for responses
class UserResponse(BaseModel):
    """Schema for user response in API."""

    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: str
    role: str
    is_active: bool
    created_at: datetime.datetime
    
    class Config:
        from_attributes = True


class OrganizationResponse(BaseModel):
    """Schema for organization response in API."""

    id: int
    name: str
    slug: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime.datetime
    
    class Config:
        from_attributes = True


# Phone Number Assignment schemas
class PhoneNumberAssignRequest(BaseModel):
    """Schema for assigning a phone number to an assistant."""

    phone_number: str = Field(..., description="Phone number in E.164 format (e.g., +1234567890)")
    friendly_name: Optional[str] = Field(None, description="Friendly name for the phone number (e.g., 'Customer Support Line')")
    auto_sync: bool = Field(True, description="Automatically sync from Twilio if number not found locally")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v):
        """Basic validation for phone number format."""
        if not v:
            raise ValueError("Phone number is required")
        
        # Remove any whitespace
        v = v.strip()
        
        # Basic E.164 format validation
        if not v.startswith('+'):
            raise ValueError("Phone number must be in E.164 format (start with +)")
        
        # Check if it contains only digits after the +
        if not v[1:].isdigit():
            raise ValueError("Phone number must contain only digits after the + symbol")
        
        # Basic length check (E.164 allows 7-15 digits)
        if len(v[1:]) < 7 or len(v[1:]) > 15:
            raise ValueError("Phone number must be between 7 and 15 digits long")
        
        return v

    @field_validator("friendly_name")
    @classmethod
    def validate_friendly_name(cls, v):
        """Validate friendly name."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            if len(v) > 100:
                raise ValueError("Friendly name must be 100 characters or less")
        return v


class PhoneNumberUnassignRequest(BaseModel):
    """Schema for unassigning a phone number from an assistant."""

    phone_number: str = Field(..., description="Phone number to unassign")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v):
        """Basic validation for phone number format."""
        if not v:
            raise ValueError("Phone number is required")
        
        # Remove any whitespace
        v = v.strip()
        
        # Basic E.164 format validation
        if not v.startswith('+'):
            raise ValueError("Phone number must be in E.164 format (start with +)")
        
        # Check if it contains only digits after the +
        if not v[1:].isdigit():
            raise ValueError("Phone number must contain only digits after the + symbol")
        
        # Basic length check (E.164 allows 7-15 digits)
        if len(v[1:]) < 7 or len(v[1:]) > 15:
            raise ValueError("Phone number must be between 7 and 15 digits long")
        
        return v
