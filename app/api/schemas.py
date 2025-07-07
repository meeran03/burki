# pylint: disable=logging-fstring-interpolation,bare-except,broad-exception-caught
import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# Nested Configuration Schemas
class LLMProviderConfig(BaseModel):
    """Configuration for the LLM provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    custom_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class LLMSettings(BaseModel):
    """Settings for the Large Language Model."""
    temperature: float = 0.5
    max_tokens: int = 1000
    system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks."
    welcome_message: Optional[str] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = Field(default_factory=list)

class InterruptionSettings(BaseModel):
    """Settings for call interruption behavior."""
    interruption_threshold: int = 3
    min_speaking_time: float = 0.5
    interruption_cooldown: float = 2.0

class TTSSettings(BaseModel):
    """Settings for Text-to-Speech (TTS) service."""
    provider: str = "elevenlabs"
    voice_id: str = "rachel"
    model_id: Optional[str] = "turbo"
    latency: int = 1
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    provider_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class STTEndpointingSettings(BaseModel):
    """Endpointing settings for Speech-to-Text (STT)."""
    silence_threshold: int = 500
    min_silence_duration: int = 500

class Keyword(BaseModel):
    """A keyword to be detected in STT, with an optional intensifier."""
    keyword: str
    intensifier: float = 1.0

class STTSettings(BaseModel):
    """Settings for Speech-to-Text (STT) service."""
    model: str = "nova-2"
    language: str = "en-US"
    punctuate: bool = True
    interim_results: bool = True
    endpointing: STTEndpointingSettings = Field(default_factory=STTEndpointingSettings)
    utterance_end_ms: int = 1000
    vad_turnoff: int = 500
    smart_format: bool = True
    keywords: List[Keyword] = Field(default_factory=list)
    keyterms: List[str] = Field(default_factory=list)
    audio_denoising: bool = False

class EndCallTool(BaseModel):
    """Configuration for the 'end call' tool."""
    enabled: bool = False
    scenarios: List[str] = Field(default_factory=list)
    custom_message: Optional[str] = None

class TransferCallTool(BaseModel):
    """Configuration for the 'transfer call' tool."""
    enabled: bool = False
    scenarios: List[str] = Field(default_factory=list)
    transfer_numbers: List[str] = Field(default_factory=list)
    custom_message: Optional[str] = None

class ToolsSettings(BaseModel):
    """Settings for integrated tools."""
    enabled_tools: List[str] = Field(default_factory=list, description="List of enabled tool names. This is auto-populated based on other settings.")
    end_call: EndCallTool = Field(default_factory=EndCallTool)
    transfer_call: TransferCallTool = Field(default_factory=TransferCallTool)
    custom_tools: List[Dict[str, Any]] = Field(default_factory=list)

class RAGSettings(BaseModel):
    """Settings for Retrieval-Augmented Generation (RAG)."""
    enabled: bool = False
    search_limit: int = 3
    similarity_threshold: float = 0.7
    embedding_model: str = "text-embedding-3-small"
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    auto_process: bool = True
    include_metadata: bool = True
    context_window_tokens: int = 4000

class LLMFallbackProvider(BaseModel):
    """Configuration for a single LLM fallback provider."""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None

class LLMFallbackSettings(BaseModel):
    """Settings for LLM fallback providers."""
    enabled: bool = False
    fallbacks: List[LLMFallbackProvider] = Field(default_factory=list)

class TwilioConfig(BaseModel):
    """Twilio account configuration."""
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None


# Assistant schemas
class AssistantBase(BaseModel):
    """Base schema for assistant data."""

    name: str
    description: Optional[str] = None
    
    # LLM Provider Configuration
    llm_provider: Optional[str] = Field(
        default="openai",
        description="LLM provider: openai, anthropic, gemini, xai, groq, custom",
    )
    llm_provider_config: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="Configuration for the selected LLM provider."
    )
    
    # Legacy API keys (for backward compatibility, will be moved to nested configs)
    openai_api_key: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use llm_provider_config.api_key instead.")
    custom_llm_url: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use llm_provider_config.base_url instead.")
    
    # Other service API keys (also for backward compatibility)
    deepgram_api_key: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use tts_settings.provider_config or stt_settings for provider-specific keys.")
    elevenlabs_api_key: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use tts_settings.provider_config for provider-specific keys.")
    
    # Twilio Configuration
    twilio_config: TwilioConfig = Field(default_factory=TwilioConfig)
    
    # Deprecated Twilio fields
    twilio_account_sid: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use twilio_config.account_sid instead.")
    twilio_auth_token: Optional[str] = Field(None, deprecated=True, description="Legacy field. Use twilio_config.auth_token instead.")
    
    # LLM Settings
    llm_settings: LLMSettings = Field(default_factory=LLMSettings)
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Interruption Settings
    interruption_settings: InterruptionSettings = Field(default_factory=InterruptionSettings)
    
    # TTS Settings
    tts_settings: TTSSettings = Field(default_factory=TTSSettings)
    
    # STT Settings
    stt_settings: STTSettings = Field(default_factory=STTSettings)
    
    # Call control settings
    end_call_message: Optional[str] = None
    transfer_call_message: Optional[str] = None
    idle_message: Optional[str] = Field(
        default="Are you still there? I'm here to help if you need anything."
    )
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None

    # Tools configuration
    tools_settings: ToolsSettings = Field(default_factory=ToolsSettings)

    # RAG settings
    rag_settings: RAGSettings = Field(default_factory=RAGSettings)

    # Fallback providers configuration
    llm_fallback_providers: LLMFallbackSettings = Field(default_factory=LLMFallbackSettings)
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = True

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Move legacy flat API keys and settings to their new nested locations."""
        
        # Ensure nested config dicts exist
        llm_config = values.get("llm_provider_config", {}) or {}
        twilio_conf = values.get("twilio_config", {}) or {}
        tts_conf = values.get("tts_settings", {}) or {}
        
        # Move legacy LLM fields
        if "openai_api_key" in values and values["openai_api_key"]:
            llm_config["api_key"] = llm_config.get("api_key") or values["openai_api_key"]
        if "custom_llm_url" in values and values["custom_llm_url"]:
            llm_config["base_url"] = llm_config.get("base_url") or values["custom_llm_url"]
        
        # Move legacy Twilio fields
        if "twilio_account_sid" in values and values["twilio_account_sid"]:
            twilio_conf["account_sid"] = twilio_conf.get("account_sid") or values["twilio_account_sid"]
        if "twilio_auth_token" in values and values["twilio_auth_token"]:
            twilio_conf["auth_token"] = twilio_conf.get("auth_token") or values["twilio_auth_token"]
            
        # Move legacy TTS/STT provider keys
        tts_provider_conf = tts_conf.get("provider_config", {}) or {}
        if "elevenlabs_api_key" in values and values["elevenlabs_api_key"]:
            tts_provider_conf["elevenlabs_api_key"] = tts_provider_conf.get("elevenlabs_api_key") or values["elevenlabs_api_key"]
        if "deepgram_api_key" in values and values["deepgram_api_key"]:
            tts_provider_conf["deepgram_api_key"] = tts_provider_conf.get("deepgram_api_key") or values["deepgram_api_key"]
        
        tts_conf["provider_config"] = tts_provider_conf
        
        # Update the main values dict
        values["llm_provider_config"] = llm_config
        values["twilio_config"] = twilio_conf
        values["tts_settings"] = tts_conf
        
        # Clean up legacy fields so they don't get processed further
        values.pop("openai_api_key", None)
        values.pop("custom_llm_url", None)
        values.pop("twilio_account_sid", None)
        values.pop("twilio_auth_token", None)
        values.pop("elevenlabs_api_key", None)
        values.pop("deepgram_api_key", None)

        return values

    @field_validator("llm_provider_config", mode="before")
    @classmethod
    def validate_llm_provider_config(cls, v):
        """Ensure llm_provider_config has required structure."""
        if v is None:
            return LLMProviderConfig()
        return v

    @field_validator("tts_settings", mode="before")
    @classmethod
    def validate_tts_settings(cls, v):
        """Ensure TTS settings have proper structure."""
        if v is None:
            return TTSSettings()

        # Ensure provider_config exists
        if "provider_config" not in v:
            v["provider_config"] = {}

        return v

    @field_validator("stt_settings", mode="before")
    @classmethod
    def validate_stt_settings(cls, v):
        """Ensure STT settings have proper structure."""
        if v is None:
            return STTSettings()

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
            return ToolsSettings()

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
            return LLMFallbackSettings()
        if isinstance(v, dict) and "fallbacks" not in v:
             v["fallbacks"] = []
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

    # Remove deprecated fields from the create schema to encourage correct usage
    openai_api_key: Optional[str] = Field(None, exclude=True)
    custom_llm_url: Optional[str] = Field(None, exclude=True)
    deepgram_api_key: Optional[str] = Field(None, exclude=True)
    elevenlabs_api_key: Optional[str] = Field(None, exclude=True)
    twilio_account_sid: Optional[str] = Field(None, exclude=True)
    twilio_auth_token: Optional[str] = Field(None, exclude=True)
    
    # organization_id and user_id will be set from the authenticated user
    pass


class AssistantUpdate(AssistantBase):
    """Schema for updating an assistant."""

    # Override all fields from AssistantBase to make them optional
    name: Optional[str] = None
    description: Optional[str] = None
    
    # LLM Provider Configuration
    llm_provider: Optional[str] = None
    llm_provider_config: Optional[LLMProviderConfig] = None
    
    # Twilio Configuration
    twilio_config: Optional[TwilioConfig] = None
    
    # LLM Settings
    llm_settings: Optional[LLMSettings] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Interruption Settings
    interruption_settings: Optional[InterruptionSettings] = None
    
    # TTS Settings
    tts_settings: Optional[TTSSettings] = None
    
    # STT Settings
    stt_settings: Optional[STTSettings] = None
    
    # Call control settings
    end_call_message: Optional[str] = None
    transfer_call_message: Optional[str] = None
    idle_message: Optional[str] = None
    max_idle_messages: Optional[int] = None
    idle_timeout: Optional[int] = None

    # Tools configuration
    tools_settings: Optional[ToolsSettings] = None

    # RAG settings
    rag_settings: Optional[RAGSettings] = None

    # Fallback providers configuration
    llm_fallback_providers: Optional[LLMFallbackSettings] = None
    
    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    # Exclude legacy fields from update schema as well
    openai_api_key: Optional[str] = Field(None, exclude=True)
    custom_llm_url: Optional[str] = Field(None, exclude=True)
    deepgram_api_key: Optional[str] = Field(None, exclude=True)
    elevenlabs_api_key: Optional[str] = Field(None, exclude=True)
    twilio_account_sid: Optional[str] = Field(None, exclude=True)
    twilio_auth_token: Optional[str] = Field(None, exclude=True)

class AssistantResponse(AssistantBase):
    """Schema for assistant response."""

    id: int
    organization_id: int
    user_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    # Exclude legacy fields from response schema
    openai_api_key: Optional[str] = Field(None, exclude=True)
    custom_llm_url: Optional[str] = Field(None, exclude=True)
    deepgram_api_key: Optional[str] = Field(None, exclude=True)
    elevenlabs_api_key: Optional[str] = Field(None, exclude=True)
    twilio_account_sid: Optional[str] = Field(None, exclude=True)
    twilio_auth_token: Optional[str] = Field(None, exclude=True)

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


class ToolResponse(BaseModel):
    """Response from a tool."""
    output: str


# Schemas for initiating an outbound call
class InitiateCallRequest(BaseModel):
    """Request model for initiating an outbound call."""
    assistant_id: int
    to_phone_number: str
    welcome_message: Optional[str] = None
    agenda: Optional[str] = None

class InitiateCallResponse(BaseModel):
    """Response model for initiating an outbound call."""
    message: str
    call_sid: str


# Schemas for Call Analytics
class DailyStat(BaseModel):
    """Statistics for a single day."""
    total: int
    completed: int
    failed: int

class AssistantStat(BaseModel):
    """Statistics for a single assistant."""
    assistant_id: int
    assistant_name: str
    calls: int
    duration: float

class CallAnalyticsResponse(BaseModel):
    """Response model for call analytics."""
    total_calls: int
    completed_calls: int
    failed_calls: int
    ongoing_calls: int
    total_duration: float
    avg_duration: float
    success_rate: float
    daily_stats: Dict[str, DailyStat]
    top_assistants: List[AssistantStat]

# Schemas for Call Count and Stats
class CallCountResponse(BaseModel):
    """Response model for call count."""
    count: int

class CallStatsResponse(BaseModel):
    """Response model for high-level call statistics."""
    total_calls: int
    total_duration: int
    average_duration: float

# Schema for updating call metadata
class UpdateCallMetadataRequest(BaseModel):
    """Request model for updating call metadata."""
    metadata: Dict[str, Any]


# Schemas for Organization Phone Numbers
class SyncPhoneNumbersResponse(BaseModel):
    """Response model for syncing phone numbers."""
    success: bool
    message: str
    synced_count: int

class OrganizationAssistantInfo(BaseModel):
    """Minimal assistant info for phone number list."""
    id: int
    name: str
    is_active: bool

class OrganizationPhoneNumberResponse(BaseModel):
    """Response model for listing organization phone numbers."""
    id: int
    phone_number: str
    friendly_name: Optional[str] = None
    twilio_sid: str
    is_active: bool
    capabilities: Dict[str, bool]
    assistant: Optional[OrganizationAssistantInfo] = None
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime] = None


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str
