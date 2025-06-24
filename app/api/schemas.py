"""
This file contains the schemas for the API.
"""
from typing import Dict, List, Optional, Any
import datetime
from pydantic import BaseModel, Field


# Assistant schemas
class AssistantBase(BaseModel):
    """Base schema for assistant data."""

    name: str
    phone_number: str
    description: Optional[str] = None

    # LLM Provider Configuration
    llm_provider: Optional[str] = Field(
        default="openai",
        description="LLM provider: openai, anthropic, gemini, xai, groq",
    )
    llm_provider_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "api_key": None,
            "base_url": None,
            "model": "gpt-4o-mini",
            "custom_config": {},
        }
    )

    # LLM Fallback Providers Configuration
    llm_fallback_providers: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"enabled": False, "fallbacks": []}
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
            "audio_denoising": False,
        }
    )

    # Recording Settings
    recording_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": False,
            "format": "mp3",
            "sample_rate": 8000,
            "channels": 1,
            "record_user_audio": True,
            "record_assistant_audio": True,
            "record_mixed_audio": True,
            "auto_save": True,
            "recordings_dir": "recordings",
            "create_database_records": True,
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
            "end_call": {
                "enabled": False,
                "scenarios": [],
                "custom_message": None,
            },
            "transfer_call": {
                "enabled": False,
                "scenarios": [],
                "transfer_numbers": [],
                "custom_message": None,
            },
            "custom_tools": [],
        }
    )

    # RAG (Retrieval Augmented Generation) settings
    rag_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": True,
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

    # Additional settings
    custom_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = True


class AssistantCreate(AssistantBase):
    """Schema for creating a new assistant."""

    # organization_id and user_id will be set from the authenticated user
    pass


class AssistantUpdate(BaseModel):
    """Schema for updating an assistant."""

    name: Optional[str] = None
    phone_number: Optional[str] = None
    description: Optional[str] = None

    # LLM Provider Configuration
    llm_provider: Optional[str] = None
    llm_provider_config: Optional[Dict[str, Any]] = None
    llm_fallback_providers: Optional[Dict[str, Any]] = None

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

    # Recording Settings
    recording_settings: Optional[Dict[str, Any]] = None

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
        orm_mode = True


# Call schemas
class CallBase(BaseModel):
    """Base schema for call data."""

    call_sid: str
    to_phone_number: str
    customer_phone_number: str
    call_meta: Optional[Dict[str, Any]] = None


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
        orm_mode = True


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
        orm_mode = True


# Recording schemas
class RecordingBase(BaseModel):
    """Base schema for recording data."""

    recording_sid: Optional[str] = None

    # S3 Storage fields
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    s3_bucket: Optional[str] = None

    # Recording metadata
    duration: Optional[float] = None
    file_size: Optional[int] = None
    format: Optional[str] = "mp3"
    sample_rate: Optional[int] = 22050
    channels: Optional[int] = 1

    # Recording classification
    recording_type: Optional[str] = "mixed"  # user, assistant, mixed
    recording_source: Optional[str] = "s3"  # s3, twilio

    # Status and timestamps
    status: Optional[str] = "recording"  # recording, completed, failed, uploaded
    uploaded_at: Optional[datetime.datetime] = None

    # Additional metadata
    recording_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RecordingCreate(RecordingBase):
    """Schema for creating a new recording."""

    call_id: int


class RecordingResponse(RecordingBase):
    """Schema for recording response."""

    id: int
    call_id: int
    created_at: Optional[datetime.datetime] = None

    class Config:
        orm_mode = True


# Document schemas
class DocumentBase(BaseModel):
    """Base schema for document data."""

    name: str
    original_filename: str
    content_type: str
    file_size: int
    file_hash: str

    # S3 storage information
    s3_key: str
    s3_url: Optional[str] = None
    s3_bucket: Optional[str] = None

    # Document processing status
    processing_status: Optional[str] = (
        "pending"  # pending, processing, completed, failed
    )
    processing_error: Optional[str] = None

    # Document type and metadata
    document_type: Optional[str] = None
    language: Optional[str] = "en"

    # Chunking configuration
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunking_strategy: Optional[str] = "recursive"

    # Document metadata
    document_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Tags and categories
    tags: Optional[List[str]] = Field(default_factory=list)
    category: Optional[str] = None

    # Access control
    is_active: Optional[bool] = True
    is_public: Optional[bool] = False


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    assistant_id: int


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    name: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    document_metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None


class DocumentResponse(DocumentBase):
    """Schema for document response."""

    id: int
    assistant_id: int
    organization_id: int
    total_chunks: int
    processed_chunks: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    processed_at: Optional[datetime.datetime] = None

    class Config:
        orm_mode = True


# Document Chunk schemas
class DocumentChunkBase(BaseModel):
    """Base schema for document chunk data."""

    chunk_index: int
    content: str
    chunk_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating a new document chunk."""

    document_id: int


class DocumentChunkResponse(DocumentChunkBase):
    """Schema for document chunk response."""

    id: int
    document_id: int
    created_at: datetime.datetime
    embedded_at: Optional[datetime.datetime] = None

    class Config:
        orm_mode = True


# Chat Message schemas
class ChatMessageBase(BaseModel):
    """Base schema for chat message data."""

    role: str  # system, user, assistant
    content: str
    message_index: int
    timestamp: Optional[datetime.datetime] = None

    # LLM provider information
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    # Token usage and costs
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Additional metadata
    message_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating a new chat message."""

    call_id: int


class ChatMessageResponse(ChatMessageBase):
    """Schema for chat message response."""

    id: int
    call_id: int
    timestamp: datetime.datetime

    class Config:
        orm_mode = True


# Webhook Log schemas
class WebhookLogBase(BaseModel):
    """Base schema for webhook log data."""

    webhook_url: str
    webhook_type: str  # status-update, end-of-call-report

    # Request details
    request_payload: Dict[str, Any]
    request_headers: Optional[Dict[str, Any]] = None

    # Response details
    response_status_code: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Optional[Dict[str, Any]] = None

    # Timing and status
    attempted_at: Optional[datetime.datetime] = None
    response_time_ms: Optional[int] = None
    success: Optional[bool] = False

    # Error information
    error_message: Optional[str] = None
    retry_count: Optional[int] = 0

    # Additional metadata
    webhook_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WebhookLogCreate(WebhookLogBase):
    """Schema for creating a new webhook log."""

    call_id: int
    assistant_id: int


class WebhookLogResponse(WebhookLogBase):
    """Schema for webhook log response."""

    id: int
    call_id: int
    assistant_id: int
    attempted_at: datetime.datetime

    class Config:
        orm_mode = True


# Billing schemas
class BillingPlanBase(BaseModel):
    """Base schema for billing plan data."""

    name: str
    description: Optional[str] = None
    stripe_price_id: Optional[str] = None
    monthly_minutes: Optional[int] = None  # null = unlimited
    price_cents: int = 0
    features: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "unlimited_assistants": False,
            "webhook_support": True,
            "api_access": True,
            "priority_support": False,
            "custom_integrations": False,
        }
    )
    is_active: Optional[bool] = True
    sort_order: Optional[int] = 0


class BillingPlanResponse(BillingPlanBase):
    """Schema for billing plan response."""

    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True


class BillingAccountBase(BaseModel):
    """Base schema for billing account data."""

    plan_id: int
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_period_minutes_used: Optional[int] = 0
    current_period_start: Optional[datetime.datetime] = None
    current_period_end: Optional[datetime.datetime] = None
    auto_topup_enabled: Optional[bool] = False
    topup_threshold_minutes: Optional[int] = 10
    topup_amount_minutes: Optional[int] = 100
    topup_price_cents: Optional[int] = 500
    status: Optional[str] = "active"  # active, suspended, cancelled
    is_payment_method_attached: Optional[bool] = False
    billing_settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "email_notifications": True,
            "usage_alerts": True,
            "low_balance_threshold": 50,
        }
    )


class BillingAccountResponse(BillingAccountBase):
    """Schema for billing account response."""

    id: int
    organization_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True


class UsageRecordBase(BaseModel):
    """Base schema for usage record data."""

    minutes_used: float
    usage_type: Optional[str] = "call"  # call, topup_credit, adjustment
    description: Optional[str] = None
    billing_period_start: datetime.datetime
    billing_period_end: datetime.datetime
    record_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UsageRecordCreate(UsageRecordBase):
    """Schema for creating a new usage record."""

    billing_account_id: int
    call_id: Optional[int] = None


class UsageRecordResponse(UsageRecordBase):
    """Schema for usage record response."""

    id: int
    billing_account_id: int
    call_id: Optional[int] = None
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class BillingTransactionBase(BaseModel):
    """Base schema for billing transaction data."""

    transaction_type: str  # subscription, topup, refund, adjustment
    amount_cents: int
    currency: Optional[str] = "usd"
    description: Optional[str] = None
    stripe_payment_intent_id: Optional[str] = None
    stripe_invoice_id: Optional[str] = None
    stripe_charge_id: Optional[str] = None
    status: Optional[str] = "pending"  # pending, succeeded, failed, cancelled
    minutes_credited: Optional[int] = None
    billing_period_start: Optional[datetime.datetime] = None
    billing_period_end: Optional[datetime.datetime] = None
    transaction_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BillingTransactionCreate(BillingTransactionBase):
    """Schema for creating a new billing transaction."""

    billing_account_id: int


class BillingTransactionResponse(BillingTransactionBase):
    """Schema for billing transaction response."""

    id: int
    billing_account_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True


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
    is_verified: bool
    avatar_url: Optional[str] = None
    last_login_at: Optional[datetime.datetime] = None
    login_count: int
    preferences: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class OrganizationResponse(BaseModel):
    """Schema for organization response in API."""

    id: int
    name: str
    slug: str
    description: Optional[str] = None
    domain: Optional[str] = None
    is_active: bool
    settings: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True


# User API Key schemas
class UserAPIKeyBase(BaseModel):
    """Base schema for user API key data."""

    name: str
    permissions: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "read": True,
            "write": True,
            "admin": False,
        }
    )
    rate_limit: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
        }
    )
    is_active: Optional[bool] = True


class UserAPIKeyCreate(UserAPIKeyBase):
    """Schema for creating a new user API key."""

    pass


class UserAPIKeyResponse(BaseModel):
    """Schema for user API key response."""

    id: int
    user_id: int
    name: str
    key_prefix: str
    last_used_at: Optional[datetime.datetime] = None
    usage_count: int
    is_active: bool
    permissions: Dict[str, Any]
    rate_limit: Dict[str, Any]
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True
