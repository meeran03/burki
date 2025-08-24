import datetime
import hashlib
import secrets
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Float,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import Optional, Dict, Any

# pgvector imports
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Fallback for when pgvector is not available
    class Vector:
        def __init__(self, dim):
            pass

Base = declarative_base()


class Organization(Base):
    """
    Organization model represents a company or entity that has users and assistants.
    """

    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=True)  # For domain-based signup restrictions
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Telephony Provider Credentials (organization-level)
    twilio_account_sid = Column(String(255), nullable=True)
    twilio_auth_token = Column(String(255), nullable=True)
    telnyx_api_key = Column(String(255), nullable=True)
    telnyx_connection_id = Column(String(255), nullable=True)
    
    # Organization settings
    settings = Column(JSON, nullable=True, default=lambda: {
        "allow_user_registration": True,
        "require_email_verification": False,
        "max_users": 100,
        "max_assistants": 10,
        "telephony": {
            "twilio": {
                "webhook_url": None,
                "auto_configure_webhooks": True,
            },
            "telnyx": {
                "webhook_url": None,
                "auto_configure_webhooks": True,
            }
        }
    })

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    assistants = relationship("Assistant", back_populates="organization", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="organization", cascade="all, delete-orphan")
    phone_numbers = relationship("PhoneNumber", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}', slug='{self.slug}')>"


class User(Base):
    """
    User model represents a user within an organization.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    full_name = Column(String(100), nullable=False)  # Kept for backward compatibility
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth-only users
    
    # OAuth fields
    google_id = Column(String(100), nullable=True, index=True)
    avatar_url = Column(String(500), nullable=True)
    
    # User status and role
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    role = Column(String(20), nullable=False, default="user")  # admin, user, viewer
    
    # Login tracking
    last_login_at = Column(DateTime, nullable=True)
    login_count = Column(Integer, nullable=False, default=0)
    
    # User preferences
    preferences = Column(JSON, nullable=True, default=lambda: {
        "timezone": "UTC",
        "notifications": {
            "email": True,
            "browser": True,
        },
        "dashboard": {
            "refresh_interval": 30,
            "default_view": "overview",
        },
    })

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    organization = relationship("Organization", back_populates="users")
    api_keys = relationship("UserAPIKey", back_populates="user", cascade="all, delete-orphan")
    assistants = relationship("Assistant", back_populates="user", cascade="all, delete-orphan")

    # Unique constraint for email within organization
    __table_args__ = (
        Index('idx_user_org_email', 'organization_id', 'email', unique=True),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', organization_id={self.organization_id})>"

    def set_password(self, password):
        """Set password hash."""
        import bcrypt
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        """Check if password matches hash."""
        if not self.password_hash:
            return False
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def set_full_name(self, first_name, last_name):
        """Set first_name, last_name, and full_name together."""
        self.first_name = first_name.strip() if first_name else ""
        self.last_name = last_name.strip() if last_name else ""
        self.full_name = f"{self.first_name} {self.last_name}".strip()
    
    def split_full_name_if_needed(self):
        """Split full_name into first_name and last_name if they're not set."""
        if self.full_name and (not self.first_name or not self.last_name):
            parts = self.full_name.strip().split(' ', 1)
            self.first_name = parts[0] if len(parts) > 0 else ""
            self.last_name = parts[1] if len(parts) > 1 else ""


class UserAPIKey(Base):
    """
    UserAPIKey model represents API keys generated by users for system access.
    """

    __tablename__ = "user_api_keys"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, index=True)  # SHA256 hash of the key
    key_prefix = Column(String(20), nullable=False)  # First few chars for identification
    
    # Key metadata
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Permissions (can be expanded later)
    permissions = Column(JSON, nullable=True, default=lambda: {
        "read": True,
        "write": True,
        "admin": False,
    })
    
    # Rate limiting
    rate_limit = Column(JSON, nullable=True, default=lambda: {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
    })

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<UserAPIKey(id={self.id}, user_id={self.user_id}, name='{self.name}', prefix='{self.key_prefix}')>"

    @staticmethod
    def generate_api_key():
        """Generate a new API key."""
        # Generate a secure random key
        key = f"burki_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
        return key

    @staticmethod
    def hash_key(key):
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    @staticmethod
    def get_key_prefix(key):
        """Get the prefix of an API key for identification."""
        return key[:12] + "..."

    def verify_key(self, key):
        """Verify if the provided key matches this record."""
        return self.key_hash == self.hash_key(key)


class PhoneNumber(Base):
    """
    PhoneNumber model represents a phone number owned by an organization
    that can be assigned to assistants.
    """

    __tablename__ = "phone_numbers"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Phone number details
    phone_number = Column(String(20), unique=True, nullable=False, index=True)
    friendly_name = Column(String(100), nullable=True)
    
    # Provider information
    provider = Column(String(20), nullable=False, default="twilio")  # "twilio" or "telnyx"
    provider_phone_id = Column(String(100), nullable=True)  # Twilio SID or Telnyx phone number ID
    
    # Current assignment
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=True, index=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Provider capabilities (both Twilio and Telnyx support these)
    capabilities = Column(JSON, nullable=True, default=lambda: {
        "voice": True,
        "sms": False,
        "mms": False,
        "fax": False
    })
    
    # Phone number metadata
    phone_metadata = Column(JSON, nullable=True, default=lambda: {})
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="phone_numbers")
    assistant = relationship("Assistant", back_populates="phone_numbers")
    calls = relationship("Call", back_populates="phone_number")

    def __repr__(self):
        return f"<PhoneNumber(id={self.id}, phone_number='{self.phone_number}', organization_id={self.organization_id})>"

    def get_display_name(self) -> str:
        """Get display name for the phone number."""
        if self.friendly_name:
            return f"{self.friendly_name} ({self.phone_number})"
        return self.phone_number


class Assistant(Base):
    """
    Assistant model represents a virtual assistant configuration.
    Phone numbers are now managed separately in the PhoneNumber model.
    """

    __tablename__ = "assistants"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    # LLM Provider Configuration
    llm_provider = Column(String(50), nullable=False, default="openai")  # openai, anthropic, gemini, xai, groq
    llm_provider_config = Column(JSON, nullable=True, default=lambda: {
        "api_key": None,
        "base_url": None,
        "model": "gpt-4o-mini",
        "custom_config": {}
    })

    # LLM Fallback Providers Configuration
    llm_fallback_providers = Column(JSON, nullable=True, default=lambda: {
        "enabled": False,
        "fallbacks": []
    })

    # Legacy API Keys (for backward compatibility - will be deprecated)
    openai_api_key = Column(String(255), nullable=True)
    custom_llm_url = Column(String(255), nullable=True)

    # Other service API keys
    deepgram_api_key = Column(String(255), nullable=True)
    inworld_bearer_token = Column(String(255), nullable=True)
    inworld_workspace_id = Column(String(255), nullable=True)
    resemble_api_key = Column(String(255), nullable=True)
    elevenlabs_api_key = Column(String(255), nullable=True)
    
    # Telephony Provider Configuration removed - now handled at organization level
    # Phone numbers specify their provider and use organization-level credentials

    llm_settings = Column(JSON, nullable=True, default=lambda: {
        "temperature": 0.5,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful assistant that can answer questions and help with tasks.",
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop_sequences": [],
    })

    # Webhook settings
    webhook_url = Column(String(255), nullable=True)
    sms_webhook_url = Column(String(255), nullable=True)
    messaging_service_sid = Column(String(255), nullable=True)

    # Interruption settings as JSON
    interruption_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "interruption_threshold": 3,  # Number of words to trigger interruption
            "min_speaking_time": 0.5,  # Minimum time AI must be speaking before interruption is valid (in seconds)
            "interruption_cooldown": 2.0,  # Cooldown period in seconds between interruptions
        },
    )

    # TTS (Text-to-Speech) settings as JSON
    tts_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "provider": "elevenlabs",  # Provider: elevenlabs, deepgram, etc.
            "voice_id": "rachel",  # Default voice ID
            "model_id": "turbo",  # Default model ID
            "latency": 1,  # Lowest latency setting
            "stability": 0.5,  # Voice stability (0-1)
            "similarity_boost": 0.75,  # Voice similarity boost (0-1)
            "style": 0.0,  # Voice style (0-1)
            "use_speaker_boost": True,  # Whether to use speaker boost
            # Provider-specific additional settings can go here
            "provider_config": {},
        },
    )

    # STT (Speech-to-Text) settings as JSON
    stt_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "model": "nova-2",  # Deepgram model to use
            "language": "en-US",  # Language code
            "punctuate": True,  # Whether to add punctuation
            "interim_results": True,  # Whether to return interim results
            "endpointing": {  # Endpointing settings
                "silence_threshold": 500,  # Silence threshold in ms
                "min_silence_duration": 500,  # Minimum silence duration in ms
            },
            "utterance_end_ms": 1000,  # Time in ms to wait before considering an utterance complete
            "vad_turnoff": 500,  # Voice activity detection turnoff in ms
            "smart_format": True,  # Whether to use smart formatting
            "audio_denoising": False,  # Whether to enable real-time audio denoising
        },
    )

    # Recording settings as JSON
    recording_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "enabled": False,  # Whether local recording is enabled
            "format": "mp3",  # Audio format ("wav", "mp3") - MP3 recommended for better quality
            "sample_rate": 8000,  # Audio sample rate in Hz (will be upsampled to 22050 for MP3)
            "channels": 1,  # Number of audio channels
            "record_user_audio": True,  # Whether to record user audio
            "record_assistant_audio": True,  # Whether to record assistant audio
            "record_mixed_audio": True,  # Whether to record mixed audio (both user and assistant)
            "auto_save": True,  # Whether to automatically save recordings when call ends
            "recordings_dir": "recordings",  # Directory to save recordings
            "create_database_records": True,  # Whether to create database records for local recordings
        },
    )

    # Call control settings
    end_call_message = Column(String(255), nullable=True)
    transfer_call_message = Column(String(255), nullable=True)
    idle_message = Column(String(255), nullable=True, default="Are you still there? I'm here to help if you need anything.")
    max_idle_messages = Column(Integer, nullable=True)
    idle_timeout = Column(Integer, nullable=True)

    # Tools configuration as JSON
    tools_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "enabled_tools": [],  # List of enabled tool names: ["endCall", "transferCall"]
            "end_call": {
                "enabled": False,
                "scenarios": [],  # List of scenarios when to end call
                "custom_message": None,  # Custom end call message
            },
            "transfer_call": {
                "enabled": False,
                "scenarios": [],  # List of scenarios when to transfer
                "transfer_numbers": [],  # List of phone numbers to transfer to
                "custom_message": None,  # Custom transfer message
            },
            "custom_tools": [],  # List of custom tool definitions
        },
    )

    # RAG (Retrieval Augmented Generation) settings as JSON
    rag_settings = Column(
        JSON,
        nullable=True,
        default=lambda: {
            "enabled": True,  # Whether RAG is enabled for this assistant
            "search_limit": 3,  # Number of document chunks to retrieve
            "similarity_threshold": 0.7,  # Minimum similarity score for relevance
            "embedding_model": "text-embedding-3-small",  # Embedding model to use
            "chunking_strategy": "recursive",  # Default chunking strategy for new documents
            "chunk_size": 1000,  # Default chunk size for new documents
            "chunk_overlap": 200,  # Default chunk overlap for new documents
            "auto_process": True,  # Whether to auto-process uploaded documents
            "include_metadata": True,  # Whether to include document metadata in responses
            "context_window_tokens": 4000,  # Max tokens to use for document context
        },
    )

    # Additional settings
    custom_settings = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=True, default=True)

    # Timestamps
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=True,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    organization = relationship("Organization", back_populates="assistants")
    user = relationship("User", back_populates="assistants")
    phone_numbers = relationship("PhoneNumber", back_populates="assistant")
    calls = relationship(
        "Call", back_populates="assistant", cascade="all, delete-orphan"
    )
    documents = relationship("Document", back_populates="assistant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Assistant(id={self.id}, name='{self.name}')>"


class Call(Base):
    """
    Call model represents a phone call with a specific assistant.
    """

    __tablename__ = "calls"

    id = Column(Integer, primary_key=True)
    call_sid = Column(String(100), unique=True, nullable=False)
    assistant_id = Column(
        Integer, ForeignKey("assistants.id"), nullable=False, index=True
    )
    phone_number_id = Column(Integer, ForeignKey("phone_numbers.id"), nullable=True, index=True)
    to_phone_number = Column(String(20), nullable=False)
    customer_phone_number = Column(String(20), nullable=False)
    status = Column(
        String(20), nullable=False, index=True
    )  # ongoing, completed, failed
    duration = Column(Integer, nullable=True)  # Duration in seconds
    started_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    call_meta = Column(JSON, nullable=True)  # Changed from 'metadata' to 'call_meta'

    # Relationships
    assistant = relationship("Assistant", back_populates="calls")
    phone_number = relationship("PhoneNumber", back_populates="calls")
    recordings = relationship(
        "Recording", back_populates="call", cascade="all, delete-orphan"
    )
    transcripts = relationship(
        "Transcript", back_populates="call", cascade="all, delete-orphan"
    )
    chat_messages = relationship(
        "ChatMessage", back_populates="call", cascade="all, delete-orphan"
    )
    webhook_logs = relationship(
        "WebhookLog", back_populates="call", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<Call(id={self.id}, call_sid='{self.call_sid}', status='{self.status}')>"
        )


class Recording(Base):
    """
    Recording model represents an audio recording of a call.
    """

    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    recording_sid = Column(String(100), nullable=True)  # Twilio Recording SID (deprecated)
    
    # S3 Storage fields
    s3_key = Column(String(500), nullable=True)  # S3 object key
    s3_url = Column(String(1000), nullable=True)  # S3 public URL
    s3_bucket = Column(String(100), nullable=True)  # S3 bucket name
    
    # Recording metadata
    duration = Column(Float, nullable=True)  # Duration in seconds
    file_size = Column(Integer, nullable=True)  # File size in bytes
    format = Column(String(20), nullable=False, default="mp3")  # Audio format
    sample_rate = Column(Integer, nullable=True, default=22050)  # Sample rate in Hz
    channels = Column(Integer, nullable=True, default=1)  # Number of audio channels
    
    # Recording classification
    recording_type = Column(String(20), nullable=False, default="mixed")  # user, assistant, mixed
    recording_source = Column(String(20), nullable=False, default="s3")  # s3, twilio (deprecated)
    
    # Status and timestamps
    status = Column(String(20), nullable=False, default="recording")  # recording, completed, failed, uploaded
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    uploaded_at = Column(DateTime, nullable=True)  # When uploaded to S3
    
    # Additional metadata stored as JSON
    recording_metadata = Column(JSON, nullable=True, default=lambda: {})

    # Relationships
    call = relationship("Call", back_populates="recordings")

    def __repr__(self):
        return (
            f"<Recording(id={self.id}, call_id={self.call_id}, s3_key='{self.s3_key}', type='{self.recording_type}')>"
        )

    def get_download_url(self) -> Optional[str]:
        """
        Get the download URL for this recording.
        
        Returns:
            Optional[str]: S3 URL if available, None otherwise
        """
        return self.s3_url

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get file information for this recording.
        
        Returns:
            Dict[str, Any]: File information including size, format, etc.
        """
        return {
            "id": self.id,
            "s3_key": self.s3_key,
            "s3_url": self.s3_url,
            "format": self.format,
            "duration": self.duration,
            "file_size": self.file_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "recording_type": self.recording_type,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "recording_metadata": self.recording_metadata or {},
        }


class Transcript(Base):
    """
    Transcript model represents a speech transcript segment.
    """

    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    is_final = Column(Boolean, nullable=True, default=True)
    segment_start = Column(Float, nullable=True)  # Start time in seconds
    segment_end = Column(Float, nullable=True)  # End time in seconds
    confidence = Column(Float, nullable=True)  # Confidence score
    speaker = Column(String(20), nullable=True, index=True)  # user, assistant
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)

    # Relationships
    call = relationship("Call", back_populates="transcripts")

    def __repr__(self):
        return f"<Transcript(id={self.id}, call_id={self.call_id}, speaker='{self.speaker}')>"
class Document(Base):
    """
    Document model represents uploaded documents for RAG functionality.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Document metadata
    name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash for deduplication
    
    # S3 storage information
    s3_key = Column(String(500), nullable=False)  # S3 object key
    s3_url = Column(String(1000), nullable=True)  # S3 public URL
    s3_bucket = Column(String(100), nullable=True)  # S3 bucket name
    
    # Document processing status
    processing_status = Column(String(20), nullable=False, default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)  # Error message if processing failed
    
    # Document content and chunks
    total_chunks = Column(Integer, nullable=False, default=0)
    processed_chunks = Column(Integer, nullable=False, default=0)
    
    # Document type and metadata
    document_type = Column(String(50), nullable=True)  # pdf, docx, txt, md, etc.
    language = Column(String(10), nullable=True, default="en")  # Document language
    
    # Chunking configuration used
    chunk_size = Column(Integer, nullable=True)
    chunk_overlap = Column(Integer, nullable=True)
    chunking_strategy = Column(String(50), nullable=True, default="recursive")  # recursive, semantic, etc.
    
    # Document metadata
    document_metadata = Column(JSON, nullable=True, default=lambda: {})
    
    # Tags and categories for organization
    tags = Column(JSON, nullable=True, default=lambda: [])  # List of tags
    category = Column(String(100), nullable=True)  # Document category
    
    # Access control
    is_active = Column(Boolean, nullable=False, default=True)
    is_public = Column(Boolean, nullable=False, default=False)  # Whether accessible to all assistants in org

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
    processed_at = Column(DateTime, nullable=True)  # When processing completed

    # Relationships
    assistant = relationship("Assistant", back_populates="documents")
    organization = relationship("Organization", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, name='{self.name}', assistant_id={self.assistant_id}, status='{self.processing_status}')>"

    def get_processing_progress(self) -> float:
        """Get processing progress as a percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    def is_processing_complete(self) -> bool:
        """Check if document processing is complete."""
        return self.processing_status == "completed" and self.processed_chunks == self.total_chunks


class DocumentChunk(Base):
    """
    DocumentChunk model represents chunks of documents with embeddings for RAG.
    """

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)  # Index within the document
    content = Column(Text, nullable=False)  # The actual text content
    # Embedding vector - using pgvector
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small dimensions
    else:
        embedding = Column(Text, nullable=True)  # Fallback for development without pgvector
    
    # Chunk metadata from the original document
    chunk_metadata = Column(JSON, nullable=True, default=lambda: {})
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    embedded_at = Column(DateTime, nullable=True)  # When embedding was created

    # Relationships
    document = relationship("Document", back_populates="chunks")

    # Add unique constraint for chunk within document
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_index', unique=True),
    )

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
    
    @staticmethod
    def get_embedding_field():
        """Return the name of the embedding field for PostgresSearcher."""
        return "embedding"
    
    @staticmethod
    def get_text_search_field():
        """Return the name of the text search field for PostgresSearcher."""
        return "content"


class ChatMessage(Base):
    """
    ChatMessage model represents individual messages in the conversation history.
    This stores the actual LLM conversation flow (system, user, assistant messages)
    separate from transcripts which are the raw speech-to-text output.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    
    # Message content and metadata
    role = Column(String(20), nullable=False, index=True)  # system, user, assistant
    content = Column(Text, nullable=False)  # The message content
    message_index = Column(Integer, nullable=False)  # Order within the conversation
    
    # Message timing
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    # LLM provider information
    llm_provider = Column(String(50), nullable=True)  # Which LLM provider generated this (for assistant messages)
    llm_model = Column(String(100), nullable=True)  # Which model was used (for assistant messages)
    
    # Token usage and costs (for assistant messages)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    # Additional metadata
    message_metadata = Column(JSON, nullable=True, default=lambda: {})

    # Relationships
    call = relationship("Call", back_populates="chat_messages")

    # Add unique constraint for message within call
    __table_args__ = (
        Index('idx_call_message_index', 'call_id', 'message_index', unique=True),
    )

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, call_id={self.call_id}, role='{self.role}', index={self.message_index})>"


class WebhookLog(Base):
    """
    WebhookLog model represents webhook delivery attempts and their results.
    """

    __tablename__ = "webhook_logs"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False, index=True)
    
    # Webhook details
    webhook_url = Column(String(1000), nullable=False)
    webhook_type = Column(String(50), nullable=False, index=True)  # status-update, end-of-call-report
    
    # Request details
    request_payload = Column(JSON, nullable=False)  # The payload that was sent
    request_headers = Column(JSON, nullable=True)  # Headers sent with the request
    
    # Response details
    response_status_code = Column(Integer, nullable=True)  # HTTP status code
    response_body = Column(Text, nullable=True)  # Response body (truncated if too long)
    response_headers = Column(JSON, nullable=True)  # Response headers
    
    # Timing and status
    attempted_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    response_time_ms = Column(Integer, nullable=True)  # Response time in milliseconds
    success = Column(Boolean, nullable=False, default=False)  # Whether the webhook was successful
    
    # Error information
    error_message = Column(Text, nullable=True)  # Error message if the webhook failed
    retry_count = Column(Integer, nullable=False, default=0)  # Number of retries attempted
    
    # Additional metadata
    webhook_metadata = Column(JSON, nullable=True, default=lambda: {})

    # Relationships
    call = relationship("Call", back_populates="webhook_logs")
    assistant = relationship("Assistant")

    def __repr__(self):
        return f"<WebhookLog(id={self.id}, call_id={self.call_id}, type='{self.webhook_type}', success={self.success})>"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the webhook attempt."""
        return {
            "id": self.id,
            "call_id": self.call_id,
            "assistant_id": self.assistant_id,
            "webhook_url": self.webhook_url,
            "webhook_type": self.webhook_type,
            "success": self.success,
            "response_status_code": self.response_status_code,
            "response_time_ms": self.response_time_ms,
            "attempted_at": self.attempted_at.isoformat() if self.attempted_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


class Tool(Base):
    """
    Tool model represents custom tools that can be used by assistants.
    Tools are organization-scoped and can be shared across multiple assistants.
    """

    __tablename__ = "tools"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Creator
    
    # Tool identification
    name = Column(String(100), nullable=False)  # Function name (snake_case)
    display_name = Column(String(200), nullable=False)  # Human-readable name
    description = Column(Text, nullable=False)  # Tool description for LLM
    tool_type = Column(String(50), nullable=False, index=True)  # 'endpoint', 'python_function', 'lambda'
    
    # Tool configuration (JSON) - specific to tool type
    configuration = Column(JSON, nullable=False, default=lambda: {})
    
    # Function definition for LLM (JSON)
    function_definition = Column(JSON, nullable=False, default=lambda: {})
    
    # Tool settings
    timeout_seconds = Column(Integer, nullable=False, default=30)
    retry_attempts = Column(Integer, nullable=False, default=3)
    is_active = Column(Boolean, nullable=False, default=True)
    is_public = Column(Boolean, nullable=False, default=False)  # Whether available to all orgs
    
    # Usage statistics
    execution_count = Column(Integer, nullable=False, default=0)
    last_executed_at = Column(DateTime, nullable=True)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    
    # Version control
    version = Column(String(20), nullable=False, default="1.0.0")
    parent_tool_id = Column(Integer, ForeignKey("tools.id"), nullable=True)  # For duplicated tools
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    organization = relationship("Organization", back_populates="tools")
    user = relationship("User", back_populates="created_tools")
    assistant_tools = relationship("AssistantTool", back_populates="tool", cascade="all, delete-orphan")
    execution_logs = relationship("ToolExecutionLog", back_populates="tool", cascade="all, delete-orphan")
    
    # Self-referencing relationship for duplicated tools
    duplicated_tools = relationship("Tool", remote_side=[id])

    # Unique constraint for tool name within organization
    __table_args__ = (
        Index('idx_tool_org_name', 'organization_id', 'name', unique=True),
    )

    def __repr__(self):
        return f"<Tool(id={self.id}, name='{self.name}', type='{self.tool_type}', organization_id={self.organization_id})>"

    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        total_executions = self.success_count + self.failure_count
        if total_executions == 0:
            return 0.0
        return (self.success_count / total_executions) * 100


class AssistantTool(Base):
    """
    AssistantTool model represents the many-to-many relationship between assistants and tools.
    Allows for assistant-specific configuration overrides.
    """

    __tablename__ = "assistant_tools"

    id = Column(Integer, primary_key=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False, index=True)
    tool_id = Column(Integer, ForeignKey("tools.id"), nullable=False, index=True)
    
    # Assignment settings
    enabled = Column(Boolean, nullable=False, default=True)
    custom_configuration = Column(JSON, nullable=True)  # Assistant-specific overrides
    
    # Assignment metadata
    assigned_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    # Usage statistics for this assignment
    execution_count = Column(Integer, nullable=False, default=0)
    last_executed_at = Column(DateTime, nullable=True)

    # Relationships
    assistant = relationship("Assistant", back_populates="assistant_tools")
    tool = relationship("Tool", back_populates="assistant_tools")
    assigned_by = relationship("User")

    # Unique constraint for assistant-tool assignment
    __table_args__ = (
        Index('idx_assistant_tool', 'assistant_id', 'tool_id', unique=True),
    )

    def __repr__(self):
        return f"<AssistantTool(id={self.id}, assistant_id={self.assistant_id}, tool_id={self.tool_id}, enabled={self.enabled})>"


class ToolExecutionLog(Base):
    """
    ToolExecutionLog model represents execution history and logs for tool calls.
    """

    __tablename__ = "tool_execution_logs"

    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey("tools.id"), nullable=False, index=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True, index=True)
    
    # Execution details
    parameters = Column(JSON, nullable=True)  # Input parameters
    result = Column(JSON, nullable=True)  # Execution result
    status = Column(String(20), nullable=False, index=True)  # 'success', 'error', 'timeout'
    error_message = Column(Text, nullable=True)  # Error details if failed
    
    # Timing information
    started_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Execution duration in milliseconds
    
    # Context information
    execution_context = Column(JSON, nullable=True)  # Call context, user info, etc.
    
    # Provider-specific information
    provider_response = Column(JSON, nullable=True)  # Raw response from provider
    retry_count = Column(Integer, nullable=False, default=0)

    # Relationships
    tool = relationship("Tool", back_populates="execution_logs")
    assistant = relationship("Assistant")
    call = relationship("Call")

    def __repr__(self):
        return f"<ToolExecutionLog(id={self.id}, tool_id={self.tool_id}, status='{self.status}', duration_ms={self.duration_ms})>"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "id": self.id,
            "tool_id": self.tool_id,
            "assistant_id": self.assistant_id,
            "call_id": self.call_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


# Update existing models to include tool relationships
def add_tool_relationships():
    """Add tool relationships to existing models."""
    
    # Add tools relationship to Organization
    if not hasattr(Organization, 'tools'):
        Organization.tools = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    
    # Add created_tools relationship to User
    if not hasattr(User, 'created_tools'):
        User.created_tools = relationship("Tool", back_populates="user", cascade="all, delete-orphan")
    
    # Add assistant_tools relationship to Assistant
    if not hasattr(Assistant, 'assistant_tools'):
        Assistant.assistant_tools = relationship("AssistantTool", back_populates="assistant", cascade="all, delete-orphan")

# Update existing models to include document relationships
# Add to Organization class
def add_documents_relationship_to_organization():
    """Add documents relationship to Organization model."""
    if not hasattr(Organization, 'documents'):
        Organization.documents = relationship("Document", back_populates="organization", cascade="all, delete-orphan")

# Add to Assistant class  
def add_documents_relationship_to_assistant():
    """Add documents relationship to Assistant model."""
    if not hasattr(Assistant, 'documents'):
        Assistant.documents = relationship("Document", back_populates="assistant", cascade="all, delete-orphan")

# Call the functions to add relationships
add_tool_relationships()
add_documents_relationship_to_organization()
add_documents_relationship_to_assistant()
