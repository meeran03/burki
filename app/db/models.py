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
    
    # Organization settings
    settings = Column(JSON, nullable=True, default=lambda: {
        "allow_user_registration": True,
        "require_email_verification": False,
        "max_users": 100,
        "max_assistants": 10,
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
        key = f"diwaar_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
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


class Assistant(Base):
    """
    Assistant model represents a virtual assistant configuration
    with its own phone number and settings.
    """

    __tablename__ = "assistants"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    phone_number = Column(String(20), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    # LLM Provider Configuration
    llm_provider = Column(String(50), nullable=False, default="openai")  # openai, anthropic, gemini, xai, groq
    llm_provider_config = Column(JSON, nullable=True, default=lambda: {
        "api_key": None,
        "base_url": None,
        "model": "gpt-4o-mini",
        "custom_config": {}
    })

    # Legacy API Keys (for backward compatibility - will be deprecated)
    openai_api_key = Column(String(255), nullable=True)
    custom_llm_url = Column(String(255), nullable=True)

    # Other service API keys
    deepgram_api_key = Column(String(255), nullable=True)
    elevenlabs_api_key = Column(String(255), nullable=True)
    twilio_account_sid = Column(String(255), nullable=True)
    twilio_auth_token = Column(String(255), nullable=True)

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
            "voice_id": "rachel",  # Default voice ID
            "model_id": "turbo",  # Default model ID
            "latency": 1,  # Lowest latency setting
            "stability": 0.5,  # Voice stability (0-1)
            "similarity_boost": 0.75,  # Voice similarity boost (0-1)
            "style": 0.0,  # Voice style (0-1)
            "use_speaker_boost": True,  # Whether to use speaker boost
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
        },
    )
    # Call control settings
    end_call_message = Column(String(255), nullable=True)
    transfer_call_message = Column(String(255), nullable=True)
    idle_message = Column(String(255), nullable=True, default="Are you still there? I'm here to help if you need anything.")
    max_idle_messages = Column(Integer, nullable=True)
    idle_timeout = Column(Integer, nullable=True)

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
    calls = relationship(
        "Call", back_populates="assistant", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Assistant(id={self.id}, name='{self.name}', phone_number='{self.phone_number}')>"


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
    recordings = relationship(
        "Recording", back_populates="call", cascade="all, delete-orphan"
    )
    transcripts = relationship(
        "Transcript", back_populates="call", cascade="all, delete-orphan"
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
    recording_sid = Column(String(100), nullable=True)  # Twilio Recording SID
    file_path = Column(String(255), nullable=True)  # Local file path (optional)
    recording_url = Column(String(500), nullable=True)  # Twilio recording URL
    duration = Column(Float, nullable=True)  # Duration in seconds
    format = Column(String(20), nullable=True)  # wav, mp3, etc.
    recording_type = Column(String(20), nullable=True, default="full")  # full, segment
    recording_source = Column(String(20), nullable=True, default="twilio")  # twilio, local
    status = Column(String(20), nullable=True, default="recording")  # recording, completed, failed
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)

    # Relationships
    call = relationship("Call", back_populates="recordings")

    def __repr__(self):
        return (
            f"<Recording(id={self.id}, call_id={self.call_id}, recording_sid='{self.recording_sid}')>"
        )


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
