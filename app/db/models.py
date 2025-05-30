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
    billing_account = relationship("BillingAccount", back_populates="organization", uselist=False, cascade="all, delete-orphan")

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


class BillingPlan(Base):
    """
    BillingPlan model represents available subscription plans.
    """

    __tablename__ = "billing_plans"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)  # "Starter", "Pro"
    description = Column(Text, nullable=True)
    stripe_price_id = Column(String(100), nullable=True)  # Stripe price ID for paid plans
    
    # Plan limits and pricing
    monthly_minutes = Column(Integer, nullable=True)  # null = unlimited
    price_cents = Column(Integer, nullable=False, default=0)  # Price in cents
    
    # Plan features
    features = Column(JSON, nullable=True, default=lambda: {
        "unlimited_assistants": False,
        "webhook_support": True,
        "api_access": True,
        "priority_support": False,
        "custom_integrations": False,
    })
    
    # Plan status
    is_active = Column(Boolean, nullable=False, default=True)
    sort_order = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    billing_accounts = relationship("BillingAccount", back_populates="plan")

    def __repr__(self):
        return f"<BillingPlan(id={self.id}, name='{self.name}', price_cents={self.price_cents})>"


class BillingAccount(Base):
    """
    BillingAccount model represents billing information for an organization.
    """

    __tablename__ = "billing_accounts"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, unique=True, index=True)
    plan_id = Column(Integer, ForeignKey("billing_plans.id"), nullable=False, index=True)
    
    # Stripe information
    stripe_customer_id = Column(String(100), nullable=True, index=True)
    stripe_subscription_id = Column(String(100), nullable=True, index=True)
    
    # Usage tracking
    current_period_minutes_used = Column(Integer, nullable=False, default=0)
    current_period_start = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    current_period_end = Column(DateTime, nullable=False)  # Will be set based on plan
    
    # Top-up configuration
    auto_topup_enabled = Column(Boolean, nullable=False, default=False)
    topup_threshold_minutes = Column(Integer, nullable=False, default=10)  # When to trigger auto top-up
    topup_amount_minutes = Column(Integer, nullable=False, default=100)  # How many minutes to add
    topup_price_cents = Column(Integer, nullable=False, default=500)  # Price for top-up in cents
    
    # Account status
    status = Column(String(20), nullable=False, default="active")  # active, suspended, cancelled
    is_payment_method_attached = Column(Boolean, nullable=False, default=False)
    
    # Billing settings
    billing_settings = Column(JSON, nullable=True, default=lambda: {
        "email_notifications": True,
        "usage_alerts": True,
        "low_balance_threshold": 50,  # Alert when below this many minutes
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
    organization = relationship("Organization", back_populates="billing_account")
    plan = relationship("BillingPlan", back_populates="billing_accounts")
    usage_records = relationship("UsageRecord", back_populates="billing_account", cascade="all, delete-orphan")
    billing_transactions = relationship("BillingTransaction", back_populates="billing_account", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BillingAccount(id={self.id}, organization_id={self.organization_id}, status='{self.status}')>"

    def get_remaining_minutes(self):
        """Calculate remaining minutes for current period."""
        if self.plan.monthly_minutes is None:  # Unlimited plan
            return float('inf')
        return max(0, self.plan.monthly_minutes - self.current_period_minutes_used)

    def is_within_limits(self, additional_minutes=0):
        """Check if usage is within plan limits."""
        if self.plan.monthly_minutes is None:  # Unlimited plan
            return True
        return (self.current_period_minutes_used + additional_minutes) <= self.plan.monthly_minutes

    def needs_payment_method(self):
        """Check if account needs a payment method attached."""
        # Free tier doesn't need payment method
        if self.plan.price_cents == 0:
            return False
        # Paid plans or auto-topup enabled require payment method
        return self.plan.price_cents > 0 or self.auto_topup_enabled


class UsageRecord(Base):
    """
    UsageRecord model tracks detailed usage for billing purposes.
    """

    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True)
    billing_account_id = Column(Integer, ForeignKey("billing_accounts.id"), nullable=False, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True, index=True)  # Optional link to call
    
    # Usage details
    minutes_used = Column(Float, nullable=False)  # Can be fractional
    usage_type = Column(String(20), nullable=False, default="call")  # call, topup_credit, adjustment
    description = Column(Text, nullable=True)
    
    # Billing period this usage belongs to
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    
    # Record metadata
    record_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    # Relationships
    billing_account = relationship("BillingAccount", back_populates="usage_records")
    call = relationship("Call")

    def __repr__(self):
        return f"<UsageRecord(id={self.id}, billing_account_id={self.billing_account_id}, minutes_used={self.minutes_used})>"


class BillingTransaction(Base):
    """
    BillingTransaction model tracks all billing-related transactions.
    """

    __tablename__ = "billing_transactions"

    id = Column(Integer, primary_key=True)
    billing_account_id = Column(Integer, ForeignKey("billing_accounts.id"), nullable=False, index=True)
    
    # Transaction details
    transaction_type = Column(String(20), nullable=False, index=True)  # subscription, topup, refund, adjustment
    amount_cents = Column(Integer, nullable=False)  # Amount in cents
    currency = Column(String(3), nullable=False, default="usd")
    description = Column(Text, nullable=True)
    
    # Stripe information
    stripe_payment_intent_id = Column(String(100), nullable=True, index=True)
    stripe_invoice_id = Column(String(100), nullable=True, index=True)
    stripe_charge_id = Column(String(100), nullable=True, index=True)
    
    # Transaction status
    status = Column(String(20), nullable=False, default="pending")  # pending, succeeded, failed, cancelled
    
    # Usage credit (for top-ups)
    minutes_credited = Column(Integer, nullable=True)  # Minutes added to account
    
    # Billing period
    billing_period_start = Column(DateTime, nullable=True)
    billing_period_end = Column(DateTime, nullable=True)
    
    # Transaction metadata
    transaction_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # Relationships
    billing_account = relationship("BillingAccount", back_populates="billing_transactions")

    def __repr__(self):
        return f"<BillingTransaction(id={self.id}, type='{self.transaction_type}', amount_cents={self.amount_cents}, status='{self.status}')>"
