import datetime
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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Assistant(Base):
    """
    Assistant model represents a virtual assistant configuration
    with its own phone number and settings.
    """

    __tablename__ = "assistants"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    phone_number = Column(String(20), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    # API Keys
    openai_api_key = Column(String(255), nullable=True)
    custom_llm_url = Column(String(255), nullable=True)
    deepgram_api_key = Column(String(255), nullable=True)
    elevenlabs_api_key = Column(String(255), nullable=True)
    twilio_account_sid = Column(String(255), nullable=True)
    twilio_auth_token = Column(String(255), nullable=True)

    llm_settings = Column(JSON, nullable=True, default=lambda: {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful assistant that can answer questions and help with tasks.",
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
