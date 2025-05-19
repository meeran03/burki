import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float, JSON
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
    
    # Configuration
    system_prompt = Column(Text, nullable=True)
    elevenlabs_voice_id = Column(String(100), nullable=True)
    openai_model = Column(String(100), nullable=True)
    openai_temperature = Column(Float, nullable=True)
    openai_max_tokens = Column(Integer, nullable=True)
    
    # Webhook settings
    webhook_url = Column(String(255), nullable=True)
    
    # Endpointing settings
    silence_min_duration_ms = Column(Integer, nullable=True)
    energy_threshold = Column(Integer, nullable=True)
    wait_after_speech_ms = Column(Integer, nullable=True)
    no_punctuation_wait_ms = Column(Integer, nullable=True)
    
    # Interruption settings
    voice_seconds_threshold = Column(Integer, nullable=True)
    word_count_threshold = Column(Integer, nullable=True)
    
    # Call control settings
    end_call_message = Column(String(255), nullable=True)
    max_idle_messages = Column(Integer, nullable=True)
    idle_timeout = Column(Integer, nullable=True)
    
    # Additional settings
    custom_settings = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=True, default=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    calls = relationship("Call", back_populates="assistant", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Assistant(id={self.id}, name='{self.name}', phone_number='{self.phone_number}')>"

class Call(Base):
    """
    Call model represents a phone call with a specific assistant.
    """
    __tablename__ = "calls"
    
    id = Column(Integer, primary_key=True)
    call_sid = Column(String(100), unique=True, nullable=False)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False, index=True)
    to_phone_number = Column(String(20), nullable=False)
    customer_phone_number = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, index=True)  # ongoing, completed, failed
    duration = Column(Integer, nullable=True)  # Duration in seconds
    started_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    call_meta = Column(JSON, nullable=True)  # Changed from 'metadata' to 'call_meta'
    
    # Relationships
    assistant = relationship("Assistant", back_populates="calls")
    recordings = relationship("Recording", back_populates="call", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="call", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Call(id={self.id}, call_sid='{self.call_sid}', status='{self.status}')>"

class Recording(Base):
    """
    Recording model represents an audio recording of a call.
    """
    __tablename__ = "recordings"
    
    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False, index=True)
    file_path = Column(String(255), nullable=False)
    duration = Column(Float, nullable=True)  # Duration in seconds
    format = Column(String(20), nullable=True)  # wav, mp3, etc.
    recording_type = Column(String(20), nullable=True)  # full, segment
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    
    # Relationships
    call = relationship("Call", back_populates="recordings")
    
    def __repr__(self):
        return f"<Recording(id={self.id}, call_id={self.call_id}, format='{self.format}')>"

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