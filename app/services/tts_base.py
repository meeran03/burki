"""
Base TTS interface for modular text-to-speech providers.
This allows easy swapping of TTS providers without changing the core call handling logic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum


class TTSProvider(Enum):
    """Enumeration of supported TTS providers."""
    ELEVENLABS = "elevenlabs"
    DEEPGRAM = "deepgram"
    INWORLD = "inworld"
    OPENAI = "openai"
    AZURE = "azure"
    AWS_POLLY = "aws_polly"
    GOOGLE = "google"


@dataclass
class TTSOptions:
    """Configuration options for TTS that are provider-agnostic."""
    voice_id: str = "default"
    model_id: str = "default"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    latency: int = 1
    conditioning_phrase: Optional[str] = None
    language: Optional[str] = None
    output_format: str = "ulaw_8000"
    sample_rate: int = 8000


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    id: str
    name: str
    gender: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ModelInfo:
    """Information about an available TTS model."""
    id: str
    name: str
    description: Optional[str] = None
    supported_languages: Optional[List[str]] = None


class BaseTTSService(ABC):
    """
    Abstract base class for TTS services.
    All TTS providers should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the TTS service.

        Args:
            call_sid: The unique identifier for this call
            api_key: API key for the TTS provider
            **kwargs: Additional provider-specific configuration
        """
        self.call_sid = call_sid
        self.api_key = api_key
        self.is_connected = False
        self.audio_callback = None
        self.metadata = {}

    @abstractmethod
    async def start_session(
        self,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session.

        Args:
            options: Configuration options for TTS
            audio_callback: Callback function for handling audio data
            metadata: Additional metadata for the session

        Returns:
            bool: Whether the session was started successfully
        """
        pass

    @abstractmethod
    async def process_text(self, text: str, force_flush: bool = False) -> bool:
        """
        Process text and convert to speech.

        Args:
            text: The text to convert to speech
            force_flush: Whether to force immediate conversion

        Returns:
            bool: Whether the text was processed successfully
        """
        pass

    @abstractmethod
    async def stop_synthesis(self) -> None:
        """
        Stop ongoing TTS synthesis.
        Used when an interruption is detected.
        """
        pass

    @abstractmethod
    async def end_session(self) -> None:
        """
        End the TTS session and clean up resources.
        """
        pass

    @classmethod
    @abstractmethod
    def get_available_voices(cls) -> Dict[str, VoiceInfo]:
        """
        Get available voices for this provider.

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information
        """
        pass

    @classmethod
    @abstractmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for this provider.

        Returns:
            Dict[str, ModelInfo]: Mapping of model names to model information
        """
        pass

    @classmethod
    @abstractmethod
    def get_voice_id(cls, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The name of the voice

        Returns:
            str: The voice ID
        """
        pass

    @classmethod
    @abstractmethod
    def get_model_id(cls, model_name: str) -> str:
        """
        Get a model ID by name.

        Args:
            model_name: The name of the model

        Returns:
            str: The model ID
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> TTSProvider:
        """
        Get the TTS provider type.

        Returns:
            TTSProvider: The provider type
        """
        pass

    def is_session_active(self) -> bool:
        """
        Check if the TTS session is active.

        Returns:
            bool: Whether the session is active
        """
        return self.is_connected

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata.

        Returns:
            Dict[str, Any]: Session metadata
        """
        return self.metadata.copy() 