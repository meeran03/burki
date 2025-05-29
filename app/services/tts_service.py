"""
Modular TTS service that supports multiple providers.
This replaces the old ElevenLabs-specific implementation with a provider-agnostic interface.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import logging
from typing import Optional, Dict, Any, Callable

from .tts_base import BaseTTSService, TTSOptions, TTSProvider
from .tts_factory import TTSFactory

# Configure logging
logger = logging.getLogger(__name__)


class TTSService:
    """
    Modular TTS service that can work with multiple providers.
    This class acts as a wrapper around the actual provider-specific implementation.
    """

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        latency: int = 1,
        provider: TTSProvider = TTSProvider.ELEVENLABS,
    ):
        """
        Initialize the TTS service.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key for this call
            voice_id: The voice ID to use (default is Rachel)
            model_id: The model ID to use (default is Turbo)
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice similarity boost (0.0-1.0)
            style: Speaking style (0.0-1.0)
            use_speaker_boost: Enhance speech clarity and fidelity
            latency: 1-4, where 1 is lowest latency
            provider: The TTS provider to use
        """
        self.call_sid = call_sid
        self.provider_type = provider

        # Create the actual TTS service instance using the factory
        self._tts_service = TTSFactory.create_tts_service(
            provider=provider,
            call_sid=call_sid,
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
            latency=latency,
        )

        logger.info(f"TTSService initialized with provider {provider.value} for call {call_sid}")

    @classmethod
    def create_from_assistant(
        cls,
        call_sid: Optional[str] = None,
        assistant: Optional[Any] = None,
        **kwargs
    ) -> "TTSService":
        """
        Create a TTS service instance from assistant settings.

        Args:
            call_sid: The unique identifier for this call
            assistant: Assistant object containing TTS configuration
            **kwargs: Additional configuration overrides

        Returns:
            TTSService: A configured TTS service instance
        """
        # Create the provider-specific service using the factory
        tts_service_impl = TTSFactory.create_from_settings(
            call_sid=call_sid,
            assistant=assistant,
            **kwargs
        )

        # Create a wrapper instance
        wrapper = cls.__new__(cls)
        wrapper.call_sid = call_sid
        wrapper.provider_type = tts_service_impl.provider
        wrapper._tts_service = tts_service_impl

        logger.info(f"TTSService created from assistant settings with provider {wrapper.provider_type.value} for call {call_sid}")
        return wrapper

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
        return await self._tts_service.start_session(options, audio_callback, metadata)

    async def process_text(self, text: str, force_flush: bool = False) -> bool:
        """
        Process text and convert to speech.

        Args:
            text: The text to process
            force_flush: Whether to force immediate speech conversion

        Returns:
            bool: Whether the text was processed successfully
        """
        return await self._tts_service.process_text(text, force_flush)

    async def stop_synthesis(self) -> None:
        """
        Stop ongoing TTS synthesis.
        Used when an interruption is detected.
        """
        await self._tts_service.stop_synthesis()

    async def end_session(self) -> None:
        """
        End the TTS session and clean up resources.
        """
        await self._tts_service.end_session()

    @property
    def is_connected(self) -> bool:
        """
        Check if the TTS session is connected.

        Returns:
            bool: Whether the session is connected
        """
        return self._tts_service.is_session_active()

    @property
    def provider(self) -> TTSProvider:
        """
        Get the TTS provider type.

        Returns:
            TTSProvider: The provider type
        """
        return self._tts_service.provider

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata.

        Returns:
            Dict[str, Any]: Session metadata
        """
        return self._tts_service.get_session_metadata()

    @classmethod
    def get_voice_id(cls, voice_name: str, provider: TTSProvider = TTSProvider.ELEVENLABS) -> str:
        """
        Get a voice ID by name for a specific provider.

        Args:
            voice_name: The name of the voice to use
            provider: The TTS provider to use

        Returns:
            str: The voice ID
        """
        return TTSFactory._get_voice_id_for_provider(provider, voice_name)

    @classmethod
    def get_model_id(cls, model_name: str, provider: TTSProvider = TTSProvider.ELEVENLABS) -> str:
        """
        Get a model ID by name for a specific provider.

        Args:
            model_name: The name of the model to use
            provider: The TTS provider to use

        Returns:
            str: The model ID
        """
        return TTSFactory._get_model_id_for_provider(provider, model_name)

    @classmethod
    def get_available_providers(cls) -> list[TTSProvider]:
        """
        Get a list of available TTS providers.

        Returns:
            list[TTSProvider]: List of available providers
        """
        return TTSFactory.get_available_providers()

    @classmethod
    def get_available_voices(cls, provider: TTSProvider = TTSProvider.ELEVENLABS) -> Dict[str, Any]:
        """
        Get available voices for a specific provider.

        Args:
            provider: The TTS provider

        Returns:
            Dict[str, Any]: Mapping of voice names to voice information
        """
        return TTSFactory.get_available_voices(provider)

    @classmethod
    def get_available_models(cls, provider: TTSProvider = TTSProvider.ELEVENLABS) -> Dict[str, Any]:
        """
        Get available models for a specific provider.

        Args:
            provider: The TTS provider

        Returns:
            Dict[str, Any]: Mapping of model names to model information
        """
        return TTSFactory.get_available_models(provider)

    def __getattr__(self, name):
        """
        Delegate any unknown attributes to the underlying TTS service.
        This provides backward compatibility for any direct access to the old implementation.
        """
        return getattr(self._tts_service, name)


# Legacy compatibility: Export the old class names and interfaces
# This ensures existing code continues to work without changes

# For code that imports TTSOptions directly
from .tts_base import TTSOptions

# For backward compatibility with the old available_voices and available_models class attributes
def _get_legacy_available_voices():
    """Get available voices in the old format for backward compatibility."""
    voices = TTSFactory.get_available_voices(TTSProvider.ELEVENLABS)
    # Convert to old format (dict with string keys and string values)
    return {name: voice_info.id for name, voice_info in voices.items()}

def _get_legacy_available_models():
    """Get available models in the old format for backward compatibility."""
    models = TTSFactory.get_available_models(TTSProvider.ELEVENLABS)
    # Convert to old format (dict with string keys and string values)
    return {name: model_info.id for name, model_info in models.items()}

# Add class attributes for backward compatibility
TTSService.available_voices = _get_legacy_available_voices()
TTSService.available_models = _get_legacy_available_models()

# Log the migration
logger.info("TTSService has been migrated to use modular provider architecture")
logger.info(f"Available voices: {list(TTSService.available_voices.keys())}")
logger.info(f"Available models: {list(TTSService.available_models.keys())}")
