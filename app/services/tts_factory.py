"""
TTS Factory for creating TTS service instances based on provider type.
This allows easy switching between different TTS providers.
"""

import os
import logging
from typing import Optional, Dict, Any

from .tts_base import BaseTTSService, TTSProvider, TTSOptions
from .tts_elevenlabs import ElevenLabsTTSService
from .tts_deepgram import DeepgramTTSService
from .tts_inworld import InworldTTSService
from .tts_resemble import ResembleTTSService

logger = logging.getLogger(__name__)


class TTSFactory:
    """
    Factory class for creating TTS service instances.
    """

    # Registry of available TTS providers
    _providers = {
        TTSProvider.ELEVENLABS: ElevenLabsTTSService,
        TTSProvider.DEEPGRAM: DeepgramTTSService,
        TTSProvider.INWORLD: InworldTTSService,
        TTSProvider.RESEMBLE: ResembleTTSService,
        # Future providers can be added here:
        # TTSProvider.OPENAI: OpenAITTSService,
        # TTSProvider.AZURE: AzureTTSService,
        # TTSProvider.AWS_POLLY: AWSPollyTTSService,
        # TTSProvider.GOOGLE: GoogleTTSService,
    }

    @classmethod
    def create_tts_service(
        cls,
        provider: TTSProvider,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseTTSService:
        """
        Create a TTS service instance for the specified provider.

        Args:
            provider: The TTS provider to use
            call_sid: The unique identifier for this call
            api_key: API key for the TTS provider
            **kwargs: Additional provider-specific configuration

        Returns:
            BaseTTSService: An instance of the specified TTS provider

        Raises:
            ValueError: If the provider is not supported
            Exception: If there's an error creating the service instance
        """
        if provider not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ValueError(
                f"Unsupported TTS provider: {provider}. "
                f"Available providers: {[p.value for p in available_providers]}"
            )

        try:
            service_class = cls._providers[provider]
            return service_class(
                call_sid=call_sid,
                api_key=api_key,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error creating TTS service for provider {provider}: {e}")
            raise

    @classmethod
    def create_from_settings(
        cls,
        call_sid: Optional[str] = None,
        assistant: Optional[Any] = None,
        **kwargs
    ) -> BaseTTSService:
        """
        Create a TTS service instance based on assistant settings.

        Args:
            call_sid: The unique identifier for this call
            assistant: Assistant object containing TTS configuration
            **kwargs: Additional configuration overrides

        Returns:
            BaseTTSService: An instance of the configured TTS provider
        """
        # Default to ElevenLabs
        provider = TTSProvider.ELEVENLABS
        api_key = None
        tts_config = {}

        # Extract configuration from assistant
        if assistant:
            # Determine provider based on available API keys and settings
            if hasattr(assistant, 'tts_settings') and assistant.tts_settings:
                provider_setting = assistant.tts_settings.get("provider", "elevenlabs").lower()
                if provider_setting == "deepgram":
                    provider = TTSProvider.DEEPGRAM
                    api_key = assistant.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
                elif provider_setting == "inworld":
                    provider = TTSProvider.INWORLD
                    api_key = assistant.inworld_bearer_token or os.getenv("INWORLD_BEARER_TOKEN")
                elif provider_setting == "resemble":
                    provider = TTSProvider.RESEMBLE
                    api_key = assistant.resemble_api_key or os.getenv("RESEMBLE_API_KEY")
                else:
                    provider = TTSProvider.ELEVENLABS
                    api_key = assistant.elevenlabs_api_key
            elif hasattr(assistant, 'deepgram_api_key') and assistant.deepgram_api_key:
                # If no provider specified but Deepgram API key is available, prefer Deepgram
                provider = TTSProvider.DEEPGRAM
                api_key = assistant.deepgram_api_key
            elif hasattr(assistant, 'inworld_bearer_token') and assistant.inworld_bearer_token:
                # If no provider specified but Inworld bearer token is available, prefer Inworld
                provider = TTSProvider.INWORLD
                api_key = assistant.inworld_bearer_token
            elif hasattr(assistant, 'resemble_api_key') and assistant.resemble_api_key:
                # If no provider specified but Resemble API key is available, prefer Resemble
                provider = TTSProvider.RESEMBLE
                api_key = assistant.resemble_api_key
            else:
                # Default to ElevenLabs
                provider = TTSProvider.ELEVENLABS
                api_key = assistant.elevenlabs_api_key

            # Extract TTS settings based on provider
            if assistant.tts_settings:
                settings = assistant.tts_settings
                
                if provider == TTSProvider.DEEPGRAM:
                    # Extract provider_config for Deepgram settings
                    provider_config = settings.get("provider_config", {})
                    tts_config.update({
                        "voice_id": cls._get_voice_id_for_provider(
                            provider, settings.get("voice_id", "asteria")
                        ),
                        "model_id": cls._get_model_id_for_provider(
                            provider, settings.get("model_id", "aura-2")
                        ),
                        "encoding": provider_config.get("encoding", "mulaw"),
                        "sample_rate": provider_config.get("sample_rate", 8000),
                    })
                elif provider == TTSProvider.INWORLD:
                    # Extract provider_config for Inworld settings
                    provider_config = settings.get("provider_config", {})
                    tts_config.update({
                        "voice_id": cls._get_voice_id_for_provider(
                            provider, settings.get("voice_id", "hades")
                        ),
                        "model_id": cls._get_model_id_for_provider(
                            provider, settings.get("model_id", "inworld-tts-1")
                        ),
                        "language": provider_config.get("language", "en"),
                        "custom_voice_id": provider_config.get("custom_voice_id"),
                    })
                elif provider == TTSProvider.RESEMBLE:
                    # Audio format is fixed for Twilio compatibility (MULAW, 8kHz, raw)
                    tts_config.update({
                        "voice_id": cls._get_voice_id_for_provider(
                            provider, settings.get("voice_id", "")
                        ),
                        "model_id": cls._get_model_id_for_provider(
                            provider, settings.get("model_id", "default")
                        ),
                        "project_uuid": settings.get("project_uuid") or os.getenv("RESEMBLE_PROJECT_UUID"),
                    })
                else:  # ElevenLabs
                    provider_config = settings.get("provider_config", {})
                    tts_config.update({
                        "voice_id": cls._get_voice_id_for_provider(
                            provider, settings.get("voice_id", "rachel")
                        ),
                        "model_id": cls._get_model_id_for_provider(
                            provider, settings.get("model_id", "turbo")
                        ),
                        "stability": settings.get("stability", 0.5),
                        "similarity_boost": settings.get("similarity_boost", 0.75),
                        "style": settings.get("style", 0.0),
                        "use_speaker_boost": settings.get("use_speaker_boost", True),
                        "latency": settings.get("latency", 1),
                        "language": provider_config.get("language", "en"),
                    })

        # Override with any provided kwargs
        tts_config.update(kwargs)

        return cls.create_tts_service(
            provider=provider,
            call_sid=call_sid,
            api_key=api_key,
            **tts_config
        )

    @classmethod
    def _get_voice_id_for_provider(cls, provider: TTSProvider, voice_name: str) -> str:
        """
        Get the voice ID for a specific provider.

        Args:
            provider: The TTS provider
            voice_name: The voice name to resolve

        Returns:
            str: The voice ID for the provider
        """
        if provider not in cls._providers:
            return voice_name

        service_class = cls._providers[provider]
        resolved_voice_id = service_class.get_voice_id(voice_name)
        
        # Provider-specific validation and fallbacks
        if provider == TTSProvider.ELEVENLABS and resolved_voice_id == voice_name:
            # Check if this is a valid ElevenLabs voice ID format (28 character alphanumeric)
            if len(voice_name) == 20 and voice_name.isalnum():
                logger.warning(f"Using custom ElevenLabs voice ID: {voice_name}")
                return voice_name
            elif voice_name.lower() not in service_class.get_available_voices():
                # Unknown voice name and not a valid custom ID format, use default
                logger.warning(f"Unknown voice name '{voice_name}' for {provider.value}, using default 'rachel'")
                return service_class.get_voice_id("rachel")
        elif provider == TTSProvider.DEEPGRAM and resolved_voice_id == voice_name:
            # Check if this is a valid Deepgram voice name or ID
            available_voices = service_class.get_available_voices()
            if voice_name.lower() not in available_voices:
                # Check if it looks like a valid Aura model ID
                if voice_name.startswith("aura-") and voice_name.endswith("-en"):
                    logger.warning(f"Using custom Deepgram voice ID: {voice_name}")
                    return voice_name
                else:
                    # Unknown voice name, use default
                    logger.warning(f"Unknown voice name '{voice_name}' for {provider.value}, using default 'asteria'")
                    return service_class.get_voice_id("asteria")
        elif provider == TTSProvider.INWORLD and resolved_voice_id == voice_name:
            # Check if this is a valid Inworld voice name or ID
            available_voices = service_class.get_available_voices()
            if voice_name.lower() not in available_voices:
                # Unknown voice name, use default
                logger.warning(f"Unknown voice name '{voice_name}' for {provider.value}, using default 'hades'")
                return service_class.get_voice_id("hades")
        
        return resolved_voice_id

    @classmethod
    def _get_model_id_for_provider(cls, provider: TTSProvider, model_name: str) -> str:
        """
        Get the model ID for a specific provider.

        Args:
            provider: The TTS provider
            model_name: The model name to resolve

        Returns:
            str: The model ID for the provider
        """
        if provider not in cls._providers:
            return model_name

        service_class = cls._providers[provider]
        return service_class.get_model_id(model_name)

    @classmethod
    def get_available_providers(cls) -> list[TTSProvider]:
        """
        Get a list of available TTS providers.

        Returns:
            list[TTSProvider]: List of available providers
        """
        return list(cls._providers.keys())

    @classmethod
    def get_available_voices(cls, provider: TTSProvider) -> Dict[str, Any]:
        """
        Get available voices for a specific provider.

        Args:
            provider: The TTS provider

        Returns:
            Dict[str, Any]: Mapping of voice names to voice information

        Raises:
            ValueError: If the provider is not supported
        """
        if provider not in cls._providers:
            raise ValueError(f"Unsupported TTS provider: {provider}")

        service_class = cls._providers[provider]
        return service_class.get_available_voices()

    @classmethod
    def get_voices_for_language(cls, provider: TTSProvider, language: str) -> Dict[str, Any]:
        """
        Get available voices for a specific provider and language.

        Args:
            provider: The TTS provider
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Dict[str, Any]: Mapping of voice names to voice information for the specified language

        Raises:
            ValueError: If the provider is not supported
        """
        if provider not in cls._providers:
            raise ValueError(f"Unsupported TTS provider: {provider}")

        service_class = cls._providers[provider]
        
        # Check if the provider supports language filtering
        if hasattr(service_class, 'get_voices_for_language'):
            return service_class.get_voices_for_language(language)
        else:
            # Fallback to all voices for providers that don't support language filtering
            return service_class.get_available_voices()

    @classmethod
    def get_available_models(cls, provider: TTSProvider) -> Dict[str, Any]:
        """
        Get available models for a specific provider.

        Args:
            provider: The TTS provider

        Returns:
            Dict[str, Any]: Mapping of model names to model information

        Raises:
            ValueError: If the provider is not supported
        """
        if provider not in cls._providers:
            raise ValueError(f"Unsupported TTS provider: {provider}")

        service_class = cls._providers[provider]
        return service_class.get_available_models()

    @classmethod
    def register_provider(
        cls, 
        provider: TTSProvider, 
        service_class: type
    ) -> None:
        """
        Register a new TTS provider.

        Args:
            provider: The TTS provider enum
            service_class: The service class that implements BaseTTSService

        Raises:
            ValueError: If the service class doesn't inherit from BaseTTSService
        """
        if not issubclass(service_class, BaseTTSService):
            raise ValueError(
                f"Service class must inherit from BaseTTSService, "
                f"got {service_class.__name__}"
            )

        cls._providers[provider] = service_class
        logger.info(f"Registered TTS provider: {provider.value}")


# For backward compatibility, create a function that matches the old TTSService interface
def create_tts_service(
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
) -> BaseTTSService:
    """
    Create a TTS service instance with backward compatibility.

    Args:
        call_sid: The unique identifier for this call
        api_key: API key for the TTS provider
        voice_id: The voice ID to use
        model_id: The model ID to use
        stability: Voice stability (0.0-1.0)
        similarity_boost: Voice similarity boost (0.0-1.0)
        style: Speaking style (0.0-1.0)
        use_speaker_boost: Enhance speech clarity and fidelity
        latency: 1-4, where 1 is lowest latency
        provider: The TTS provider to use

    Returns:
        BaseTTSService: An instance of the specified TTS provider
    """
    return TTSFactory.create_tts_service(
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