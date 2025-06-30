"""
Inworld.ai TTS service implementation using HTTP streaming API.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import os
import json
import logging
import asyncio
import base64
from typing import Optional, Dict, Any, Callable
import aiohttp

from .tts_base import BaseTTSService, TTSOptions, TTSProvider, VoiceInfo, ModelInfo

# Configure logging
logger = logging.getLogger(__name__)


class InworldTTSService(BaseTTSService):
    """
    Inworld.ai TTS service implementation using HTTP streaming API.
    Manages HTTP requests and text buffering for natural speech synthesis.
    """

    # Available voices mapping based on Inworld.ai API documentation
    _available_voices = {
        # English voices
        "hades": VoiceInfo(
            id="Hades",
            name="hades",
            gender="male",
            language="en",
            description="Deep and commanding male voice (English)",
        ),
        "alex": VoiceInfo(
            id="Alex",
            name="alex",
            gender="male",
            language="en",
            description="Clear and natural male voice (English)",
        ),
        "ashley": VoiceInfo(
            id="Ashley",
            name="ashley",
            gender="female",
            language="en",
            description="Warm and friendly female voice (English)",
        ),
        "aria": VoiceInfo(
            id="Aria",
            name="aria",
            gender="female",
            language="en",
            description="Professional and articulate female voice (English)",
        ),
        "ethan": VoiceInfo(
            id="Ethan",
            name="ethan",
            gender="male",
            language="en",
            description="Friendly and approachable male voice (English)",
        ),
        
        # Spanish voices (real voices from API)
        "diego": VoiceInfo(
            id="Diego",
            name="diego",
            gender="male",
            language="es",
            description="Warm and expressive male voice (Spanish)",
        ),
        "lupita": VoiceInfo(
            id="Lupita",
            name="lupita",
            gender="female",
            language="es",
            description="Elegant and clear female voice (Spanish)",
        ),
        "miguel": VoiceInfo(
            id="Miguel",
            name="miguel",
            gender="male",
            language="es",
            description="Strong and confident male voice (Spanish)",
        ),
        "rafael": VoiceInfo(
            id="Rafael",
            name="rafael",
            gender="male",
            language="es",
            description="Smooth and professional male voice (Spanish)",
        ),
        
        # French voices (real voices from API)
        "alain": VoiceInfo(
            id="Alain",
            name="alain",
            gender="male",
            language="fr",
            description="Sophisticated male voice (French)",
        ),
        "helene": VoiceInfo(
            id="Hélène",
            name="helene",
            gender="female",
            language="fr",
            description="Graceful and articulate female voice (French)",
        ),
        "mathieu": VoiceInfo(
            id="Mathieu",
            name="mathieu",
            gender="male",
            language="fr",
            description="Clear and natural male voice (French)",
        ),
        "etienne": VoiceInfo(
            id="Étienne",
            name="etienne",
            gender="male",
            language="fr",
            description="Expressive and warm male voice (French)",
        ),
        
        # German voices (real voices from API)
        "johanna": VoiceInfo(
            id="Johanna",
            name="johanna",
            gender="female",
            language="de",
            description="Precise and professional female voice (German)",
        ),
        "josef": VoiceInfo(
            id="Josef",
            name="josef",
            gender="male",
            language="de",
            description="Strong and clear male voice (German)",
        ),
        
        # Chinese voices (real voices from API)
        "yichen": VoiceInfo(
            id="Yichen",
            name="yichen",
            gender="male",
            language="zh",
            description="Natural and expressive male voice (Chinese)",
        ),
        "xiaoyin": VoiceInfo(
            id="Xiaoyin",
            name="xiaoyin",
            gender="female",
            language="zh",
            description="Melodic and clear female voice (Chinese)",
        ),
        "xinyi": VoiceInfo(
            id="Xinyi",
            name="xinyi",
            gender="female",
            language="zh",
            description="Gentle and articulate female voice (Chinese)",
        ),
        "jing": VoiceInfo(
            id="Jing",
            name="jing",
            gender="female",
            language="zh",
            description="Smooth and professional female voice (Chinese)",
        ),
        
        # Italian voices
        "marco": VoiceInfo(
            id="Marco",
            name="marco",
            gender="male",
            language="it",
            description="Expressive and warm male voice (Italian)",
        ),
        "giulia": VoiceInfo(
            id="Giulia",
            name="giulia",
            gender="female",
            language="it",
            description="Elegant and musical female voice (Italian)",
        ),
        
        # Portuguese voices
        "joao": VoiceInfo(
            id="Joao",
            name="joao",
            gender="male",
            language="pt",
            description="Rich and warm male voice (Portuguese)",
        ),
        "ana": VoiceInfo(
            id="Ana",
            name="ana",
            gender="female",
            language="pt",
            description="Smooth and expressive female voice (Portuguese)",
        ),
        
        # Japanese voices
        "hiroshi": VoiceInfo(
            id="Hiroshi",
            name="hiroshi",
            gender="male",
            language="ja",
            description="Polite and clear male voice (Japanese)",
        ),
        "sakura": VoiceInfo(
            id="Sakura",
            name="sakura",
            gender="female",
            language="ja",
            description="Gentle and melodic female voice (Japanese)",
        ),
        
        # Dutch voices
        "jan": VoiceInfo(
            id="Jan",
            name="jan",
            gender="male",
            language="nl",
            description="Friendly and clear male voice (Dutch)",
        ),
        "emma": VoiceInfo(
            id="Emma",
            name="emma",
            gender="female",
            language="nl",
            description="Warm and approachable female voice (Dutch)",
        ),
        
        # Korean voices
        "min": VoiceInfo(
            id="Min",
            name="min",
            gender="male",
            language="ko",
            description="Professional and clear male voice (Korean)",
        ),
        "soo": VoiceInfo(
            id="Soo",
            name="soo",
            gender="female",
            language="ko",
            description="Gentle and articulate female voice (Korean)",
        ),
        
        # Polish voices
        "piotr": VoiceInfo(
            id="Piotr",
            name="piotr",
            gender="male",
            language="pl",
            description="Strong and expressive male voice (Polish)",
        ),
        "anna": VoiceInfo(
            id="Anna",
            name="anna",
            gender="female",
            language="pl",
            description="Clear and professional female voice (Polish)",
        ),
        
        # Custom voice option
        "custom": VoiceInfo(
            id="custom",
            name="custom",
            description="Custom voice ID (enter your own voice ID)",
        ),
    }

    # Available models including the latest TTS-1 and TTS-1-Max
    _available_models = {
        "inworld-tts-1": ModelInfo(
            id="inworld-tts-1",
            name="inworld-tts-1",
            description="Inworld's flagship TTS model with realistic, context-aware speech synthesis",
            supported_languages=["en", "zh", "ko", "nl", "fr", "es", "ja", "de", "it", "pl", "pt"],
        ),
        "inworld-tts-1-max": ModelInfo(
            id="inworld-tts-1-max",
            name="inworld-tts-1-max",
            description="Larger, more expressive model (experimental)",
            supported_languages=["en", "zh", "ko", "nl", "fr", "es", "ja", "de", "it", "pl", "pt"],
        ),
    }

    # Supported languages with production readiness status
    _supported_languages = {
        "en": {"name": "English", "status": "production", "description": "All accents supported"},
        "zh": {"name": "Chinese", "status": "production", "description": "Mandarin Chinese"},
        "ko": {"name": "Korean", "status": "production", "description": "Korean"},
        "nl": {"name": "Dutch", "status": "production", "description": "Dutch"},
        "fr": {"name": "French", "status": "production", "description": "French"},
        "es": {"name": "Spanish", "status": "production", "description": "Spanish"},
        "ja": {"name": "Japanese", "status": "experimental", "description": "Japanese (experimental)"},
        "de": {"name": "German", "status": "experimental", "description": "German (experimental)"},
        "it": {"name": "Italian", "status": "experimental", "description": "Italian (experimental)"},
        "pl": {"name": "Polish", "status": "experimental", "description": "Polish (experimental)"},
        "pt": {"name": "Portuguese", "status": "experimental", "description": "Portuguese (experimental)"},
    }

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        language: str = "en",
        custom_voice_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Inworld TTS service.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key (Base64 encoded bearer token)
            voice_id: The voice ID to use (default is hades)
            model_id: The model ID to use (default is inworld-tts-1)
            language: Language code for synthesis (default is "en")
            custom_voice_id: Custom voice ID for voice cloning
            **kwargs: Additional configuration parameters
        """
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv("INWORLD_BEARER_TOKEN")
        if not api_key:
            raise ValueError(
                "INWORLD_BEARER_TOKEN environment variable is not set and no API key provided"
            )

        super().__init__(call_sid=call_sid, api_key=api_key)

        # Inworld API base URL
        self.api_base_url = "https://api.inworld.ai/tts/v1"
        self.synthesize_url = f"{self.api_base_url}/voice:stream"
        self.voices_url = f"{self.api_base_url}/voices"

        # Sentence-ending punctuation
        self.sentence_endings = {".", "!", "?", ":", ";"}

        # HTTP session
        self.session = None
        self.buffer = ""

        # Store TTS settings
        self.language = language
        self.custom_voice_id = custom_voice_id
        
        # Handle custom voice ID override
        if custom_voice_id and custom_voice_id != "" and custom_voice_id != "None":
            self.voice_id = custom_voice_id
        else:
            self.voice_id = voice_id or self.get_voice_id("hades")
            
        self.model_id = model_id or self.get_model_id("inworld-tts-1")

        # Validate language support
        if language not in self._supported_languages:
            logger.warning(f"Language '{language}' not officially supported, using 'en'")
            self.language = "en"

        logger.info(
            f"InworldTTSService initialized for call {call_sid} with voice {self.voice_id}, "
            f"model {self.model_id}, and language {self.language}"
        )

    @property
    def provider(self) -> TTSProvider:
        """Get the TTS provider type."""
        return TTSProvider.INWORLD

    async def start_session(
        self,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session.

        Args:
            options: Configuration options for TTS (optional, will use instance defaults if not provided)
            audio_callback: Callback function for handling audio data
            metadata: Additional metadata for the session

        Returns:
            bool: Whether the session was started successfully
        """
        if not self.call_sid:
            logger.error("Cannot start TTS session without call_sid")
            return False

        try:
            # Use options from parameters or instance settings
            session_options = options or TTSOptions(
                voice_id=self.voice_id,
                model_id=self.model_id,
            )

            # Validate voice ID before using it
            if not self._validate_voice_id(session_options.voice_id):
                logger.warning(
                    f"Invalid voice ID '{session_options.voice_id}', falling back to hades"
                )
                session_options.voice_id = self.get_voice_id("hades")

            # Store options in metadata for reconnection
            session_metadata = metadata or {}
            session_metadata.update(
                {
                    "voice_id": session_options.voice_id,
                    "model_id": session_options.model_id,
                }
            )

            # Store callback and metadata
            self.audio_callback = audio_callback
            self.metadata = session_metadata

            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Update session state
            self.is_connected = True

            logger.info(f"Started Inworld TTS session for call {self.call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error starting Inworld TTS session for call {self.call_sid}: {e}",
                exc_info=True,
            )
            return False

    async def process_text(self, text: str, force_flush: bool = False) -> bool:
        """
        Process text and convert to speech when appropriate.

        Args:
            text: The text to process
            force_flush: Whether to force immediate speech conversion

        Returns:
            bool: Whether the text was processed successfully
        """
        if not self.is_connected or not self.session:
            logger.warning(f"TTS session not connected for call {self.call_sid}")
            return False

        try:
            # Check for flush tag
            if "<flush/>" in text:
                # Split text at flush tag
                parts = text.split("<flush/>")
                
                # Process each part except the last one immediately
                for i, part in enumerate(parts[:-1]):
                    if part.strip():  # Only process non-empty parts
                        self.buffer += part
                        await self._convert_to_speech()
                
                # Add the last part to buffer
                last_part = parts[-1]
                if last_part.strip():  # Only add if not empty
                    self.buffer += last_part
                
                # Always convert the last part if we have anything in buffer
                # This ensures text without punctuation gets processed
                if self.buffer.strip() and (force_flush or self._should_convert(self.buffer) or len(parts) > 1):
                    await self._convert_to_speech()
            else:
                # Add text to buffer
                self.buffer += text

                # Check if we should convert to speech
                if force_flush or self._should_convert(self.buffer):
                    await self._convert_to_speech()

            return True

        except Exception as e:
            logger.error(
                f"Error processing text for call {self.call_sid}: {e}", exc_info=True
            )
            return False

    def _should_convert(self, text: str) -> bool:
        """
        Determine if the buffered text should be converted to speech.

        Args:
            text: The buffered text to check

        Returns:
            bool: Whether the text should be converted
        """
        # Convert if the text ends with sentence-ending punctuation
        return any(text.rstrip().endswith(p) for p in self.sentence_endings)

    def _validate_voice_id(self, voice_id: str) -> bool:
        """
        Validate if a voice ID is valid for Inworld.ai.

        Args:
            voice_id: The voice ID to validate

        Returns:
            bool: Whether the voice ID appears to be valid
        """
        if not voice_id:
            return False

        # Check if it's a known voice name
        if voice_id.lower() in self._available_voices:
            return True

        # Check if it's already a mapped voice ID
        for voice_info in self._available_voices.values():
            if voice_info.id == voice_id:
                return True

        # For now, accept any non-empty string as potentially valid
        return True

    async def _convert_to_speech(self) -> None:
        """
        Convert buffered text to speech using Inworld.ai API.
        """
        if not self.buffer.strip():
            return

        try:
            # Prepare the request payload
            payload = {
                "text": self.buffer.strip(),
                "voiceId": self.voice_id,
                "modelId": self.model_id,
                "language": self.language,
                "audioConfig": {
                    "audioEncoding": "MULAW",
                    "sampleRateHertz": 8000,
                }
            }

            # Prepare headers with Basic authentication
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
            }

            logger.debug(f"Sending TTS request to Inworld for call {self.call_sid}")

            # Make the request
            async with self.session.post(
                self.synthesize_url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 200:
                    # Handle potentially streaming or NDJSON response
                    response_text = await response.text()
                    
                    # Try to parse as single JSON first
                    try:
                        data = json.loads(response_text)
                        await self._process_inworld_response(data)
                    except json.JSONDecodeError:
                        # If single JSON fails, try parsing as NDJSON (newline-delimited JSON)
                        try:
                            await self._process_ndjson_response(response_text)
                        except Exception as e:
                            logger.error(f"Failed to parse Inworld response: {e}")
                            logger.debug(f"Response text: {response_text[:500]}...")
                            
                else:
                    error_text = await response.text()
                    logger.error(f"Inworld API returned status {response.status}: {error_text}")
                    
                    # Try to parse error as JSON
                    try:
                        error_data = json.loads(error_text)
                        if "error" in error_data:
                            logger.error(f"Inworld API error details: {error_data['error']}")
                    except json.JSONDecodeError:
                        pass

            # Clear the buffer
            self.buffer = ""

            logger.debug(f"Completed TTS request to Inworld for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error converting text to speech for call {self.call_sid}: {e}",
                exc_info=True,
            )
            # Reset connection state on error
            self.is_connected = False

    async def _process_inworld_response(self, data: Dict[str, Any]) -> None:
        """Process a single Inworld API response."""
        try:
            # Check for audio content
            if "result" in data and "audioContent" in data["result"]:
                # Decode Base64 audio content
                audio_base64 = data["result"]["audioContent"]
                audio_bytes = base64.b64decode(audio_base64)
                
                # Send audio data to callback
                if self.audio_callback and audio_bytes:
                    await self.audio_callback(
                        audio_bytes,
                        False,  # Not final
                        {"call_sid": self.call_sid},
                    )
            
            # Check for errors
            elif "error" in data:
                logger.error(f"Inworld API error: {data['error']}")
                
        except Exception as e:
            logger.error(f"Error processing Inworld response: {e}")

    async def _process_ndjson_response(self, response_text: str) -> None:
        """Process NDJSON (newline-delimited JSON) response."""
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        data = json.loads(line)
                        await self._process_inworld_response(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing NDJSON response: {e}")

    async def stop_synthesis(self) -> None:
        """
        Stop ongoing TTS synthesis.
        This is used when an interruption is detected to immediately stop the AI from speaking.
        """
        try:
            # Clear any buffered text
            self.buffer = ""

            logger.info(f"Stopped Inworld TTS synthesis for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error stopping Inworld TTS synthesis for call {self.call_sid}: {e}",
                exc_info=True,
            )

    async def end_session(self) -> None:
        """
        End a TTS session and clean up resources.
        """
        try:
            # Convert any remaining buffered text
            if self.buffer.strip():
                await self._convert_to_speech()

            # Signal final audio chunk
            if self.audio_callback:
                await self.audio_callback(
                    b"",  # Empty data for final signal
                    True,  # Final
                    {"call_sid": self.call_sid},
                )

            # Close HTTP session
            if self.session:
                await self.session.close()

            # Reset state
            self.is_connected = False
            self.session = None
            self.buffer = ""

            logger.info(f"Ended Inworld TTS session for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error ending Inworld TTS session for call {self.call_sid}: {e}",
                exc_info=True,
            )

    async def get_available_voices_from_api(self, language: Optional[str] = None) -> Dict[str, VoiceInfo]:
        """
        Fetch available voices from Inworld.ai API for all supported languages.

        Args:
            language: Optional specific language to filter by

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {"Authorization": f"Basic {self.api_key}"}
            voices = {}
            
            # If specific language requested, fetch only that language
            if language:
                languages_to_fetch = [language]
            else:
                # Fetch voices for all supported languages
                languages_to_fetch = list(self._supported_languages.keys())
            
            for lang in languages_to_fetch:
                try:
                    async with self.session.get(
                        f"{self.voices_url}?filter=language={lang}",
                        headers=headers,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for voice in data.get("voices", []):
                                voice_id = voice.get("voiceId", "")
                                display_name = voice.get("displayName", voice_id)
                                description = voice.get("description", "")
                                voice_languages = voice.get("languages", [lang])
                                gender = voice.get("gender", "").lower()
                                
                                # Create unique key with language prefix to avoid conflicts
                                voice_key = f"{lang}_{voice_id.lower()}" if language is None else voice_id.lower()
                                
                                voices[voice_key] = VoiceInfo(
                                    id=voice_id,
                                    name=voice_key,
                                    language=voice_languages[0] if voice_languages else lang,
                                    gender=gender if gender in ["male", "female"] else None,
                                    description=f"{description or display_name} ({self._supported_languages.get(lang, {}).get('name', lang)})",
                                )
                        else:
                            logger.warning(f"Failed to fetch voices for language {lang}: HTTP {response.status}")
                            
                except Exception as e:
                    logger.warning(f"Error fetching voices for language {lang}: {e}")
                    continue
            
            # Return the collected voices or fallback to static voices if none found
            return voices if voices else self._available_voices.copy()
                    
        except Exception as e:
            logger.error(f"Error fetching voices from Inworld API: {e}")
            return self._available_voices.copy()

    @classmethod
    def get_available_voices(cls) -> Dict[str, VoiceInfo]:
        """
        Get available voices for Inworld TTS.

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information
        """
        return cls._available_voices.copy()

    @classmethod
    def get_voices_for_language(cls, language: str) -> Dict[str, VoiceInfo]:
        """
        Get available voices for a specific language.

        Args:
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information for the specified language
        """
        filtered_voices = {}
        for voice_name, voice_info in cls._available_voices.items():
            # Include voices for the specified language, plus the custom voice option
            if voice_info.language == language or voice_name == "custom":
                filtered_voices[voice_name] = voice_info
        
        return filtered_voices

    @classmethod
    def get_supported_languages(cls) -> Dict[str, Dict[str, str]]:
        """
        Get supported languages for Inworld TTS.

        Returns:
            Dict[str, Dict[str, str]]: Mapping of language codes to language information
        """
        return cls._supported_languages.copy()

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for Inworld TTS.

        Returns:
            Dict[str, ModelInfo]: Mapping of model names to model information
        """
        return cls._available_models.copy()

    @classmethod
    def get_voice_id(cls, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The name of the voice to use

        Returns:
            str: The voice ID
        """
        voice_name = voice_name.lower()
        if voice_name in cls._available_voices:
            voice_info = cls._available_voices[voice_name]
            # Handle custom voice placeholder
            if voice_info.id == "custom":
                return voice_name  # Return the original name for custom voices
            return voice_info.id
        else:
            # Return as-is if not found in mapping (supports custom voice IDs)
            return voice_name

    @classmethod
    def is_custom_voice(cls, voice_id: str) -> bool:
        """
        Check if a voice ID is a custom voice.

        Args:
            voice_id: The voice ID to check

        Returns:
            bool: True if it's a custom voice ID
        """
        # Get all known voice IDs (both names and IDs) but exclude "custom" placeholder
        known_voice_names = {name for name in cls._available_voices.keys() if name != "custom"}
        known_voice_ids = {voice_info.id for voice_info in cls._available_voices.values() if voice_info.id != "custom"}
        
        # If it's in our predefined voices (either by name or ID), it's not custom
        return voice_id not in known_voice_names and voice_id not in known_voice_ids

    @classmethod
    def get_model_id(cls, model_name: str) -> str:
        """
        Get a model ID by name.

        Args:
            model_name: The name of the model to use

        Returns:
            str: The model ID
        """
        model_name = model_name.lower()
        if model_name in cls._available_models:
            return cls._available_models[model_name].id
        else:
            # Default to the primary model if unknown
            return "inworld-tts-1" 