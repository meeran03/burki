"""
This file contains the multi-provider LLM service for handling various LLM providers
including OpenAI, Anthropic, Google Gemini, xAI Grok, and Groq.
"""

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import httpx

# Import all LLM provider clients
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model = config.get("model")
        self.custom_config = config.get("custom_config", {})
    
    @abstractmethod
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript and generate streaming response."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncOpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 150),
                top_p=settings.get("top_p", 1.0),
                frequency_penalty=settings.get("frequency_penalty", 0.0),
                presence_penalty=settings.get("presence_penalty", 0.0),
                stop=settings.get("stop_sequences") or None,
            )

            collected_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    await response_callback(content, False, {})

            if collected_response:
                await response_callback("", True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncAnthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using Anthropic Claude."""
        try:
            # Convert OpenAI format messages to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)
            system_prompt = self._extract_system_prompt(messages)
            
            stream = await self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_prompt,
                stream=True,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 150),
                top_p=settings.get("top_p", 1.0),
                stop_sequences=settings.get("stop_sequences", []),
            )

            collected_response = ""
            async for chunk in stream:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    content = chunk.delta.text
                    collected_response += content
                    await response_callback(content, False, {})

            if collected_response:
                await response_callback("", True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )
    
    def _convert_messages_to_anthropic(self, messages: list) -> list:
        """Convert OpenAI format messages to Anthropic format."""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System messages are handled separately
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
        return anthropic_messages
    
    def _extract_system_prompt(self, messages: list) -> str:
        """Extract system prompt from messages."""
        for msg in messages:
            if msg["role"] == "system":
                return msg["content"]
        return "You are a helpful assistant."


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider using OpenAI compatibility."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncOpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Use Gemini's OpenAI-compatible endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using Google Gemini."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model or "gemini-2.0-flash",
                messages=messages,
                stream=True,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 150),
                top_p=settings.get("top_p", 1.0),
                stop=settings.get("stop_sequences") or None,
            )

            collected_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    await response_callback(content, False, {})

            if collected_response:
                await response_callback("", True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"Gemini error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )


class XAIProvider(BaseLLMProvider):
    """xAI Grok LLM provider using OpenAI compatibility."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncOpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Use xAI's OpenAI-compatible endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://api.x.ai/v1"
        )
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using xAI Grok."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model or "grok-beta",
                messages=messages,
                stream=True,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 150),
                top_p=settings.get("top_p", 1.0),
                frequency_penalty=settings.get("frequency_penalty", 0.0),
                presence_penalty=settings.get("presence_penalty", 0.0),
                stop=settings.get("stop_sequences") or None,
            )

            collected_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    await response_callback(content, False, {})

            if collected_response:
                await response_callback("", True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"xAI error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if AsyncGroq is None:
            raise ImportError("groq package not installed. Run: pip install groq")
        
        self.client = AsyncGroq(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using Groq."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model or "llama-3.3-70b-versatile",
                messages=messages,
                stream=True,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 150),
                top_p=settings.get("top_p", 1.0),
                frequency_penalty=settings.get("frequency_penalty", 0.0),
                presence_penalty=settings.get("presence_penalty", 0.0),
                stop=settings.get("stop_sequences") or None,
            )

            collected_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    await response_callback(content, False, {})

            if collected_response:
                await response_callback("", True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"Groq error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )


class CustomLLMProvider(BaseLLMProvider):
    """Custom LLM provider for custom endpoints (backward compatibility)."""
    
    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using custom LLM endpoint."""
        try:
            # Extract call metadata from settings
            to_number = settings.get("to_number", "")
            from_number = settings.get("from_number", "")
            call_sid = settings.get("call_sid", "")
            
            payload = {
                "messages": messages,
                "phoneNumber": {"number": to_number or ""},
                "call": {
                    "phoneCallProviderId": call_sid or "",
                    "customer": {"number": from_number or ""},
                },
            }

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                ) as response:
                    response.raise_for_status()
                    await self._process_custom_llm_response(response, response_callback)

        except Exception as e:
            logger.error(f"Custom LLM error: {e}")
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"error": str(e)},
            )
    
    async def _process_custom_llm_response(
        self,
        response: httpx.Response,
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process streaming response from custom LLM endpoint."""
        collected_response = ""

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("choices") and data["choices"][0].get("delta"):
                        delta = data["choices"][0]["delta"]
                        if delta.get("content") is not None:
                            content = delta["content"]
                            collected_response += content
                            await response_callback(content, False, {})
                except json.JSONDecodeError:
                    continue

        if collected_response:
            await response_callback("", True, {"full_response": collected_response})


class LLMService:
    """
    Multi-provider LLM service for handling various LLM providers.
    """

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "xai": XAIProvider,
        "groq": GroqProvider,
        "custom": CustomLLMProvider,
    }

    def __init__(
        self,
        call_sid: Optional[str] = None,
        to_number: Optional[str] = None,
        from_number: Optional[str] = None,
        assistant: Optional[Any] = None,
    ):
        """
        Initialize the LLM service.

        Args:
            call_sid: The unique identifier for this call
            to_number: The destination phone number
            from_number: The caller's phone number
            assistant: Assistant object containing LLM configuration
        """
        self.call_sid = call_sid
        self.to_number = to_number
        self.from_number = from_number
        self.assistant = assistant

        # Initialize LLM provider
        self.provider = self._initialize_provider()

        # Default system prompt
        self.default_system_prompt = """You are a helpful AI assistant for a customer service call center. 
        Your role is to assist customers professionally and efficiently. 
        Keep responses concise, clear, and focused on resolving customer needs."""

        # Initialize conversation history with system prompt
        system_prompt = self._get_system_prompt()
        self.conversation_history = [{"role": "system", "content": system_prompt}]

        if call_sid:
            logger.info(f"Started new conversation for call {call_sid} using {self._get_provider_name()}")

    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate LLM provider based on assistant configuration."""
        if not self.assistant:
            # Fallback to OpenAI with environment variable
            config = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4o-mini",
            }
            return OpenAIProvider(config)

        # Check for legacy configuration (backward compatibility)
        if hasattr(self.assistant, 'custom_llm_url') and self.assistant.custom_llm_url:
            config = {
                "base_url": self.assistant.custom_llm_url,
            }
            return CustomLLMProvider(config)

        # Use new provider configuration
        provider_name = getattr(self.assistant, 'llm_provider', 'openai')
        provider_config = getattr(self.assistant, 'llm_provider_config', {})
                                
        # Fallback to legacy OpenAI key if no provider config
        if not provider_config.get("api_key") and hasattr(self.assistant, 'openai_api_key'):
            provider_config = {
                "api_key": self.assistant.openai_api_key,
                "model": "gpt-4o-mini",
            }
            provider_name = "openai"

        provider_class = self.PROVIDERS.get(provider_name, OpenAIProvider)
        return provider_class(provider_config)

    def _get_provider_name(self) -> str:
        """Get the name of the current provider."""
        if not self.assistant:
            return "openai"
        return getattr(self.assistant, 'llm_provider', 'openai')

    def _get_system_prompt(self) -> str:
        """Get the system prompt from assistant configuration."""
        if not self.assistant:
            return self.default_system_prompt

        llm_settings = getattr(self.assistant, 'llm_settings', {})
        return llm_settings.get("system_prompt", self.default_system_prompt)

    async def process_transcript(
        self,
        transcript: str,
        is_final: bool,
        metadata: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """
        Process a transcript and get streaming response from LLM.

        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcript
            metadata: Additional metadata about the transcript
            response_callback: Callback function to handle streaming responses
        """
        # Only process final transcripts to avoid too many API calls
        if not is_final:
            return

        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcript})

            # Get LLM settings
            llm_settings = {}
            if self.assistant:
                llm_settings = getattr(self.assistant, 'llm_settings', {})

            # Add call metadata to settings for custom LLM providers that need it
            enhanced_settings = llm_settings.copy()
            enhanced_settings.update({
                "call_sid": self.call_sid,
                "to_number": self.to_number,
                "from_number": self.from_number,
                "assistant": self.assistant,
            })

            # Add call metadata to response callback
            enhanced_callback = self._create_enhanced_callback(response_callback)

            # Process using the configured provider
            await self.provider.process_transcript(
                self.conversation_history,
                enhanced_settings,
                enhanced_callback
            )

        except Exception as e:
            logger.error(
                f"Error processing transcript for call {self.call_sid}: {e}",
                exc_info=True,
            )
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"call_sid": self.call_sid, "error": str(e)},
            )

    def _create_enhanced_callback(
        self, 
        original_callback: Callable[[str, bool, Dict[str, Any]], None]
    ) -> Callable[[str, bool, Dict[str, Any]], None]:
        """Create an enhanced callback that adds call metadata and updates conversation history."""
        
        async def enhanced_callback(content: str, is_final: bool, metadata: Dict[str, Any]) -> None:
            # Add call metadata
            metadata["call_sid"] = self.call_sid
            
            # Update conversation history if this is the final response
            if is_final and metadata.get("full_response"):
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": metadata["full_response"]
                })
            
            # Call the original callback
            await original_callback(content, is_final, metadata)
        
        return enhanced_callback

    def get_conversation_history(self) -> list:
        """
        Get the conversation history.

        Returns:
            list: The conversation history
        """
        return self.conversation_history

    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported LLM providers."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def get_provider_requirements(cls, provider: str) -> Dict[str, Any]:
        """Get the configuration requirements for a specific provider."""
        requirements = {
            "openai": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                "pip_install": "openai",
            },
            "anthropic": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
                "pip_install": "anthropic",
            },
            "gemini": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["gemini-2.0-flash", "gemini-1.5-pro"],
                "pip_install": "openai",
                "note": "Uses OpenAI compatibility mode",
            },
            "xai": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["grok-beta", "grok-2-1212"],
                "pip_install": "openai",
                "note": "Uses OpenAI compatibility mode",
            },
            "groq": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                "pip_install": "groq",
            },
            "custom": {
                "required_fields": ["base_url"],
                "optional_fields": ["api_key", "model"],
                "default_models": [],
                "pip_install": "httpx",
                "note": "For custom LLM endpoints",
            },
        }
        return requirements.get(provider, {})
