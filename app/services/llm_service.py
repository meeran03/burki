"""
This file contains the multi-provider LLM service for handling various LLM providers
including OpenAI, Anthropic, Google Gemini, xAI Grok, and Groq using LangChain.
"""

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
import os
import json
import logging
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import httpx

# LangChain imports
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# RAG imports
try:
    from app.services.rag_service import RAGService

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Import tools
from app.tools.basic import get_all_tools

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

    async def _handle_tool_call(
        self,
        tool_call: Dict[str, Any],
        assistant: Any,
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Handle tool call execution - shared across all providers."""
        try:
            import json

            # Parse arguments if they exist
            args = {}
            if tool_call.get("arguments"):
                try:
                    args = json.loads(tool_call["arguments"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool arguments: {tool_call['arguments']}"
                    )

            tool_name = tool_call["name"]
            logger.info(f"Processing {tool_name} tool call")

            if tool_name == "endCall":
                # Get custom end call message from tools_settings
                end_call_message = "Thank you for calling. Goodbye!"
                if assistant:
                    tools_settings = getattr(assistant, "tools_settings", {}) or {}
                    end_call_config = tools_settings.get("end_call", {})
                    custom_message = end_call_config.get("custom_message")
                    if custom_message:
                        end_call_message = custom_message
                    # Fallback to legacy field for backward compatibility
                    elif (
                        hasattr(assistant, "end_call_message")
                        and assistant.end_call_message
                    ):
                        end_call_message = assistant.end_call_message

                # First send the custom message to be spoken
                await response_callback(
                    end_call_message,
                    False,  # Not final yet, speak the message first
                    {"speak_before_action": True, "action": "end_call"},
                )

                # Then send the action to end the call
                await response_callback(
                    "",
                    True,
                    {"action": "end_call"},
                )

            elif tool_name == "transferCall":
                destination = args.get("destination")
                if destination:
                    # Get custom transfer call message from tools_settings
                    transfer_call_message = "Please hold while I transfer your call."
                    if assistant:
                        tools_settings = getattr(assistant, "tools_settings", {}) or {}
                        transfer_call_config = tools_settings.get("transfer_call", {})
                        custom_message = transfer_call_config.get("custom_message")
                        if custom_message:
                            transfer_call_message = custom_message
                        # Fallback to legacy field for backward compatibility
                        elif (
                            hasattr(assistant, "transfer_call_message")
                            and assistant.transfer_call_message
                        ):
                            transfer_call_message = assistant.transfer_call_message

                    # First send the custom message to be spoken
                    await response_callback(
                        transfer_call_message,
                        False,  # Not final yet, speak the message first
                        {"speak_before_action": True},
                    )

                    # Then send the action to transfer the call
                    await response_callback(
                        "",
                        True,
                        {
                            "action": "transfer_call",
                            "destination": destination,
                        },
                    )
                else:
                    logger.error("Transfer call requested but no destination provided")

            else:
                logger.warning(f"Unknown tool call: {tool_name}")

        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            logger.exception(e)


class LangChainProvider(BaseLLMProvider):
    """LangChain-based provider for all LLM providers except custom."""

    def __init__(self, config: Dict[str, Any], provider_name: str):
        super().__init__(config)
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages not installed. Please install langchain and required provider packages."
            )

        self.provider_name = provider_name
        self.llm = self._initialize_llm()
        self.rag_service = None  # Will be initialized when needed

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the appropriate LangChain LLM based on provider."""

        if self.provider_name == "openai":
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model or "gpt-4o-mini",
                streaming=True,
                **self.custom_config,
            )

        elif self.provider_name == "anthropic":
            return ChatAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model or "claude-3-5-sonnet-latest",
                streaming=True,
                **self.custom_config,
            )

        elif self.provider_name == "gemini":
            return ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model or "gemini-2.0-flash",
                streaming=True,
                **self.custom_config,
            )

        elif self.provider_name == "groq":
            return ChatGroq(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model or "llama-3.3-70b-versatile",
                streaming=True,
                **self.custom_config,
            )

        elif self.provider_name == "xai":
            # xAI uses OpenAI-compatible API
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.x.ai/v1",
                model=self.model or "grok-beta",
                streaming=True,
                **self.custom_config,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

    def _convert_to_langchain_messages(self, messages: list) -> list:
        """Convert message format to LangChain messages."""
        langchain_messages = []

        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        return langchain_messages

    def _initialize_rag_service(self) -> None:
        """Initialize RAG service if not already initialized."""
        if not RAG_AVAILABLE:
            return

        if self.rag_service is None:
            try:
                self.rag_service = RAGService.create_default_instance()
                logger.info("RAG service initialized for LangChain provider")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG service: {e}")

    async def _enhance_messages_with_rag(
        self, messages: list, assistant_id: int
    ) -> list:
        """
        Enhance messages with relevant document context using RAG.

        Args:
            messages: Original messages
            assistant_id: Assistant ID for document filtering

        Returns:
            list: Enhanced messages with document context
        """
        if not RAG_AVAILABLE:
            return messages

        try:
            # Initialize RAG service if needed
            self._initialize_rag_service()

            if not self.rag_service:
                return messages

            # Get the latest user message for context retrieval
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            if not user_messages:
                return messages

            latest_query = user_messages[-1]["content"]

            # Search for relevant documents
            relevant_chunks = await self.rag_service.search_documents(
                query=latest_query,
                assistant_id=assistant_id,
                limit=3,  # Limit to top 3 most relevant chunks
                similarity_threshold=0.3,
            )

            if not relevant_chunks:
                logger.debug(
                    f"No relevant documents found for assistant {assistant_id}"
                )
                return messages

            # Build context from retrieved documents
            context_parts = []
            for chunk in relevant_chunks:
                doc_info = chunk.get("document", {})
                context_parts.append(
                    f"From document '{doc_info.get('name', 'Unknown')}': {chunk['content']}"
                )

            context = "\n\n".join(context_parts)

            # Find the system message and enhance it with context
            enhanced_messages = []
            context_added = False

            for msg in messages:
                if msg["role"] == "system" and not context_added:
                    # Add document context to system prompt
                    enhanced_content = f"""{msg["content"]}

## Available Document Context:
The following information from the knowledge base may be relevant to answer questions:

{context}

Use this context when relevant to provide accurate and detailed responses. If the context doesn't contain relevant information for the user's query, rely on your general knowledge."""

                    enhanced_messages.append(
                        {"role": "system", "content": enhanced_content}
                    )
                    context_added = True
                else:
                    enhanced_messages.append(msg)

            # If no system message exists, add one with context
            if not context_added:
                system_message = {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant. Use the following document context when relevant:

## Available Document Context:
{context}

Use this context when relevant to provide accurate and detailed responses.""",
                }
                enhanced_messages.insert(0, system_message)

            logger.info(
                f"Enhanced messages with {len(relevant_chunks)} document chunks for assistant {assistant_id}"
            )
            return enhanced_messages

        except Exception as e:
            logger.error(f"Error enhancing messages with RAG: {e}")
            return messages

    async def process_transcript(
        self,
        messages: list,
        settings: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Process transcript using LangChain with optional RAG enhancement."""
        try:
            # Get tools from assistant if available
            tools = []
            assistant = settings.get("assistant")
            if assistant:
                tools = get_all_tools(assistant)

            # Check if RAG should be enabled
            enable_rag = False

            if assistant and RAG_AVAILABLE:
                # Check if assistant has documents
                rag_settings = getattr(assistant, "rag_settings", {})
                enable_rag = rag_settings.get(
                    "enabled", True
                )  # Default to enabled if documents exist

            # Enhance messages with RAG if enabled
            if enable_rag and assistant:
                messages = await self._enhance_messages_with_rag(
                    messages, assistant.id
                )

            # Convert messages to LangChain format
            langchain_messages = self._convert_to_langchain_messages(messages)

            # Configure LLM parameters - create a copy of the LLM with specific settings
            llm_kwargs = {}

            # Add provider-specific parameters
            if self.provider_name in ["openai", "xai", "groq"]:
                llm_kwargs.update(
                    {
                        "temperature": settings.get("temperature", 0.7),
                        "max_tokens": settings.get("max_tokens", 150),
                        "top_p": settings.get("top_p", 1.0),
                        "frequency_penalty": settings.get("frequency_penalty", 0.0),
                        "presence_penalty": settings.get("presence_penalty", 0.0),
                    }
                )
            else:
                # For Anthropic, Gemini
                llm_kwargs.update(
                    {
                        "temperature": settings.get("temperature", 0.7),
                        "max_tokens": settings.get("max_tokens", 150),
                        "top_p": settings.get("top_p", 1.0),
                    }
                )

            # Configure the LLM with runtime parameters
            configured_llm = self.llm.bind(**llm_kwargs)

            # Handle tools if available
            if tools:
                # For tool calling, we need to use the bind_tools method
                llm_with_tools = configured_llm.bind_tools(tools)

                # Process with tools - use ainvoke for tool calls (non-streaming)
                try:
                    response = await llm_with_tools.ainvoke(langchain_messages)

                    # Check if there are tool calls
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        for tool_call in response.tool_calls:
                            # Convert LangChain tool call format to our format
                            formatted_tool_call = {
                                "name": tool_call["name"],
                                "id": tool_call.get("id", ""),
                                "arguments": json.dumps(tool_call.get("args", {})),
                            }
                            await self._handle_tool_call(
                                formatted_tool_call, assistant, response_callback
                            )
                            return

                    # If no tool calls, treat as regular response
                    if response.content:
                        await response_callback(
                            response.content, True, {"full_response": response.content}
                        )

                except Exception as tool_error:
                    logger.warning(
                        f"Tool calling failed, falling back to streaming: {tool_error}"
                    )
                    # Fall back to regular streaming if tool calling fails
                    await self._stream_response(
                        configured_llm, langchain_messages, response_callback
                    )
            else:
                # Regular streaming without tools
                await self._stream_response(
                    configured_llm, langchain_messages, response_callback
                )

        except Exception as e:
            logger.error(f"LangChain {self.provider_name} error: {e}")
            # Re-raise the exception so fallback providers can be tried
            raise e

    async def _stream_response(
        self,
        llm: BaseChatModel,
        messages: list,
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """Handle streaming response from LangChain LLM."""
        collected_response = ""

        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    collected_response += chunk.content
                    await response_callback(chunk.content, False, {})

            # Send final response indicator
            if collected_response:
                await response_callback(collected_response, True, {"full_response": collected_response})

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # If streaming fails, try a single non-streaming call
            try:
                response = await llm.ainvoke(messages)
                if response.content:
                    await response_callback(
                        response.content, True, {"full_response": response.content}
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                raise e


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
            assistant = settings.get("assistant")

            # Temporarily store assistant in config for access in _process_custom_llm_response
            if assistant:
                self.config["assistant"] = assistant

            # Get tools from assistant if available
            tools = []
            if assistant:
                tools = get_all_tools(assistant)

            payload = {
                "messages": messages,
                "phoneNumber": {"number": to_number or ""},
                "call": {
                    "phoneCallProviderId": call_sid or "",
                    "customer": {"number": from_number or ""},
                },
            }

            # Add tools if available
            if tools:
                payload["tools"] = tools

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
            logger.exception(e)
            # Re-raise the exception so fallback providers can be tried
            raise e

    async def _process_custom_llm_response(
        self,
        response: httpx.Response,
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """
        Process streaming response from custom LLM endpoint.

        Args:
            response: The streaming response from custom LLM
            response_callback: Callback function to handle streaming responses
        """
        collected_response = ""
        current_tool_call = None

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix

                    # Handle tool calls
                    if data.get("choices") and data["choices"][0].get("delta"):
                        delta = data["choices"][0]["delta"]

                        # Handle regular content
                        if delta.get("content") is not None:
                            if delta["content"]:  # Only process non-null content
                                content = delta["content"]
                                collected_response += content
                                await response_callback(content, False, {})

                        # Check for tool calls
                        if delta.get("tool_calls"):
                            tool_call = delta["tool_calls"][0]

                            # Handle tool call start - first chunk contains the name
                            if tool_call.get("function", {}).get("name"):
                                current_tool_call = {
                                    "name": tool_call["function"]["name"],
                                    "id": tool_call.get("id"),
                                    "arguments": "",
                                }
                                logger.info(
                                    f"Started tool call: {current_tool_call['name']}"
                                )

                            # Accumulate arguments if they exist - second chunk contains arguments
                            if (
                                tool_call.get("function", {}).get("arguments")
                                is not None
                            ):
                                if current_tool_call:
                                    current_tool_call["arguments"] += tool_call[
                                        "function"
                                    ]["arguments"]
                                    logger.info(
                                        f"Accumulated arguments: {current_tool_call['arguments']}"
                                    )

                    # Check for finish_reason to handle end of tool call - third chunk indicates completion
                    if (
                        data["choices"][0].get("finish_reason") == "tool_calls"
                        and current_tool_call
                    ):
                        logger.info(
                            f"Completing tool call: {current_tool_call['name']}"
                        )
                        try:
                            assistant = self.config.get("assistant")
                            await self._handle_tool_call(
                                current_tool_call, assistant, response_callback
                            )
                            return
                        except Exception as e:
                            logger.error(f"Error processing tool call completion: {e}")
                        finally:
                            current_tool_call = None

                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON from response: {line}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing response line: {e}")
                    continue

        # Only send final response if we haven't already sent a tool action
        if collected_response and not current_tool_call:
            await response_callback(
                collected_response,
                True,
                {"full_response": collected_response},
            )


class LLMService:
    """
    Multi-provider LLM service for handling various LLM providers using LangChain.
    """

    PROVIDERS = {
        "openai": "langchain",
        "anthropic": "langchain",
        "gemini": "langchain",
        "xai": "langchain",
        "groq": "langchain",
        "custom": "custom",
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

        # Initialize primary and fallback providers
        self.primary_provider = self._initialize_provider()
        self.fallback_providers = self._initialize_fallback_providers()
        self.current_provider_name = self._get_provider_name()
        self.current_provider_index = -1  # -1 for primary, 0+ for fallbacks

        # Default system prompt
        self.default_system_prompt = """You are a helpful AI assistant for a customer service call center. 
        Your role is to assist customers professionally and efficiently. 
        Keep responses concise, clear, and focused on resolving customer needs."""

        # Initialize conversation history with system prompt
        system_prompt = self._get_system_prompt()
        self.conversation_history = [{"role": "system", "content": system_prompt}]

        if call_sid:
            logger.info(
                f"Started new conversation for call {call_sid} using {self.current_provider_name} "
                f"with {len(self.fallback_providers)} fallback(s) available"
            )

    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate LLM provider based on assistant configuration."""
        if not self.assistant:
            # Fallback to OpenAI with environment variable
            config = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4o-mini",
            }
            logger.info("No assistant provided, using default OpenAI provider")
            return LangChainProvider(config, "openai")

        # Use new provider configuration (prioritize this over legacy settings)
        provider_name = getattr(self.assistant, "llm_provider", "openai")
        provider_config = getattr(self.assistant, "llm_provider_config", {})

        logger.info(
            f"Assistant {self.assistant.name} - Provider: {provider_name}, Config: {provider_config}"
        )

        # Handle custom provider with legacy support
        if provider_name == "custom":
            # Check for legacy custom LLM URL configuration
            if (
                hasattr(self.assistant, "custom_llm_url")
                and self.assistant.custom_llm_url
            ):
                config = {
                    "base_url": self.assistant.custom_llm_url,
                }
                logger.info(
                    f"Using legacy custom LLM URL: {self.assistant.custom_llm_url}"
                )
                return CustomLLMProvider(config)

            # Use provider config for custom
            if provider_config.get("base_url"):
                return CustomLLMProvider(provider_config)

            # Fallback to legacy OpenAI key if no provider config
            if (
                hasattr(self.assistant, "openai_api_key")
                and self.assistant.openai_api_key
            ):
                provider_config = {
                    "api_key": self.assistant.openai_api_key,
                    "model": "gpt-4o-mini",
                }
                logger.info("Using legacy OpenAI API key")
                return LangChainProvider(provider_config, "openai")

        # For all other providers, use LangChain
        if self.PROVIDERS.get(provider_name) == "langchain":
            logger.info(
                f"Initialized LangChain {provider_name} provider with config: {provider_config}"
            )
            return LangChainProvider(provider_config, provider_name)

        # Fallback to OpenAI if provider not found
        logger.warning(f"Unknown provider {provider_name}, falling back to OpenAI")
        fallback_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
        }
        return LangChainProvider(fallback_config, "openai")

    def _get_provider_name(self) -> str:
        """Get the name of the current provider."""
        if not self.assistant:
            return "openai"
        return getattr(self.assistant, "llm_provider", "openai")

    def _get_system_prompt(self) -> str:
        """Get the system prompt from assistant configuration."""
        if not self.assistant:
            return self.default_system_prompt

        llm_settings = getattr(self.assistant, "llm_settings", {})
        return llm_settings.get("system_prompt", self.default_system_prompt)

    def _initialize_fallback_providers(self) -> list:
        """Initialize fallback providers based on assistant configuration."""
        fallback_providers = []
        
        if not self.assistant:
            return fallback_providers
            
        fallback_config = getattr(self.assistant, "llm_fallback_providers", {})
        
        if not fallback_config.get("enabled", False):
            return fallback_providers
            
        fallbacks = fallback_config.get("fallbacks", [])
        
        for fallback in fallbacks:
            if not fallback.get("enabled", False):
                continue
                
            provider_name = fallback.get("provider")
            provider_config = fallback.get("config", {})
            
            if not provider_name or not provider_config:
                continue
                
            try:
                if self.PROVIDERS.get(provider_name) == "langchain":
                    provider = LangChainProvider(provider_config, provider_name)
                elif provider_name == "custom":
                    provider = CustomLLMProvider(provider_config)
                else:
                    logger.warning(f"Unknown fallback provider {provider_name}, skipping")
                    continue
                    
                fallback_providers.append({
                    "name": provider_name,
                    "provider": provider,
                    "config": provider_config
                })
                logger.info(f"Initialized fallback provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize fallback provider {provider_name}: {e}")
                continue
                
        return fallback_providers

    def _get_current_provider(self) -> BaseLLMProvider:
        """Get the currently active provider (primary or fallback)."""
        if self.current_provider_index == -1:
            return self.primary_provider
        elif 0 <= self.current_provider_index < len(self.fallback_providers):
            return self.fallback_providers[self.current_provider_index]["provider"]
        else:
            # Should not happen, but fallback to primary
            logger.error(f"Invalid provider index {self.current_provider_index}, falling back to primary")
            self.current_provider_index = -1
            return self.primary_provider

    def _get_current_provider_name(self) -> str:
        """Get the name of the currently active provider."""
        if self.current_provider_index == -1:
            return self._get_provider_name()
        elif 0 <= self.current_provider_index < len(self.fallback_providers):
            return self.fallback_providers[self.current_provider_index]["name"]
        else:
            return self._get_provider_name()

    async def _try_next_provider(
        self,
        transcript: str,
        is_final: bool,
        metadata: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
        enhanced_settings: Dict[str, Any],
        enhanced_callback: Callable[[str, bool, Dict[str, Any]], None],
        last_error: str
    ) -> bool:
        """
        Try the next fallback provider.
        
        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcript
            metadata: Additional metadata about the transcript
            response_callback: Original callback function
            enhanced_settings: Enhanced settings with call metadata
            enhanced_callback: Enhanced callback with metadata
            last_error: Error from the previous provider
            
        Returns:
            bool: True if a fallback was attempted, False if no more fallbacks
        """
        # Move to next provider
        self.current_provider_index += 1
        
        if self.current_provider_index >= len(self.fallback_providers):
            # No more fallbacks available
            logger.error(
                f"All LLM providers failed for call {self.call_sid}. "
                f"Primary: {self._get_provider_name()}, "
                f"Fallbacks: {[f['name'] for f in self.fallback_providers]}. "
                f"Last error: {last_error}"
            )
            await response_callback(
                "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment.",
                True,
                {"call_sid": self.call_sid, "error": "All LLM providers failed", "last_error": last_error},
            )
            return False
            
        # Try the current fallback provider
        current_provider = self._get_current_provider()
        current_name = self._get_current_provider_name()
        
        logger.warning(
            f"Primary provider failed for call {self.call_sid}, trying fallback {self.current_provider_index + 1}: {current_name}"
        )
        
        try:
            await current_provider.process_transcript(
                self.conversation_history, enhanced_settings, enhanced_callback
            )
            
            # Update current provider name for logging
            self.current_provider_name = current_name
            logger.info(f"Successfully switched to fallback provider {current_name} for call {self.call_sid}")
            return True
            
        except Exception as e:
            logger.error(f"Fallback provider {current_name} failed for call {self.call_sid}: {e}")
            # Recursively try the next fallback
            return await self._try_next_provider(
                transcript, is_final, metadata, response_callback, 
                enhanced_settings, enhanced_callback, str(e)
            )

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
                llm_settings = getattr(self.assistant, "llm_settings", {})

            # Add call metadata to settings for custom LLM providers that need it
            enhanced_settings = llm_settings.copy()
            enhanced_settings.update(
                {
                    "call_sid": self.call_sid,
                    "to_number": self.to_number,
                    "from_number": self.from_number,
                    "assistant": self.assistant,
                }
            )

            # Add call metadata to response callback
            enhanced_callback = self._create_enhanced_callback(response_callback)

            # Try the current provider (primary or previously successful fallback)
            current_provider = self._get_current_provider()
            current_name = self._get_current_provider_name()

            try:
                # Process using the current provider
                await current_provider.process_transcript(
                    self.conversation_history, enhanced_settings, enhanced_callback
                )
                
            except Exception as e:
                logger.error(
                    f"LLM provider {current_name} failed for call {self.call_sid}: {e}"
                )
                
                # Try fallback providers if we haven't exhausted them
                fallback_attempted = await self._try_next_provider(
                    transcript, is_final, metadata, response_callback,
                    enhanced_settings, enhanced_callback, str(e)
                )
                
                # If no fallback was attempted or all failed, the error response 
                # was already sent in _try_next_provider
                if not fallback_attempted:
                    # Remove the user message from history since we couldn't process it
                    if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                        self.conversation_history.pop()

        except Exception as e:
            logger.error(
                f"Critical error processing transcript for call {self.call_sid}: {e}",
                exc_info=True,
            )
            # Remove the user message from history since we couldn't process it
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
                
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,
                {"call_sid": self.call_sid, "error": str(e)},
            )

    def _create_enhanced_callback(
        self, original_callback: Callable[[str, bool, Dict[str, Any]], None]
    ) -> Callable[[str, bool, Dict[str, Any]], None]:
        """Create an enhanced callback that adds call metadata and updates conversation history."""

        async def enhanced_callback(
            content: str, is_final: bool, metadata: Dict[str, Any]
        ) -> None:
            # Add call metadata
            metadata["call_sid"] = self.call_sid

            # Update conversation history if this is the final response
            if is_final and metadata.get("full_response"):
                self.conversation_history.append(
                    {"role": "assistant", "content": metadata["full_response"]}
                )

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
                "pip_install": "langchain-openai",
                "note": "Uses LangChain OpenAI integration",
            },
            "anthropic": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": [
                    "claude-3-5-sonnet-latest",
                    "claude-3-haiku-20240307",
                ],
                "pip_install": "langchain-anthropic",
                "note": "Uses LangChain Anthropic integration",
            },
            "gemini": {
                "required_fields": ["api_key"],
                "optional_fields": ["model"],
                "default_models": ["gemini-2.0-flash", "gemini-1.5-pro"],
                "pip_install": "langchain-google-genai",
                "note": "Uses LangChain Google Generative AI integration",
            },
            "xai": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["grok-beta", "grok-2-1212"],
                "pip_install": "langchain-openai",
                "note": "Uses LangChain OpenAI integration with xAI endpoint",
            },
            "groq": {
                "required_fields": ["api_key"],
                "optional_fields": ["base_url", "model"],
                "default_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                "pip_install": "langchain-groq",
                "note": "Uses LangChain Groq integration",
            },
            "custom": {
                "required_fields": ["base_url"],
                "optional_fields": ["api_key", "model"],
                "default_models": [],
                "pip_install": "httpx",
                "note": "For custom LLM endpoints (unchanged implementation)",
            },
        }
        return requirements.get(provider, {})
