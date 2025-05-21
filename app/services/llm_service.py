"""
This file contains the LLM service for handling OpenAI GPT-4 streaming responses.
"""

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
import os
import json
import logging
from typing import Dict, Any, Optional, Callable
import httpx

from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for handling OpenAI GPT-4 streaming responses and custom LLM endpoints.
    """

    def __init__(self, custom_llm_url: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            custom_llm_url: Optional URL for custom LLM endpoint. If provided, this will be used instead of OpenAI.
        """
        self.custom_llm_url = custom_llm_url
        
        # Initialize OpenAI client only if custom_llm_url is not provided
        if not custom_llm_url:
            # Get API key from environment variable
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            # Initialize OpenAI client
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("LLMService initialized with OpenAI client")
        else:
            logger.info(f"LLMService initialized with custom LLM URL: {custom_llm_url}")

        # Default system prompt
        self.default_system_prompt = """You are a helpful AI assistant for a customer service call center. 
        Your role is to assist customers professionally and efficiently. 
        Keep responses concise, clear, and focused on resolving customer needs."""

        # Track active conversations
        self.active_conversations: Dict[str, list] = {}

    async def start_conversation(
        self, call_sid: str, system_prompt: Optional[str] = None
    ) -> None:
        """
        Start a new conversation for a call.

        Args:
            call_sid: The Twilio call SID
            system_prompt: Optional custom system prompt
        """
        # Initialize conversation history
        self.active_conversations[call_sid] = [
            {"role": "system", "content": system_prompt or self.default_system_prompt}
        ]
        logger.info(f"Started new conversation for call {call_sid}")

    async def _process_custom_llm_response(
        self,
        call_sid: str,
        response: httpx.Response,
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """
        Process streaming response from custom LLM endpoint.

        Args:
            call_sid: The Twilio call SID
            response: The streaming response from custom LLM
            response_callback: Callback function to handle streaming responses
        """
        collected_response = ""
        current_tool_call = None

        async for line in response.aiter_lines():
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    
                    # Handle tool calls
                    if data.get('choices') and data['choices'][0].get('delta'):
                        delta = data['choices'][0]['delta']
                        
                        # Handle regular content
                        if delta.get('content') is not None:
                            if delta['content']:  # Only process non-null content
                                content = delta['content']
                                collected_response += content
                                await response_callback(content, False, {"call_sid": call_sid})
                        
                        # Check for tool calls
                        if delta.get('tool_calls'):
                            tool_call = delta['tool_calls'][0]
                            
                            # Handle tool call start - first chunk contains the name
                            if tool_call.get('function', {}).get('name'):
                                current_tool_call = {
                                    'name': tool_call['function']['name'],
                                    'id': tool_call.get('id'),
                                    'arguments': ''
                                }
                                logger.info(f"Started tool call: {current_tool_call['name']}")
                            
                            # Accumulate arguments if they exist - second chunk contains arguments
                            if tool_call.get('function', {}).get('arguments') is not None:
                                if current_tool_call:
                                    current_tool_call['arguments'] += tool_call['function']['arguments']
                                    logger.info(f"Accumulated arguments: {current_tool_call['arguments']}")

                    # Check for finish_reason to handle end of tool call - third chunk indicates completion
                    if data['choices'][0].get('finish_reason') == 'tool_calls' and current_tool_call:
                        logger.info(f"Completing tool call: {current_tool_call['name']}")
                        try:
                            # Parse arguments if they exist, otherwise use empty dict
                            args = {}
                            if current_tool_call['arguments']:
                                try:
                                    args = json.loads(current_tool_call['arguments'])
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse arguments: {current_tool_call['arguments']}, using empty dict")
                            
                            if current_tool_call['name'] == "endCall":
                                logger.info("Processing endCall tool call")
                                # Send special metadata to indicate call should end
                                await response_callback(
                                    "",
                                    True,
                                    {
                                        "call_sid": call_sid,
                                        "action": "end_call"
                                    }
                                )
                                return
                            elif current_tool_call['name'] == "transferCall":
                                logger.info("Processing transferCall tool call")
                                destination = args.get("destination")
                                if destination:
                                    # Send special metadata for call transfer
                                    await response_callback(
                                        "",
                                        True,
                                        {
                                            "call_sid": call_sid,
                                            "action": "transfer_call",
                                            "destination": destination
                                        }
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
            self.active_conversations[call_sid].append(
                {"role": "assistant", "content": collected_response}
            )
            await response_callback(
                "",
                True,
                {"call_sid": call_sid, "full_response": collected_response},
            )

    async def process_transcript(
        self,
        call_sid: str,
        transcript: str,
        is_final: bool,
        metadata: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """
        Process a transcript and get streaming response from LLM.

        Args:
            call_sid: The Twilio call SID
            transcript: The transcribed text
            is_final: Whether this is a final transcript
            metadata: Additional metadata about the transcript
            response_callback: Callback function to handle streaming responses
        """
        if call_sid not in self.active_conversations:
            await self.start_conversation(call_sid)

        # Only process final transcripts to avoid too many API calls
        if not is_final:
            return

        try:
            # Add user message to conversation history
            self.active_conversations[call_sid].append(
                {"role": "user", "content": transcript}
            )

            if self.custom_llm_url:
                # Prepare request payload for custom LLM
                payload = {
                    "messages": self.active_conversations[call_sid],
                    "phoneNumber": {
                        "number": metadata.get("to_number", "")
                    },
                    "call": {
                        "phoneCallProviderId": call_sid,
                        "customer": {
                            "number": metadata.get("from_number", "")
                        }
                    }
                }

                # Make streaming request to custom LLM
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        'POST',
                        self.custom_llm_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=30.0
                    ) as response:
                        response.raise_for_status()
                        await self._process_custom_llm_response(call_sid, response, response_callback)
            else:
                # Use OpenAI client
                stream = await self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Using GPT-4 for best performance
                    messages=self.active_conversations[call_sid],
                    stream=True,
                    temperature=0.7,  # Balanced between creativity and consistency
                    max_tokens=150,  # Keep responses concise
                )

                # Process the streaming response
                collected_response = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_response += content

                        # Send chunk to callback
                        await response_callback(
                            content, False, {"call_sid": call_sid}  # Not final
                        )

                # Add assistant's response to conversation history
                if collected_response:
                    self.active_conversations[call_sid].append(
                        {"role": "assistant", "content": collected_response}
                    )

                    # Send final signal
                    await response_callback(
                        "",  # Empty content for final signal
                        True,  # Final
                        {"call_sid": call_sid, "full_response": collected_response},
                    )

        except Exception as e:
            logger.error(
                f"Error processing transcript for call {call_sid}: {e}", exc_info=True
            )
            # Send error notification through callback
            await response_callback(
                "I apologize, but I'm having trouble processing that right now.",
                True,  # Final
                {"call_sid": call_sid, "error": str(e)},
            )

    async def end_conversation(self, call_sid: str) -> None:
        """
        End a conversation and clean up resources.

        Args:
            call_sid: The Twilio call SID
        """
        if call_sid in self.active_conversations:
            del self.active_conversations[call_sid]
            logger.info(f"Ended conversation for call {call_sid}")

    def get_conversation_history(self, call_sid: str) -> list:
        """
        Get the conversation history for a call.

        Args:
            call_sid: The Twilio call SID

        Returns:
            list: The conversation history
        """
        return self.active_conversations.get(call_sid, [])
