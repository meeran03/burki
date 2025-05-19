"""
This file contains the LLM service for handling OpenAI GPT-4 streaming responses.
"""

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
import os
import logging
from typing import Dict, Any, Optional, Callable

from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for handling OpenAI GPT-4 streaming responses.
    """

    def __init__(self):
        """Initialize the OpenAI client."""
        # Get API key from environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info("LLMService initialized with OpenAI client")

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

    async def process_transcript(
        self,
        call_sid: str,
        transcript: str,
        is_final: bool,
        metadata: Dict[str, Any],
        response_callback: Callable[[str, bool, Dict[str, Any]], None],
    ) -> None:
        """
        Process a transcript and get streaming response from GPT-4.

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

            # Get streaming response from GPT-4
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
