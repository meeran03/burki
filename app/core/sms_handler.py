"""
SMS Handler for managing SMS conversations with AI assistants.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from app.services.llm_service import LLMService
from app.services.conversation_service import ConversationService
from app.services.webhook_service import WebhookService
from app.services.redis_service import RedisService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SMSConversationState:
    """Represents the state of an SMS conversation."""
    
    message_sid: str
    assistant_id: int
    to_number: str
    from_number: str
    start_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_service: Optional[Any] = None
    assistant: Optional[Any] = None
    conversation_id: Optional[int] = None  # Database conversation ID
    conversation_history: list = field(default_factory=list)  # Store LLM conversation history
    last_activity_time: datetime = field(default_factory=datetime.now)
    
    def to_redis_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "message_sid": self.message_sid,
            "assistant_id": self.assistant_id,
            "to_number": self.to_number,
            "from_number": self.from_number,
            "start_time": self.start_time.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
            "conversation_id": self.conversation_id,
            "conversation_history": self.conversation_history,
            "last_activity_time": self.last_activity_time.isoformat(),
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, Any], assistant: Any = None) -> 'SMSConversationState':
        """Create instance from Redis data."""
        state = cls(
            message_sid=data["message_sid"],
            assistant_id=data["assistant_id"],
            to_number=data["to_number"],
            from_number=data["from_number"],
            start_time=datetime.fromisoformat(data["start_time"]),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
            conversation_id=data.get("conversation_id"),
            conversation_history=data.get("conversation_history", []),
            last_activity_time=datetime.fromisoformat(data.get("last_activity_time", data["start_time"])),
            assistant=assistant,
        )
        return state


class SMSHandler:
    """
    Handles SMS conversations with AI assistants.
    Much simpler than CallHandler as there's no streaming audio or real-time requirements.
    """
    
    def __init__(self):
        """Initialize the SMS handler."""
        self.active_conversations: Dict[str, SMSConversationState] = {}
        self.redis_service = RedisService()
        self._cleanup_task = None
        
    async def start(self):
        """Start the SMS handler and background tasks."""
        # Start auto-cleanup task
        self._cleanup_task = asyncio.create_task(self._auto_cleanup_task())
        logger.info("Started SMS handler with auto-cleanup")
        
    async def stop(self):
        """Stop the SMS handler and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        await self.redis_service.close()
        logger.info("Stopped SMS handler")
    
    async def _auto_cleanup_task(self):
        """Background task to clean up stale conversations from memory cache."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Only clean up memory cache, Redis handles its own TTL
                current_time = datetime.now()
                stale_keys = []
                
                # Remove conversations from memory after 1 hour of inactivity
                # This is just for memory management, Redis remains the source of truth
                for key, state in list(self.active_conversations.items()):
                    age_hours = (current_time - state.last_activity_time).total_seconds() / 3600
                    if age_hours > 1:  # Keep in memory for only 1 hour
                        stale_keys.append(key)
                
                # Remove from memory only
                for key in stale_keys:
                    del self.active_conversations[key]
                
                if stale_keys:
                    logger.info(f"Removed {len(stale_keys)} inactive conversations from memory cache")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SMS memory cleanup task: {e}")
    
    async def _save_to_redis(self, conversation_key: str, state: SMSConversationState, ttl_hours: int = 24) -> bool:
        """Save conversation state to Redis."""
        try:
            # Get TTL from assistant settings if available
            if state.assistant and hasattr(state.assistant, 'sms_settings'):
                sms_settings = state.assistant.sms_settings or {}
                redis_settings = sms_settings.get('redis_persistence', {})
                if redis_settings.get('enabled', True):
                    ttl_hours = redis_settings.get('ttl_hours', ttl_hours)
                else:
                    return False  # Redis disabled for this assistant
            
            ttl_seconds = ttl_hours * 3600
            return await self.redis_service.set_sms_conversation(
                conversation_key,
                state.to_redis_dict(),
                ttl_seconds
            )
        except Exception as e:
            logger.error(f"Error saving conversation to Redis: {e}")
            return False
    
    async def _load_from_redis(self, conversation_key: str, assistant: Any = None) -> Optional[SMSConversationState]:
        """Load conversation state from Redis."""
        try:
            data = await self.redis_service.get_sms_conversation(conversation_key)
            if data:
                state = SMSConversationState.from_redis_dict(data, assistant)
                
                # Recreate LLM service with conversation history
                if assistant:
                    state.llm_service = LLMService(
                        channel_sid=state.message_sid,
                        to_number=state.to_number,
                        from_number=state.from_number,
                        assistant=assistant,
                    )
                    
                    # Restore conversation history
                    if state.conversation_history:
                        state.llm_service.conversation_history = state.conversation_history
                
                return state
        except Exception as e:
            logger.error(f"Error loading conversation from Redis: {e}")
        
        return None
    
    async def _update_redis_ttl(self, conversation_key: str, state: SMSConversationState):
        """Update Redis TTL when conversation is active."""
        try:
            if state.assistant and hasattr(state.assistant, 'sms_settings'):
                sms_settings = state.assistant.sms_settings or {}
                ttl_hours = sms_settings.get('conversation_ttl_hours', 24)
                ttl_seconds = ttl_hours * 3600
                await self.redis_service.update_sms_conversation_ttl(conversation_key, ttl_seconds)
        except Exception as e:
            logger.error(f"Error updating Redis TTL: {e}")
    
    async def process_incoming_sms(
        self,
        message_sid: str,
        from_number: str,
        to_number: str,
        message_body: str,
        assistant: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Process an incoming SMS message and generate a response.
        
        Args:
            message_sid: Twilio Message SID
            from_number: Customer's phone number
            to_number: Assistant's phone number
            message_body: The SMS message content
            assistant: The assistant configuration
            metadata: Additional metadata from Twilio
            
        Returns:
            Optional[str]: The response message to send back
        """
        try:
            # Create or retrieve conversation state
            conversation_key = f"{from_number}:{to_number}"
            existing_conversation = None
            agenda = None
            
            # Try to load from Redis first
            if conversation_key not in self.active_conversations:
                state = await self._load_from_redis(conversation_key, assistant)
                if state:
                    self.active_conversations[conversation_key] = state
                    logger.info(f"Loaded SMS conversation from Redis: {conversation_key}")
            
            # Check if this is a reply to an outbound SMS with agenda
            # Look for existing conversation by phone numbers (reverse lookup for outbound)
            if conversation_key not in self.active_conversations:
                # Try to find existing conversation in database
                # For outbound SMS, customer number is the "to" number
                existing_conversation = await ConversationService.get_conversation_by_phone_numbers(
                    customer_phone_number=from_number,
                    assistant_phone_number=to_number,
                    conversation_type="sms",
                    status_not_in=["completed", "failed"]
                )
                
                if existing_conversation and existing_conversation.conversation_metadata:
                    metadata_obj = existing_conversation.conversation_metadata
                    if isinstance(metadata_obj, dict) and metadata_obj.get("agenda"):
                        agenda = metadata_obj["agenda"]
                        logger.info(f"Found existing conversation with agenda: {agenda[:100]}...")
            
            if conversation_key not in self.active_conversations:
                # Create new conversation state
                self.active_conversations[conversation_key] = SMSConversationState(
                    message_sid=message_sid,
                    assistant_id=assistant.id,
                    to_number=to_number,
                    from_number=from_number,
                    metadata=metadata or {},
                    assistant=assistant,
                )
                
                # Create LLM service for this conversation
                self.active_conversations[conversation_key].llm_service = LLMService(
                    channel_sid=message_sid,  # Using message_sid as a unique identifier
                    to_number=to_number,
                    from_number=from_number,
                    assistant=assistant,
                )
                
                # If we found an existing conversation with agenda, inject it into the LLM context
                if existing_conversation and agenda:
                    # Use the existing conversation ID
                    self.active_conversations[conversation_key].conversation_id = existing_conversation.id
                    
                    # Inject agenda into LLM conversation history
                    llm_service = self.active_conversations[conversation_key].llm_service
                    if llm_service and len(llm_service.conversation_history) > 0:
                        # Add agenda as a system message after the initial system prompt
                        agenda_message = {
                            "role": "system",
                            "content": f"SMS CONVERSATION AGENDA: {agenda}. This is an ongoing SMS conversation that started with an outbound message. Continue the conversation based on this agenda and the customer's responses."
                        }
                        llm_service.conversation_history.insert(1, agenda_message)
                        logger.info(f"Injected agenda into SMS conversation history")
                    
                    # Load existing conversation history if available
                    existing_messages = await ConversationService.get_chat_messages_for_conversation(existing_conversation.id)
                    if existing_messages:
                        for msg in existing_messages:
                            if msg.role in ["user", "assistant"] and msg.content:
                                llm_service.conversation_history.append({
                                    "role": msg.role,
                                    "content": msg.content
                                })
                        logger.info(f"Loaded {len(existing_messages)} existing messages into conversation history")
                else:
                    # Create new conversation record in database
                    conversation = await ConversationService.create_conversation(
                        assistant_id=assistant.id,
                        channel_sid=message_sid,
                        conversation_type="sms",
                        to_phone_number=to_number,
                        customer_phone_number=from_number,
                        metadata=metadata,
                    )
                    
                    if conversation:
                        self.active_conversations[conversation_key].conversation_id = conversation.id
                        logger.info(f"Created SMS conversation record {conversation.id} for {conversation_key}")
                
                # Send initial webhook
                if assistant.webhook_url and self.active_conversations[conversation_key].conversation_id:
                    # Get the conversation object (either existing or newly created)
                    conv_id = self.active_conversations[conversation_key].conversation_id
                    conv = existing_conversation or await ConversationService.get_conversation_by_id(conv_id)
                    if conv:
                        await WebhookService.send_sms_webhook(
                            assistant=assistant,
                            conversation_id=conv.id,
                            webhook_type="sms-received",
                            from_number=from_number,
                            to_number=to_number,
                            message_body=message_body,
                            direction="inbound",
                        )
            
            state = self.active_conversations[conversation_key]
            
            # Store user message in database
            if state.conversation_id:
                # Get current message count to determine index
                existing_messages = await ConversationService.get_chat_messages_for_conversation(state.conversation_id)
                message_index = len(existing_messages)
                
                await ConversationService.create_chat_message(
                    conversation_id=state.conversation_id,
                    role="user",
                    content=message_body,
                    message_index=message_index,
                )
                
                # Note: No need to create transcript for SMS - chat messages are sufficient
                # SMS conversations are stored as chat messages in the database
            
            # Process message with LLM
            response_text = None
            
            async def response_callback(content: str, is_final: bool, metadata: Dict[str, Any]):
                nonlocal response_text
                if is_final and metadata.get("full_response"):
                    response_text = metadata["full_response"]
            
            # Process the message
            await state.llm_service.process_transcript(
                transcript=message_body,
                is_final=True,
                metadata={"sms": True},
                response_callback=response_callback,
            )
            
            # Store assistant response in database
            if response_text and state.conversation_id:
                # Get current message count to determine index
                existing_messages = await ConversationService.get_chat_messages_for_conversation(state.conversation_id)
                message_index = len(existing_messages)
                
                await ConversationService.create_chat_message(
                    conversation_id=state.conversation_id,
                    role="assistant",
                    content=response_text,
                    message_index=message_index,
                )
                
                # Note: No need to create transcript for SMS - chat messages are sufficient
                # SMS conversations are stored as chat messages in the database
                
                # Send response webhook
                if assistant.webhook_url:
                    await WebhookService.send_sms_webhook(
                        assistant=assistant,
                        conversation_id=state.conversation_id,
                        webhook_type="sms-sent",
                        from_number=to_number,
                        to_number=from_number,
                        message_body=response_text,
                        direction="outbound",
                    )
            
            # Update usage for billing
            if state.conversation_id:
                await ConversationService.update_conversation_status(
                    channel_sid=message_sid,
                    status="delivered",
                )
                
                # Record SMS usage for billing
                conversation = await ConversationService.get_conversation_by_id(state.conversation_id)
                if conversation:
                    from app.services.billing_service import BillingService
                    await BillingService.record_sms_usage(conversation.id)
            
            # Update activity time
            state.last_activity_time = datetime.now()
            
            # Save conversation history to state
            if state.llm_service:
                state.conversation_history = state.llm_service.get_conversation_history()
            
            # Save to Redis after processing
            await self._save_to_redis(conversation_key, state)
            
            # Update Redis TTL to extend expiration
            await self._update_redis_ttl(conversation_key, state)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing SMS {message_sid}: {e}", exc_info=True)
            
            # Try to send error webhook
            if assistant and assistant.webhook_url and conversation_key in self.active_conversations:
                state = self.active_conversations[conversation_key]
                if state.conversation_id:
                    try:
                        await WebhookService.send_sms_webhook(
                            assistant=assistant,
                            conversation_id=state.conversation_id,
                            webhook_type="sms-error",
                            from_number=from_number,
                            to_number=to_number,
                            message_body="",
                            direction="incoming",
                            metadata={"error_message": str(e)},
                        )
                    except Exception as webhook_error:
                        logger.error(f"Error sending error webhook: {webhook_error}")
            
            # Return a generic error message
            return "I apologize, but I'm having trouble processing your message right now. Please try again later."
    
    def get_conversation_state(self, from_number: str, to_number: str) -> Optional[SMSConversationState]:
        """Get the current state of an SMS conversation."""
        conversation_key = f"{from_number}:{to_number}"
        return self.active_conversations.get(conversation_key)
    
    async def end_conversation(self, from_number: str, to_number: str) -> None:
        """
        End an SMS conversation and clean up resources.
        
        Args:
            from_number: Customer's phone number
            to_number: Assistant's phone number
        """
        conversation_key = f"{from_number}:{to_number}"
        
        if conversation_key in self.active_conversations:
            state = self.active_conversations[conversation_key]
            
            # Update conversation status in database
            if state.conversation_id:
                await ConversationService.update_conversation_status(
                    channel_sid=state.message_sid,
                    status="completed",
                )
            
            # Note: No need to delete from Redis - let TTL handle it
            # This prevents accidentally removing active conversations
            
            # Clean up from memory only
            del self.active_conversations[conversation_key]
            logger.info(f"Ended SMS conversation {conversation_key}")
    
    async def cleanup_stale_conversations(self, max_age_hours: Optional[int] = None) -> int:
        """
        Mark conversations as completed in the database after inactivity.
        Note: This doesn't remove from Redis (TTL handles that) or memory (auto-cleanup handles that).
        
        Args:
            max_age_hours: Maximum age of conversations in hours (uses assistant settings if not provided)
            
        Returns:
            int: Number of conversations marked as completed
        """
        current_time = datetime.now()
        completed_count = 0
        
        # Check all conversations in Redis (not just memory)
        all_conversations = await self.redis_service.get_all_sms_conversations()
        
        for conversation_key, conv_data in all_conversations.items():
            try:
                # Parse last activity time
                last_activity = datetime.fromisoformat(conv_data.get('last_activity_time', conv_data['start_time']))
                
                # Get max age from data or use default
                if max_age_hours is None:
                    max_age = 72  # Default 72 hours
                else:
                    max_age = max_age_hours
                
                # Check if conversation should be marked as completed
                age_hours = (current_time - last_activity).total_seconds() / 3600
                if age_hours > max_age and conv_data.get('is_active', True):
                    # Just update database status, don't remove from Redis
                    if conv_data.get('conversation_id'):
                        await ConversationService.update_conversation_status(
                            channel_sid=conv_data['message_sid'],
                            status="completed",
                        )
                        completed_count += 1
                        
            except Exception as e:
                logger.error(f"Error processing conversation {conversation_key}: {e}")
        
        if completed_count > 0:
            logger.info(f"Marked {completed_count} SMS conversations as completed")
        
        return completed_count 