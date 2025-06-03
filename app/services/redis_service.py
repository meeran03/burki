"""
Redis service for managing SMS conversation persistence.
Provides caching with TTL support and graceful fallback to memory storage.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
import redis.asyncio as redis
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisService:
    """Service for managing Redis connections and operations."""
    
    _instance: Optional['RedisService'] = None
    _redis_client: Optional[redis.Redis] = None
    _connection_error_logged: bool = False
    
    def __new__(cls):
        """Singleton pattern for Redis service."""
        if cls._instance is None:
            cls._instance = super(RedisService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Redis service (only runs once due to singleton)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._max_retries = 3
            self._retry_delay = 1.0
    
    async def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client with lazy connection."""
        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("Successfully connected to Redis")
                self._connection_error_logged = False
            except Exception as e:
                if not self._connection_error_logged:
                    logger.warning(f"Redis connection failed: {e}. Falling back to memory storage.")
                    self._connection_error_logged = True
                self._redis_client = None
        
        return self._redis_client
    
    async def set_sms_conversation(
        self,
        conversation_key: str,
        conversation_data: Dict[str, Any],
        ttl_seconds: int = 86400  # 24 hours default
    ) -> bool:
        """
        Store SMS conversation in Redis with TTL.
        
        Args:
            conversation_key: Unique key for the conversation
            conversation_data: Conversation state data
            ttl_seconds: Time to live in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = await self.get_client()
            if client is None:
                return False
            
            # Serialize conversation data
            serialized_data = json.dumps(conversation_data, default=str)
            
            # Store with TTL
            await client.setex(
                f"sms:conversation:{conversation_key}",
                ttl_seconds,
                serialized_data
            )
            
            logger.debug(f"Stored SMS conversation {conversation_key} in Redis with TTL {ttl_seconds}s")
            return True
            
        except Exception as e:
            logger.error(f"Error storing SMS conversation in Redis: {e}")
            return False
    
    async def get_sms_conversation(
        self,
        conversation_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve SMS conversation from Redis.
        
        Args:
            conversation_key: Unique key for the conversation
            
        Returns:
            Optional[Dict[str, Any]]: Conversation data if found
        """
        try:
            client = await self.get_client()
            if client is None:
                return None
            
            # Get conversation data
            data = await client.get(f"sms:conversation:{conversation_key}")
            
            if data:
                # Deserialize
                conversation_data = json.loads(data)
                logger.debug(f"Retrieved SMS conversation {conversation_key} from Redis")
                return conversation_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving SMS conversation from Redis: {e}")
            return None
    
    async def delete_sms_conversation(
        self,
        conversation_key: str
    ) -> bool:
        """
        Delete SMS conversation from Redis.
        
        Args:
            conversation_key: Unique key for the conversation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = await self.get_client()
            if client is None:
                return False
            
            # Delete the key
            result = await client.delete(f"sms:conversation:{conversation_key}")
            
            if result:
                logger.debug(f"Deleted SMS conversation {conversation_key} from Redis")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting SMS conversation from Redis: {e}")
            return False
    
    async def update_sms_conversation_ttl(
        self,
        conversation_key: str,
        ttl_seconds: int
    ) -> bool:
        """
        Update TTL for an existing SMS conversation.
        
        Args:
            conversation_key: Unique key for the conversation
            ttl_seconds: New TTL in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = await self.get_client()
            if client is None:
                return False
            
            # Update TTL
            result = await client.expire(
                f"sms:conversation:{conversation_key}",
                ttl_seconds
            )
            
            if result:
                logger.debug(f"Updated TTL for SMS conversation {conversation_key} to {ttl_seconds}s")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating SMS conversation TTL in Redis: {e}")
            return False
    
    async def get_all_sms_conversations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all SMS conversations from Redis (for debugging/admin).
        
        Returns:
            Dict[str, Dict[str, Any]]: All conversations keyed by conversation_key
        """
        conversations = {}
        
        try:
            client = await self.get_client()
            if client is None:
                return conversations
            
            # Find all SMS conversation keys
            keys = await client.keys("sms:conversation:*")
            
            if keys:
                # Get all conversations
                for key in keys:
                    data = await client.get(key)
                    if data:
                        # Extract conversation key from Redis key
                        conversation_key = key.replace("sms:conversation:", "")
                        conversations[conversation_key] = json.loads(data)
            
            logger.debug(f"Retrieved {len(conversations)} SMS conversations from Redis")
            
        except Exception as e:
            logger.error(f"Error retrieving all SMS conversations from Redis: {e}")
        
        return conversations
    
    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            logger.info("Closed Redis connection") 