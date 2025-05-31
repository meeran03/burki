#!/usr/bin/env python3
"""
Demonstration script for LLM Provider Fallback functionality.

This script shows how the fallback system works when primary LLM providers fail.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MockAssistant:
    """Mock assistant with fallback configuration for demonstration."""
    id: int = 1
    name: str = "Demo Assistant"
    llm_provider: str = "openai"
    llm_provider_config: dict = None
    llm_fallback_providers: dict = None
    llm_settings: dict = None

    def __post_init__(self):
        if self.llm_provider_config is None:
            self.llm_provider_config = {
                "api_key": "invalid-primary-key",  # This will fail
                "model": "gpt-4o-mini",
                "base_url": None
            }
        
        if self.llm_fallback_providers is None:
            self.llm_fallback_providers = {
                "enabled": True,
                "fallbacks": [
                    {
                        "enabled": True,
                        "provider": "anthropic",
                        "config": {
                            "api_key": "invalid-anthropic-key",  # This will also fail
                            "model": "claude-3-5-sonnet-latest",
                            "base_url": None
                        }
                    },
                    {
                        "enabled": True,
                        "provider": "groq",
                        "config": {
                            "api_key": "valid-groq-key",  # This would work in real scenario
                            "model": "llama-3.3-70b-versatile",
                            "base_url": None
                        }
                    },
                    {
                        "enabled": True,
                        "provider": "openai",
                        "config": {
                            "api_key": "backup-openai-key",  # Final fallback
                            "model": "gpt-3.5-turbo",
                            "base_url": None
                        }
                    }
                ]
            }
        
        if self.llm_settings is None:
            self.llm_settings = {
                "temperature": 0.7,
                "max_tokens": 150,
                "system_prompt": "You are a helpful assistant."
            }


class MockLLMService:
    """Mock LLM service to demonstrate fallback behavior."""
    
    def __init__(self, call_sid: str, assistant: MockAssistant):
        self.call_sid = call_sid
        self.assistant = assistant
        self.primary_provider_name = assistant.llm_provider
        self.current_provider_index = -1  # -1 for primary, 0+ for fallbacks
        self.fallback_providers = self._initialize_fallback_providers()
        logger.info(f"Initialized LLM service for call {call_sid} with primary provider: {self.primary_provider_name}")
        logger.info(f"Available fallbacks: {[f['name'] for f in self.fallback_providers]}")

    def _initialize_fallback_providers(self) -> list:
        """Initialize fallback providers from assistant configuration."""
        fallback_providers = []
        
        if not self.assistant.llm_fallback_providers.get("enabled", False):
            return fallback_providers
            
        fallbacks = self.assistant.llm_fallback_providers.get("fallbacks", [])
        
        for fallback in fallbacks:
            if not fallback.get("enabled", False):
                continue
                
            provider_name = fallback.get("provider")
            if provider_name:
                fallback_providers.append({
                    "name": provider_name,
                    "config": fallback.get("config", {})
                })
                
        return fallback_providers

    def _get_current_provider_name(self) -> str:
        """Get the name of the currently active provider."""
        if self.current_provider_index == -1:
            return self.primary_provider_name
        elif 0 <= self.current_provider_index < len(self.fallback_providers):
            return self.fallback_providers[self.current_provider_index]["name"]
        else:
            return self.primary_provider_name

    async def _simulate_provider_call(self, provider_name: str, config: dict) -> tuple[bool, str]:
        """Simulate a call to an LLM provider."""
        # Simulate network delay
        await asyncio.sleep(0.5)
        
        # Simulate failures based on API key validity
        api_key = config.get("api_key", "")
        
        if "invalid" in api_key:
            if provider_name == "openai":
                raise Exception("OpenAI API Error: Invalid API key")
            elif provider_name == "anthropic":
                raise Exception("Anthropic API Error: Authentication failed")
            else:
                raise Exception(f"{provider_name} API Error: Invalid credentials")
        
        # Simulate success
        model = config.get("model", "unknown")
        response = f"Hello! This response is from {provider_name} using model {model}. How can I help you today?"
        return True, response

    async def process_transcript(self, transcript: str) -> Optional[str]:
        """Process transcript with fallback logic."""
        logger.info(f"Processing transcript: '{transcript}'")
        
        # Try primary provider first
        current_name = self._get_current_provider_name()
        config = self.assistant.llm_provider_config
        
        try:
            logger.info(f"Trying primary provider: {current_name}")
            success, response = await self._simulate_provider_call(current_name, config)
            if success:
                logger.info(f"✅ Primary provider {current_name} succeeded")
                return response
        except Exception as e:
            logger.warning(f"❌ Primary provider {current_name} failed: {e}")
            
            # Try fallback providers
            return await self._try_fallbacks(transcript, str(e))
        
        return None

    async def _try_fallbacks(self, transcript: str, last_error: str) -> Optional[str]:
        """Try fallback providers in sequence."""
        for i, fallback in enumerate(self.fallback_providers):
            self.current_provider_index = i
            provider_name = fallback["name"]
            config = fallback["config"]
            
            try:
                logger.info(f"Trying fallback {i+1}/{len(self.fallback_providers)}: {provider_name}")
                success, response = await self._simulate_provider_call(provider_name, config)
                if success:
                    logger.info(f"✅ Fallback provider {provider_name} succeeded!")
                    return response
            except Exception as e:
                logger.warning(f"❌ Fallback provider {provider_name} failed: {e}")
                last_error = str(e)
                continue
        
        # All providers failed
        logger.error(f"🚨 All LLM providers failed! Last error: {last_error}")
        return "I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment."


async def demonstrate_fallback_system():
    """Demonstrate the fallback system in action."""
    print("🤖 LLM Provider Fallback System Demonstration")
    print("=" * 60)
    
    # Create mock assistant with fallback configuration
    assistant = MockAssistant()
    
    print("\n📋 Assistant Configuration:")
    print(f"Primary Provider: {assistant.llm_provider}")
    print(f"Fallback Enabled: {assistant.llm_fallback_providers['enabled']}")
    print("Fallback Providers:")
    for i, fallback in enumerate(assistant.llm_fallback_providers['fallbacks']):
        if fallback['enabled']:
            provider = fallback['provider']
            model = fallback['config']['model']
            print(f"  {i+1}. {provider} ({model})")
    
    print("\n🔄 Starting demonstration...")
    print("Note: Primary and first fallback are configured to fail for demo purposes\n")
    
    # Create LLM service
    llm_service = MockLLMService("demo-call-123", assistant)
    
    # Test scenarios
    test_transcripts = [
        "Hello, I need help with my account",
        "What services do you offer?",
        "Can you help me place an order?"
    ]
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\n🎤 Test {i}: User says: '{transcript}'")
        print("-" * 40)
        
        response = await llm_service.process_transcript(transcript)
        
        if response:
            print(f"🤖 Assistant responds: {response}")
        else:
            print("🚨 No response received - all providers failed")
        
        print()


async def main():
    """Run the demonstration."""
    try:
        await demonstrate_fallback_system()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 