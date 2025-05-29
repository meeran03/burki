# Modular TTS Architecture

This document explains the new modular Text-to-Speech (TTS) architecture that replaces the previous ElevenLabs-specific implementation with a provider-agnostic system.

## Overview

The system has been refactored to support multiple TTS providers through a modular architecture. This makes it easy to:

- Switch between different TTS providers
- Add new TTS providers without changing core logic
- Maintain backward compatibility with existing code
- Configure different providers per assistant/call

## Architecture Components

### 1. Base TTS Interface (`tts_base.py`)

Defines the contract that all TTS providers must implement:

```python
from app.services.tts_base import BaseTTSService, TTSProvider, TTSOptions

class MyTTSProvider(BaseTTSService):
    async def start_session(self, options, audio_callback, metadata):
        # Implementation specific to your provider
        pass
    
    async def process_text(self, text, force_flush=False):
        # Convert text to speech
        pass
    
    async def stop_synthesis(self):
        # Stop current synthesis
        pass
    
    async def end_session(self):
        # Clean up resources
        pass
```

### 2. Provider Implementations

#### ElevenLabs (`tts_elevenlabs.py`)
- Uses ElevenLabs WebSocket API
- Supports multiple voices (Rachel, Domi, Bella, Antoni, Eli, etc.)
- Real-time streaming with buffer management
- Configurable voice settings (stability, similarity boost, style)

#### Deepgram (`tts_deepgram.py`)
- Uses Deepgram Aura WebSocket API
- Supports both Aura and Aura-2 generation models
- Multiple voice options (Asteria, Luna, Stella, Orion, Zeus, etc.)
- Optimized for real-time conversational AI
- 3x faster than ElevenLabs according to benchmarks

### 3. TTS Factory (`tts_factory.py`)

Creates TTS service instances based on provider type:

```python
from app.services.tts_factory import TTSFactory
from app.services.tts_base import TTSProvider

# Create ElevenLabs instance
elevenlabs_tts = TTSFactory.create_tts_service(
    provider=TTSProvider.ELEVENLABS,
    call_sid="call_123",
    voice_id="rachel"
)

# Create Deepgram instance
deepgram_tts = TTSFactory.create_tts_service(
    provider=TTSProvider.DEEPGRAM,
    call_sid="call_123",
    voice_id="asteria"
)

# Create from assistant settings
tts_service = TTSFactory.create_from_settings(
    call_sid="call_123",
    assistant=assistant_obj
)
```

### 4. Main TTS Service (`tts_service.py`)

Provides a unified interface that wraps provider-specific implementations:

```python
from app.services.tts_service import TTSService
from app.services.tts_base import TTSProvider

# Create with specific provider
tts = TTSService(
    call_sid="call_123",
    provider=TTSProvider.DEEPGRAM,
    voice_id="asteria"
)

# Or create from assistant settings
tts = TTSService.create_from_assistant(
    call_sid="call_123",
    assistant=assistant_obj
)
```

## Available Providers

### ElevenLabs
- **Provider**: `TTSProvider.ELEVENLABS`
- **API Key**: `ELEVENLABS_API_KEY`
- **Voices**: rachel, domi, bella, antoni, eli, josh, arnold, adam, sam
- **Models**: turbo, enhanced, multilingual
- **Features**: High-quality voices, extensive customization options

### Deepgram
- **Provider**: `TTSProvider.DEEPGRAM`
- **API Key**: `DEEPGRAM_API_KEY`
- **Voices**: asteria, luna, stella, athena, hera, orion, arcas, perseus, angus, orpheus, helios, zeus, thalia, aurora
- **Models**: aura, aura-2
- **Features**: Ultra-low latency, optimized for real-time AI, 3x faster than ElevenLabs

## Adding New Providers

To add a new TTS provider:

1. **Create Provider Implementation**:
```python
# app/services/tts_myprovider.py
from .tts_base import BaseTTSService, TTSProvider

class MyProviderTTSService(BaseTTSService):
    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.MYPROVIDER
    
    # Implement required methods...
```

2. **Add to Provider Enum**:
```python
# app/services/tts_base.py
class TTSProvider(Enum):
    ELEVENLABS = "elevenlabs"
    DEEPGRAM = "deepgram"
    MYPROVIDER = "myprovider"  # Add this
```

3. **Register in Factory**:
```python
# app/services/tts_factory.py
from .tts_myprovider import MyProviderTTSService

class TTSFactory:
    _providers = {
        TTSProvider.ELEVENLABS: ElevenLabsTTSService,
        TTSProvider.DEEPGRAM: DeepgramTTSService,
        TTSProvider.MYPROVIDER: MyProviderTTSService,  # Add this
    }
```

## Configuration

### Environment Variables
```bash
# ElevenLabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key
```

### Assistant Settings
```python
# In your assistant configuration
assistant.tts_settings = {
    "provider": "deepgram",  # or "elevenlabs"
    "voice_id": "asteria",   # Provider-specific voice
    "model_id": "aura-2",    # Provider-specific model
    # Other provider-specific settings...
}
```

### Web Form Integration

The assistant creation form now includes:
- **TTS Provider Selection**: Choose between ElevenLabs and Deepgram
- **Dynamic Voice Options**: Voice list updates based on selected provider
- **Dynamic Model Options**: Model list updates based on selected provider
- **Provider-Specific Help**: Context-aware tooltips and recommendations

### Database Schema

The `Assistant` model's `tts_settings` field now includes:
```json
{
    "provider": "elevenlabs|deepgram",
    "voice_id": "provider_specific_voice",
    "model_id": "provider_specific_model",
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": true,
    "latency": 1,
    "provider_config": {}
}
```

## Usage Examples

### Basic Usage
```python
from app.services import TTSService, TTSProvider

# Create TTS service
tts = TTSService(
    call_sid="call_123",
    provider=TTSProvider.DEEPGRAM,
    voice_id="asteria"
)

# Start session
await tts.start_session(audio_callback=my_audio_handler)

# Process text
await tts.process_text("Hello, this is a test message.")

# End session
await tts.end_session()
```

### Switching Providers
```python
# Easy to switch providers
if use_deepgram:
    provider = TTSProvider.DEEPGRAM
    voice = "asteria"
else:
    provider = TTSProvider.ELEVENLABS
    voice = "rachel"

tts = TTSService(
    call_sid="call_123",
    provider=provider,
    voice_id=voice
)
```

## Performance Comparison

Based on Deepgram's benchmarks:

| Provider | Latency | Quality | Cost | Best For |
|----------|---------|---------|------|----------|
| Deepgram | Ultra-low (3x faster) | High | Cost-effective | Real-time AI, conversational bots |
| ElevenLabs | Low | Premium | Higher | High-quality voice synthesis, content creation |

## Migration Guide

### From Old ElevenLabs Implementation

The new system is backward compatible. Existing code will continue to work:

```python
# Old code (still works)
from app.services.tts_service import TTSService
tts = TTSService(call_sid="call_123", voice_id="rachel")

# New code (recommended)
from app.services import TTSService, TTSProvider
tts = TTSService(
    call_sid="call_123", 
    provider=TTSProvider.ELEVENLABS,
    voice_id="rachel"
)
```

### Adding Deepgram Support

To use Deepgram instead of ElevenLabs:

```python
# Simply change the provider
tts = TTSService(
    call_sid="call_123",
    provider=TTSProvider.DEEPGRAM,  # Changed from ELEVENLABS
    voice_id="asteria"              # Changed from "rachel"
)
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are set
2. **Invalid Voice IDs**: Check available voices for each provider
3. **Connection Issues**: Verify network connectivity and API key validity
4. **Audio Format**: Ensure output format matches your audio pipeline requirements

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("app.services.tts_deepgram").setLevel(logging.DEBUG)
logging.getLogger("app.services.tts_elevenlabs").setLevel(logging.DEBUG)
```

## Future Enhancements

Planned provider additions:
- OpenAI TTS
- Azure Cognitive Services
- AWS Polly
- Google Cloud Text-to-Speech

The modular architecture makes it easy to add any of these providers without changing existing code. 