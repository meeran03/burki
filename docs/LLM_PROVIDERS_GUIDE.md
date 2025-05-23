# Multi-Provider LLM Service Guide

This guide explains how to use the new multi-provider LLM service that supports OpenAI, Anthropic Claude, Google Gemini, xAI Grok, Groq, and custom LLM endpoints.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Database Migration](#database-migration)
4. [Supported Providers](#supported-providers)
5. [Configuration Examples](#configuration-examples)
6. [Migration from Legacy System](#migration-from-legacy-system)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)

## Overview

The new LLM service provides a unified interface for multiple LLM providers. Each assistant can be configured to use a different provider with specific settings, allowing you to:

- Mix and match different providers for different use cases
- Easily switch between providers without code changes
- Maintain backward compatibility with existing configurations
- Take advantage of provider-specific features

## Installation

### 1. Install Required Dependencies

```bash
# Install all LLM provider dependencies
pip install -r requirements-llm.txt

# Or install only what you need:
pip install openai>=1.0.0           # For OpenAI, Gemini, xAI
pip install anthropic>=0.34.0       # For Anthropic Claude
pip install groq>=0.25.0           # For Groq
```

### 2. Set Up Environment Variables

Create a `.env` file with your API keys:

```bash
# OpenAI (also used for Gemini and xAI with custom base URLs)
OPENAI_API_KEY=your_openai_api_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# xAI Grok
XAI_API_KEY=your_xai_api_key

# Groq
GROQ_API_KEY=your_groq_api_key
```

## Database Migration

Run the migration to add multi-provider support:

```bash
# Apply the migration
alembic upgrade head
```

## Supported Providers

### 1. OpenAI
```python
provider_config = {
    "api_key": "your_openai_api_key",
    "model": "gpt-4o-mini",  # or gpt-4o, gpt-3.5-turbo
    "base_url": None,  # Optional custom base URL
}
```

### 2. Anthropic Claude
```python
provider_config = {
    "api_key": "your_anthropic_api_key",
    "model": "claude-3-5-sonnet-latest",  # or claude-3-haiku-20240307
    "base_url": None,  # Optional custom base URL
}
```

### 3. Google Gemini
```python
provider_config = {
    "api_key": "your_gemini_api_key",
    "model": "gemini-2.0-flash",  # or gemini-1.5-pro
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
```

### 4. xAI Grok
```python
provider_config = {
    "api_key": "your_xai_api_key",
    "model": "grok-beta",  # or grok-2-1212
    "base_url": "https://api.x.ai/v1",
}
```

### 5. Groq
```python
provider_config = {
    "api_key": "your_groq_api_key",
    "model": "llama-3.3-70b-versatile",  # or llama-3.1-8b-instant
    "base_url": None,  # Optional custom base URL
}
```

### 6. Custom Endpoints
```python
provider_config = {
    "base_url": "https://your-custom-llm-endpoint.com/v1",
    "api_key": "optional_api_key",
    "model": "your_model_name",
}
```

## Configuration Examples

### Creating a New Assistant with Anthropic Claude

```python
from app.db.models import Assistant

assistant = Assistant(
    name="Claude Assistant",
    phone_number="+1234567890",
    llm_provider="anthropic",
    llm_provider_config={
        "api_key": "your_anthropic_api_key",
        "model": "claude-3-5-sonnet-latest",
        "base_url": None,
        "custom_config": {}
    },
    llm_settings={
        "temperature": 0.7,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful customer service assistant.",
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop_sequences": []
    }
)
```

### Using Google Gemini

```python
assistant = Assistant(
    name="Gemini Assistant",
    phone_number="+1234567891",
    llm_provider="gemini",
    llm_provider_config={
        "api_key": "your_gemini_api_key",
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "custom_config": {}
    },
    llm_settings={
        "temperature": 0.5,
        "max_tokens": 800,
        "system_prompt": "You are a creative writing assistant.",
        "top_p": 0.9,
        "stop_sequences": ["END", "STOP"]
    }
)
```

### Using Groq for Fast Inference

```python
assistant = Assistant(
    name="Groq Assistant",
    phone_number="+1234567892",
    llm_provider="groq",
    llm_provider_config={
        "api_key": "your_groq_api_key",
        "model": "llama-3.3-70b-versatile",
        "base_url": None,
        "custom_config": {}
    },
    llm_settings={
        "temperature": 0.3,
        "max_tokens": 500,
        "system_prompt": "You are a fast and efficient assistant.",
        "top_p": 1.0,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
)
```

## Migration from Legacy System

### Automatic Migration

The database migration will automatically convert existing assistants:

1. **OpenAI assistants**: Migrate `openai_api_key` to new format
2. **Custom LLM assistants**: Migrate `custom_llm_url` to custom provider
3. **Settings**: Move model from `llm_settings` to `llm_provider_config`

### Manual Migration

If you need to manually update an assistant:

```python
# Before (Legacy)
assistant.openai_api_key = "your_api_key"
assistant.llm_settings = {
    "model": "gpt-4o-mini",
    "temperature": 0.7
}

# After (New Format)
assistant.llm_provider = "openai"
assistant.llm_provider_config = {
    "api_key": "your_api_key",
    "model": "gpt-4o-mini",
    "base_url": None,
    "custom_config": {}
}
assistant.llm_settings = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "system_prompt": "You are a helpful assistant.",
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

## API Usage

### Initializing the LLM Service

```python
from app.services.llm_service import LLMService

# Create service with assistant configuration
llm_service = LLMService(
    call_sid="unique_call_id",
    to_number="+1234567890",
    from_number="+0987654321",
    assistant=assistant  # Assistant object with LLM configuration
)

# Process a transcript
async def response_callback(content: str, is_final: bool, metadata: dict):
    print(f"Response: {content}, Final: {is_final}")

await llm_service.process_transcript(
    transcript="Hello, how can you help me?",
    is_final=True,
    metadata={},
    response_callback=response_callback
)
```

### Getting Provider Information

```python
# Get supported providers
providers = LLMService.get_supported_providers()
print(providers)  # ['openai', 'anthropic', 'gemini', 'xai', 'groq', 'custom']

# Get provider requirements
requirements = LLMService.get_provider_requirements('anthropic')
print(requirements)
# {
#     'required_fields': ['api_key'],
#     'optional_fields': ['base_url', 'model'],
#     'default_models': ['claude-3-5-sonnet-latest', 'claude-3-haiku-20240307'],
#     'pip_install': 'anthropic'
# }
```

## Provider-Specific Features

### Anthropic Claude
- Separate system prompts (handled automatically)
- Advanced reasoning capabilities
- Tool use support (coming soon)

### Google Gemini
- Multimodal capabilities (images, text)
- Long context support
- Thinking mode with reasoning effort

### xAI Grok
- Less restrictive content policies
- Real-time information access
- Humor and personality

### Groq
- Ultra-fast inference speed
- Multiple model options
- Cost-effective for high-volume use

## Best Practices

1. **Choose the Right Provider**:
   - **OpenAI**: General purpose, reliable
   - **Anthropic**: Complex reasoning, safety-focused
   - **Gemini**: Multimodal needs, Google integration
   - **xAI**: Creative tasks, less restrictive
   - **Groq**: Speed-critical applications

2. **Model Selection**:
   - Smaller models (8B) for simple tasks
   - Larger models (70B+) for complex reasoning
   - Consider cost vs. performance trade-offs

3. **Configuration**:
   - Lower temperature (0.1-0.3) for factual responses
   - Higher temperature (0.7-0.9) for creative tasks
   - Adjust max_tokens based on expected response length

4. **Error Handling**:
   - Always handle provider-specific errors
   - Implement fallback providers for critical applications
   - Monitor usage and rate limits

## Troubleshooting

### Common Issues

1. **ImportError: Package not installed**
   ```bash
   pip install anthropic  # or specific provider package
   ```

2. **API Key Not Found**
   ```python
   # Check environment variables
   import os
   print(os.getenv('ANTHROPIC_API_KEY'))
   ```

3. **Provider Not Supported**
   ```python
   # Check available providers
   print(LLMService.get_supported_providers())
   ```

4. **Model Not Available**
   ```python
   # Check provider requirements
   requirements = LLMService.get_provider_requirements('your_provider')
   print(requirements['default_models'])
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('app.services.llm_service').setLevel(logging.DEBUG)
```

### Testing Provider Configuration

```python
from app.services.llm_service import LLMService

# Test if provider is properly configured
try:
    service = LLMService(assistant=your_assistant)
    print(f"Successfully initialized {service._get_provider_name()} provider")
except Exception as e:
    print(f"Failed to initialize provider: {e}")
```

## Support and Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude API Documentation](https://docs.anthropic.com/en/api/getting-started)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [xAI Grok API Documentation](https://docs.x.ai/docs)
- [Groq API Documentation](https://console.groq.com/docs)

For questions or issues, please refer to the provider-specific documentation or create an issue in our repository. 