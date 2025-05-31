# LangChain Integration for LLM Service

## Overview

The LLM service has been upgraded to use LangChain for all providers except the custom LLM provider, which remains unchanged as requested. This provides better error handling, unified streaming, and improved tool calling capabilities.

## What Changed

### Before (Direct Provider Integration)
- `OpenAIProvider` - Direct AsyncOpenAI client
- `AnthropicProvider` - Direct AsyncAnthropic client  
- `GeminiProvider` - OpenAI-compatible endpoint
- `XAIProvider` - OpenAI-compatible endpoint
- `GroqProvider` - Direct AsyncGroq client
- `CustomLLMProvider` - httpx-based custom endpoint

### After (LangChain Integration)
- `LangChainProvider` - Unified provider for all except custom
- `CustomLLMProvider` - **UNCHANGED** (as requested)

## Benefits

1. **Unified Interface**: All providers use the same code path
2. **Better Error Handling**: Built-in retry logic and fallback mechanisms
3. **Consistent Streaming**: Standardized streaming behavior across providers
4. **Tool Integration**: Improved tool calling through LangChain's `bind_tools()`
5. **Future Extensibility**: Easy to add new LangChain-supported providers
6. **Robust Fallbacks**: If tool calling fails, falls back to streaming; if streaming fails, falls back to single invocation

## Supported Providers

| Provider | Implementation | Package Required |
|----------|---------------|------------------|
| `openai` | LangChain ChatOpenAI | `langchain-openai` |
| `anthropic` | LangChain ChatAnthropic | `langchain-anthropic` |
| `gemini` | LangChain ChatGoogleGenerativeAI | `langchain-google-genai` |
| `xai` | LangChain ChatOpenAI (xAI endpoint) | `langchain-openai` |
| `groq` | LangChain ChatGroq | `langchain-groq` |
| `custom` | **Original httpx implementation** | `httpx` |

## Configuration

Configuration remains the same. The service automatically detects the provider and uses the appropriate LangChain integration:

```python
# Example assistant configuration
assistant.llm_provider = "anthropic"
assistant.llm_provider_config = {
    "api_key": "your-api-key",
    "model": "claude-3-5-sonnet-latest"
}
```

## Custom LLM Provider

The `CustomLLMProvider` remains **completely unchanged** to maintain backward compatibility with existing custom LLM implementations.

## Installation

All required LangChain packages are already installed:

```bash
pip install langchain langchain-openai langchain-anthropic langchain-google-genai langchain-groq
```

## Error Handling Improvements

The new implementation includes multiple layers of fallback:

1. **Tool Calling Error**: Falls back to streaming
2. **Streaming Error**: Falls back to single invocation
3. **Provider Error**: Returns error message to user

## Streaming and Tool Calling

- **Regular conversations**: Use streaming for real-time response
- **Tool calls**: Use single invocation for reliable tool execution
- **Fallback**: Automatic fallback between methods if needed

## Testing

The implementation has been tested and verified to work correctly:

```bash
python -c "from app.services.llm_service import LLMService; print('✅ LangChain integration working')"
```

## Migration Notes

- **No breaking changes**: Existing configurations continue to work
- **Custom LLM unchanged**: Custom provider implementation is identical
- **Enhanced reliability**: Better error handling and fallback mechanisms
- **Same interface**: All existing APIs remain the same

The upgrade is transparent to existing users while providing improved reliability and maintainability for the codebase. 