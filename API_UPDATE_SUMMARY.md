# API Update Summary

This document summarizes the major updates made to the Diwaar Voice AI API endpoints for assistants and calls.

## What Was Updated

### 1. Authentication System
- **Added API Key Authentication**: All endpoints now support API key authentication using Bearer tokens
- **Flexible Authentication**: Endpoints support both session-based auth (for web interface) and API key auth (for programmatic access)
- **Multi-tenant Security**: All endpoints are automatically filtered by organization - users can only access their own organization's data

### 2. Assistant API Improvements

#### New Features:
- **Multi-LLM Provider Support**: Updated to support OpenAI, Anthropic, Gemini, xAI, and Groq
- **Enhanced Configuration**: Full support for all new model fields including:
  - `llm_provider` and `llm_provider_config`
  - Updated `tts_settings`, `stt_settings`, and `interruption_settings`
  - Call control settings (`idle_message`, `end_call_message`, etc.)

#### New Endpoints:
- `GET /api/v1/assistants/count` - Get count of assistants
- `PATCH /api/v1/assistants/{id}/status` - Quick status toggle
- `GET /api/v1/assistants/providers` - List supported LLM providers
- `GET /api/v1/assistants/me/organization` - Get organization info

#### Enhanced Functionality:
- Organization-scoped operations (can only access your org's assistants)
- Better error handling with detailed messages
- Improved query parameters with validation
- Support for filtering by creator (`my_assistants_only`)

### 3. Calls API Improvements

#### New Features:
- Organization filtering for all call-related data
- Enhanced query parameters with validation
- Better relationship handling between calls and assistants

#### New Endpoints:
- `GET /api/v1/calls/count` - Get count of calls
- `GET /api/v1/calls/stats` - Comprehensive call statistics

#### Enhanced Functionality:
- Assistant ownership verification (can only access calls from your org's assistants)
- Improved error handling
- Better filtering options

### 4. Updated Data Models

#### Assistant Model Updates:
- Added support for new LLM provider configuration
- Enhanced settings for TTS, STT, and interruption handling
- Proper organization and user relationships
- Backward compatibility maintained for legacy fields

#### Schema Updates:
- Updated Pydantic schemas to match current model structure
- Added comprehensive validation
- Better field documentation
- Support for partial updates

### 5. Security Enhancements

#### Access Control:
- All operations are scoped to the authenticated user's organization
- Cross-organization data access is prevented
- API key permissions support (read, write, admin)
- Rate limiting configuration per API key

#### Authentication Improvements:
- Consistent authentication across all endpoints
- Proper HTTP status codes and error messages
- Support for both programmatic and web access

## Breaking Changes

### URL Changes:
- **Old**: `/api/assistants/` → **New**: `/api/v1/assistants/`
- **Old**: `/api/calls/` → **New**: `/api/v1/calls/`

### Schema Changes:
- `metadata` field in calls renamed to `call_meta` to match model
- Some optional fields are now properly marked as optional
- Enhanced validation for all input fields

### Authentication:
- **All endpoints now require authentication** (previously some were open)
- API responses are filtered by organization (can only see your own data)

## Migration Guide

### For Existing API Users:

1. **Update Base URLs**: Change `/api/` to `/api/v1/`
2. **Add Authentication**: Include API key in Authorization header
3. **Update Field Names**: Use `call_meta` instead of `metadata` for calls
4. **Handle Organization Scoping**: Data is now automatically filtered by your organization

### For New Users:

1. **Get an API Key**: Log into the web interface and create an API key
2. **Use New Endpoints**: Start with the `/api/v1/` endpoints
3. **Follow Documentation**: Use the comprehensive API documentation provided

## Example Usage

```python
import requests

headers = {
    "Authorization": "Bearer diwaar_your_api_key_here",
    "Content-Type": "application/json"
}

# Create assistant with new provider support
assistant_data = {
    "name": "My Assistant",
    "phone_number": "+1234567890",
    "llm_provider": "openai",
    "llm_provider_config": {
        "api_key": "your_openai_key",
        "model": "gpt-4o-mini"
    }
}

response = requests.post(
    "https://yourdomain.com/api/v1/assistants/",
    headers=headers,
    json=assistant_data
)
```

## Files Changed

### Core Files:
- `app/core/auth.py` (new) - Authentication dependencies
- `app/api/assistants.py` - Updated assistant endpoints
- `app/api/calls.py` - Updated call endpoints
- `app/api/schemas.py` - Updated data schemas
- `app/services/assistant_service.py` - Enhanced with organization filtering
- `app/main.py` - Updated router includes

### Documentation:
- `API_ENDPOINTS.md` - Comprehensive API documentation
- `API_UPDATE_SUMMARY.md` - This summary document
- `examples/api_example.py` - Python usage example

## Testing

All endpoints have been updated to support the new authentication system and organization filtering. Test with:

1. **Valid API Key**: Should access organization-scoped data
2. **Invalid API Key**: Should return 401 Unauthorized
3. **Cross-Organization Access**: Should be prevented (404 Not Found)
4. **New Model Fields**: Should accept and validate new configuration options

## Support

For questions about the API updates:
1. Check the comprehensive API documentation in `API_ENDPOINTS.md`
2. Review the example script in `examples/api_example.py`
3. Test with your API key using the documented endpoints

The updated API provides better security, multi-tenant support, and enhanced functionality while maintaining backward compatibility where possible. 