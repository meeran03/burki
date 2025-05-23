# API Endpoints Documentation

This document describes the updated API endpoints for Diwaar Voice AI, including authentication requirements and usage examples.

## Authentication

All API endpoints require authentication. You can authenticate using either:

1. **API Key** (recommended for programmatic access)
2. **Session-based authentication** (for web interface)

### API Key Authentication

Include your API key in the `Authorization` header:

```bash
Authorization: Bearer diwaar_your_api_key_here
```

### Getting an API Key

1. Log into the web interface
2. Go to "API Keys" section in your profile
3. Create a new API key with appropriate permissions
4. Copy the key (it will only be shown once)

## Base URL

All API requests should be made to: `https://yourdomain.com/api/v1/`

## Assistants API

### Create Assistant

**POST** `/api/v1/assistants/`

Create a new voice assistant.

```bash
curl -X POST "https://yourdomain.com/api/v1/assistants/" \
  -H "Authorization: Bearer diwaar_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Service Bot",
    "phone_number": "+1234567890",
    "description": "Handles customer inquiries",
    "llm_provider": "openai",
    "llm_provider_config": {
      "api_key": "your_openai_key",
      "model": "gpt-4o-mini"
    },
    "llm_settings": {
      "temperature": 0.7,
      "max_tokens": 1000,
      "system_prompt": "You are a helpful customer service representative."
    },
    "tts_settings": {
      "voice_id": "rachel",
      "model_id": "turbo"
    },
    "is_active": true
  }'
```

### List Assistants

**GET** `/api/v1/assistants/`

Get all assistants for your organization.

**Query Parameters:**
- `skip` (int, default: 0): Number of items to skip
- `limit` (int, default: 100): Maximum items to return (max: 1000)
- `active_only` (bool, default: false): Only return active assistants
- `my_assistants_only` (bool, default: false): Only return assistants created by you

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/?active_only=true&limit=50"
```

### Get Assistant by ID

**GET** `/api/v1/assistants/{assistant_id}`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/123"
```

### Get Assistant by Phone Number

**GET** `/api/v1/assistants/by-phone/{phone_number}`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/by-phone/+1234567890"
```

### Update Assistant

**PUT** `/api/v1/assistants/{assistant_id}`

Update an existing assistant.

```bash
curl -X PUT "https://yourdomain.com/api/v1/assistants/123" \
  -H "Authorization: Bearer diwaar_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Assistant Name",
    "llm_settings": {
      "temperature": 0.8,
      "system_prompt": "Updated system prompt"
    }
  }'
```

### Update Assistant Status

**PATCH** `/api/v1/assistants/{assistant_id}/status`

Quick endpoint to activate/deactivate an assistant.

**Query Parameters:**
- `is_active` (bool, required): Whether to activate or deactivate the assistant

```bash
curl -X PATCH "https://yourdomain.com/api/v1/assistants/123/status?is_active=false" \
  -H "Authorization: Bearer diwaar_your_api_key_here"
```

### Delete Assistant

**DELETE** `/api/v1/assistants/{assistant_id}`

```bash
curl -X DELETE "https://yourdomain.com/api/v1/assistants/123" \
  -H "Authorization: Bearer diwaar_your_api_key_here"
```

### Get Assistants Count

**GET** `/api/v1/assistants/count`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/count?active_only=true"
```

Response:
```json
{
  "count": 5
}
```

### Get Supported LLM Providers

**GET** `/api/v1/assistants/providers`

Get list of supported LLM providers and their models.

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/providers"
```

### Get Organization Info

**GET** `/api/v1/assistants/me/organization`

Get information about your organization.

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/assistants/me/organization"
```

## Calls API

### List Calls

**GET** `/api/v1/calls/`

Get all calls for your organization.

**Query Parameters:**
- `skip` (int, default: 0): Number of items to skip
- `limit` (int, default: 100): Maximum items to return (max: 1000)
- `status` (string): Filter by call status (ongoing, completed, failed)
- `assistant_id` (int): Filter by assistant ID

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/?status=completed&limit=50"
```

### Get Call by ID

**GET** `/api/v1/calls/{call_id}`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/123"
```

### Get Call by SID

**GET** `/api/v1/calls/sid/{call_sid}`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/sid/CA1234567890abcdef"
```

### Get Call Transcripts

**GET** `/api/v1/calls/{call_id}/transcripts`

**Query Parameters:**
- `speaker` (string): Filter by speaker (user, assistant)

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/123/transcripts?speaker=user"
```

### Get Call Transcripts by SID

**GET** `/api/v1/calls/sid/{call_sid}/transcripts`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/sid/CA1234567890abcdef/transcripts"
```

### Get Call Recordings

**GET** `/api/v1/calls/{call_id}/recordings`

**Query Parameters:**
- `recording_type` (string): Filter by recording type

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/123/recordings"
```

### Get Call Recordings by SID

**GET** `/api/v1/calls/sid/{call_sid}/recordings`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/sid/CA1234567890abcdef/recordings"
```

### Get Calls Count

**GET** `/api/v1/calls/count`

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/count?status=completed"
```

### Get Call Statistics

**GET** `/api/v1/calls/stats`

Get comprehensive call statistics for your organization.

```bash
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
  "https://yourdomain.com/api/v1/calls/stats"
```

Response:
```json
{
  "total_calls": 150,
  "ongoing_calls": 2,
  "completed_calls": 140,
  "failed_calls": 8,
  "total_duration_seconds": 45000,
  "average_duration_seconds": 321.43,
  "success_rate": 93.33
}
```

## New Features

### LLM Provider Support

The updated assistant API now supports multiple LLM providers:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- **Google Gemini**: `gemini-1.5-pro`, `gemini-1.5-flash`
- **xAI**: `grok-beta`
- **Groq**: `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`

Configure providers using the `llm_provider` and `llm_provider_config` fields.

### Multi-Tenant Security

- All endpoints are automatically filtered by your organization
- You can only access assistants and calls that belong to your organization
- Cross-organization data access is prevented

### Enhanced Settings

The assistant model now supports comprehensive configuration:

- **LLM Settings**: temperature, max_tokens, system_prompt, etc.
- **TTS Settings**: voice configuration, latency, stability
- **STT Settings**: model, language, punctuation, endpointing
- **Interruption Settings**: threshold, cooldown, timing
- **Call Control**: end call messages, idle handling

### Improved Error Handling

All endpoints now return consistent error messages with proper HTTP status codes and detailed information about what went wrong.

## Rate Limiting

API keys have configurable rate limits:
- Requests per minute: 100 (default)
- Requests per hour: 1,000 (default)
- Requests per day: 10,000 (default)

These limits can be configured when creating API keys.

## Permissions

API keys support granular permissions:
- **Read**: Access to GET endpoints
- **Write**: Access to POST, PUT, PATCH endpoints
- **Admin**: Access to DELETE endpoints and administrative functions

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "detail": "Error description",
  "type": "error_type"
}
```

Common HTTP status codes:
- `401`: Authentication required or invalid API key
- `403`: Insufficient permissions
- `404`: Resource not found in your organization
- `400`: Bad request (validation errors)
- `500`: Internal server error 