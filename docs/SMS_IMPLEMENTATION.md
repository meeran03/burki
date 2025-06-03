# SMS Implementation Guide

## Overview

This document describes the SMS functionality that has been added to the Diwaar system, which previously only supported voice calls. The implementation maintains backward compatibility while introducing a more generic conversation model that can support multiple communication channels (SMS, WhatsApp, Telegram, etc.).

## Architecture Changes

### Database Schema Evolution

#### From Call to Conversation Model

The system has evolved from a call-specific model to a generic `Conversation` model:

**Old Model (Call):**
```
- call_sid (unique identifier)
- assistant_id
- to_phone_number
- customer_phone_number
- status
- duration
- call_meta
```

**New Model (Conversation):**
```
- channel_sid (generic identifier - can be call_sid, message_sid, etc.)
- conversation_type (call, sms, whatsapp, telegram)
- assistant_id
- to_phone_number
- customer_phone_number
- status
- duration (nullable for non-call conversations)
- conversation_metadata
```

### Backward Compatibility

The system maintains full backward compatibility:

1. **Model Aliases**: `Call = Conversation` alias allows existing code to continue working
2. **Property Mapping**: All models have backward-compatible properties (e.g., `call_id` → `conversation_id`)
3. **Legacy Relationships**: Models maintain legacy relationship properties (e.g., `call` → `conversation`)

### New Components

1. **SMSHandler** (`app/core/sms_handler.py`)
   - Manages SMS conversation state
   - Processes incoming SMS messages
   - Integrates with LLMService for responses
   - No streaming or real-time requirements

2. **SMS Endpoints** (`app/api/root.py`)
   - `/sms/webhook` - Handles incoming SMS from Twilio
   - `/sms/send` - API endpoint to send SMS messages

3. **Enhanced TwilioService** (`app/twilio/twilio_service.py`)
   - `send_sms()` - Send SMS messages
   - `get_message_info()` - Retrieve message details

## How SMS Works

### Incoming SMS Flow

1. **Twilio Webhook**: When an SMS is received, Twilio calls `/sms/webhook`
2. **Assistant Lookup**: System finds the assistant by the recipient phone number
3. **Billing Check**: Verifies organization has available credits/limits
4. **Create Conversation**: Creates a conversation record with type "sms"
5. **Process Message**: SMSHandler processes the message through LLMService
6. **Store Messages**: Both user and assistant messages are stored as ChatMessages
7. **Send Response**: Response is sent back via Twilio SMS API
8. **Webhooks**: Status updates are sent to configured webhook URLs

### Outbound SMS Flow

1. **API Call**: Client calls `/sms/send` with assistant_id, phone number, and message
2. **Validation**: Phone number format and assistant existence are validated
3. **Send SMS**: Message is sent via Twilio
4. **Create Conversation**: Outbound conversation record is created
5. **Store Message**: Assistant message is stored
6. **Billing**: SMS usage is recorded

## API Usage

### Webhook Configuration (Twilio Console)

Configure your Twilio phone number to use these webhooks:
- **SMS Webhook**: `https://your-domain.com/sms/webhook` (POST)

### Sending SMS via API

```bash
curl -X POST https://your-domain.com/sms/send \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": 123,
    "to_phone_number": "+1234567890",
    "message_body": "Hello from your AI assistant!"
  }'
```

Response:
```json
{
  "success": true,
  "message_sid": "SM...",
  "message": "SMS sent successfully",
  "assistant_id": 123,
  "to_phone_number": "+1234567890",
  "from_phone_number": "+0987654321"
}
```

### Sending Agenda-Based SMS

Similar to outbound calls, you can send SMS with an agenda. When the customer replies, the AI will respond based on the agenda context:

```bash
curl -X POST https://your-domain.com/sms/send \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": 123,
    "to_phone_number": "+1234567890",
    "message_body": "Hi! This is your healthcare assistant. I'm reaching out about your upcoming appointment on Monday.",
    "agenda": "Confirm the patient's appointment on Monday at 2 PM. If they cannot make it, offer to reschedule for Tuesday or Wednesday. Collect their preferred time slot."
  }'
```

Response:
```json
{
  "success": true,
  "message_sid": "SM...",
  "message": "SMS sent successfully",
  "assistant_id": 123,
  "to_phone_number": "+1234567890",
  "from_phone_number": "+0987654321",
  "agenda": "Confirm the patient's appointment..."
}
```

When the customer replies:
1. The system recognizes this is a continuation of an agenda-based conversation
2. The agenda is injected into the AI's context
3. The AI responds according to the agenda and the conversation history
4. The conversation continues until completed

Example conversation flow:
```
Assistant: "Hi! This is your healthcare assistant. I'm reaching out about your upcoming appointment on Monday."
Customer: "What time was that again?"
Assistant: "Your appointment is scheduled for Monday at 2 PM. Will you be able to make it?"
Customer: "No, I need to reschedule"
Assistant: "No problem! I can offer you slots on Tuesday or Wednesday. What time works best for you?"
```

## Database Migration

To migrate from the old Call model to the new Conversation model:

```bash
python app/db/migrations/migrate_call_to_conversation.py
```

This migration:
1. Creates the new `conversations` table
2. Migrates all existing call records
3. Updates foreign key references in related tables
4. Maintains backward compatibility

## Billing Integration

SMS usage is tracked separately from call minutes:
- `UsageRecord` now includes `messages_used` field
- `usage_type` can be "sms" for message usage
- Billing limits are checked before processing SMS

## Webhook Events

New webhook types for SMS:
- `sms-received` - Incoming SMS received
- `sms-sent` - Outbound SMS sent
- `sms-error` - Error processing SMS

## Key Differences from Call Handling

1. **No Streaming**: SMS doesn't require WebSocket connections or streaming
2. **No TTS/STT**: No text-to-speech or speech-to-text services needed
3. **Simpler State**: No complex state management for interruptions, timeouts, etc.
4. **Request-Response**: Simple request-response pattern vs. real-time interaction
5. **Agenda Support**: Both calls and SMS support agenda-based conversations for business workflows

## Conversation State Management

### Configuration
Each assistant can have custom SMS settings configured in the `sms_settings` field:

```json
{
  "sms_settings": {
    "enabled": true,
    "conversation_ttl_hours": 24,        // How long to keep conversations in memory/Redis
    "max_conversation_length": 50,       // Max messages per conversation
    "auto_end_after_hours": 72,         // Auto-end after this many hours of inactivity
    "welcome_message": null,            // Optional custom SMS welcome
    "rate_limit": {
      "enabled": true,
      "max_messages_per_hour": 60,
      "max_messages_per_day": 100
    },
    "redis_persistence": {
      "enabled": true,                  // Use Redis for persistence
      "ttl_hours": 24                  // Redis TTL (should match conversation_ttl_hours)
    }
  }
}
```

### Persistence with Redis

The system now uses Redis for SMS conversation persistence:

1. **Automatic Save**: Conversations are saved to Redis after each message
2. **TTL Support**: Conversations expire based on assistant's `ttl_hours` setting
3. **Graceful Fallback**: If Redis is unavailable, falls back to memory-only storage
4. **Server Restart Recovery**: Conversations are loaded from Redis on first access

#### Redis Configuration

Set the Redis URL via environment variable:
```bash
export REDIS_URL="redis://localhost:6379"
# Or with authentication:
export REDIS_URL="redis://username:password@redis-host:6379/0"
```

### Duration and Cleanup

- **Configurable TTL**: Each assistant can set its own conversation TTL (default: 24 hours)
- **Activity-Based**: TTL is based on last activity, not conversation start
- **Auto-Cleanup**: Background task runs hourly to clean up expired conversations
- **Inactivity Timeout**: Conversations auto-end after `auto_end_after_hours` of inactivity

### State Persistence Features

1. **Conversation History**: Full LLM conversation history is preserved
2. **Agenda Context**: Outbound SMS agendas are maintained across restarts
3. **Metadata Preservation**: All conversation metadata is saved
4. **Activity Tracking**: Last activity time is tracked for accurate cleanup

## Production Recommendations

1. **Redis Setup**:
   - Use Redis Cluster for high availability
   - Enable Redis persistence (RDB/AOF) for data durability
   - Monitor Redis memory usage and set appropriate `maxmemory` policies

2. **TTL Configuration**:
   - Short TTL (1-6 hours) for transactional messages
   - Medium TTL (24-48 hours) for customer support
   - Long TTL (72+ hours) for appointment scheduling or surveys

3. **Monitoring**:
   - Track Redis connection failures
   - Monitor conversation count and memory usage
   - Alert on high conversation volume or memory pressure

4. **Scaling**:
   - Redis handles distributed state across multiple servers
   - Each server maintains a local cache for active conversations
   - Redis serves as the source of truth for conversation state

## Environment Variables

```bash
# Required for SMS functionality
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token

# Redis configuration (optional, defaults to localhost)
REDIS_URL=redis://localhost:6379

# Optional Redis with authentication
REDIS_URL=redis://username:password@redis-host:6379/0

# Optional Redis with TLS
REDIS_URL=rediss://redis-host:6379/0
```

## Future Extensibility

The conversation model is designed to support additional channels:
- WhatsApp (conversation_type: "whatsapp")
- Telegram (conversation_type: "telegram")
- Web Chat (conversation_type: "webchat")

Each new channel would:
1. Add a new handler (e.g., WhatsAppHandler)
2. Add channel-specific endpoints
3. Use the same Conversation model
4. Reuse existing LLMService and billing infrastructure

## Error Handling

- Invalid phone numbers return 400 error
- Missing assistants return 404 error
- Billing limits return 429 error with appropriate message
- All errors are logged and webhooks are sent when possible
- Twilio webhooks always return 200 to prevent retries

## Testing

To test SMS functionality:

1. Configure a Twilio phone number with SMS capabilities
2. Set the SMS webhook URL in Twilio console
3. Send a test SMS to the assistant's phone number
4. Check logs and database for conversation records
5. Verify billing usage is recorded correctly

## Limitations

- No MMS (media) support currently
- SMS length limits apply (160 characters per segment)
- No delivery receipts beyond Twilio's status callbacks
- Conversation state is kept in memory (consider Redis for production)

## Security Considerations

- Validate all phone numbers in E.164 format
- Check billing limits before processing
- Sanitize message content before storage
- Use environment variables for sensitive credentials
- Implement rate limiting for API endpoints (not included in base implementation) 