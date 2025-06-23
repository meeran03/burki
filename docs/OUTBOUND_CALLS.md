# Outbound Calls Feature

This document describes the outbound calls functionality that allows you to programmatically initiate phone calls through the API with custom welcome messages and agendas.

## Overview

The outbound calls feature enables your AI assistants to make phone calls to customers or leads with a specific purpose (agenda). This is useful for:

- Customer follow-ups
- Appointment reminders
- Sales outreach
- Survey calls
- Support check-ins
- Lead qualification

## API Endpoint

### `POST /calls/initiate`

Initiates an outbound call through the Twilio API.

#### Request Body

```json
{
  "assistant_id": 123,
  "to_phone_number": "+1234567890",
  "welcome_message": "Hello! This is your AI assistant calling from Acme Corp. I hope you're having a great day!",
  "agenda": "I'm calling to follow up on your recent inquiry about our services and see if you have any questions."
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `assistant_id` | integer | Yes | The ID of the assistant that will make the call |
| `to_phone_number` | string | Yes | The phone number to call (E.164 format, e.g., +1234567890) |
| `welcome_message` | string | No | Custom greeting message (overrides assistant's default) |
| `agenda` | string | Yes | The purpose/agenda of the call - this guides the AI's conversation |

#### Response

**Success (200)**:
```json
{
  "success": true,
  "call_sid": "CA1234567890abcdef1234567890abcdef",
  "message": "Outbound call initiated successfully",
  "assistant_id": 123,
  "to_phone_number": "+1234567890",
  "from_phone_number": "+0987654321"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid phone number format. Use E.164 format (e.g., +1234567890)"
}
```

**Error (404)**:
```json
{
  "detail": "Assistant not found"
}
```

**Error (429)**:
```json
{
  "detail": "Monthly usage limit exceeded. Please upgrade your plan or add top-up credits."
}
```

## How It Works

### 1. Call Initiation
When you make a POST request to `/calls/initiate`:

1. **Validation**: The system validates the phone number format, assistant existence, and billing limits
2. **Twilio Call**: A call is initiated through Twilio using the assistant's phone number
3. **Metadata Passing**: The welcome message and agenda are passed as URL parameters to the webhook
4. **Response**: Returns the call SID and confirmation

### 2. Call Handling
Once Twilio connects the call:

1. **TwiML Generation**: The `/twiml` endpoint detects it's an outbound call and extracts metadata
2. **WebSocket Connection**: A bi-directional audio stream is established
3. **Agenda Injection**: The agenda is injected into the AI's conversation context
4. **Custom Welcome**: The custom welcome message (if provided) is used instead of the default

### 3. AI Conversation
The AI assistant will:

1. **Start with Welcome**: Use the custom welcome message or fallback to the assistant's default
2. **Follow Agenda**: The agenda is injected as a system message to guide the conversation
3. **Natural Flow**: Conduct a natural conversation while staying focused on the agenda
4. **Professional Handling**: Handle objections, questions, and requests professionally

## Usage Examples

### Basic Sales Follow-up
```bash
curl -X POST "https://your-domain.com/calls/initiate" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": 1,
    "to_phone_number": "+1234567890",
    "welcome_message": "Hi! This is Sarah from TechCorp. Thanks for your interest in our AI solutions!",
    "agenda": "I wanted to follow up on the demo request you submitted yesterday and answer any questions you might have about our AI customer service platform."
  }'
```

### Appointment Reminder
```bash
curl -X POST "https://your-domain.com/calls/initiate" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": 2,
    "to_phone_number": "+1234567890",
    "welcome_message": "Hello! This is a friendly reminder call from MedCenter.",
    "agenda": "I'm calling to remind you about your appointment tomorrow at 2 PM with Dr. Smith. Please confirm if you can still make it or if you need to reschedule."
  }'
```

### Customer Satisfaction Survey
```bash
curl -X POST "https://your-domain.com/calls/initiate" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": 3,
    "to_phone_number": "+1234567890",
    "welcome_message": "Hi! I'm calling from CustomerFirst to get your feedback.",
    "agenda": "We recently provided you with customer support, and I'd love to hear about your experience. This will only take 2-3 minutes and helps us improve our service."
  }'
```

## Configuration Requirements

### Assistant Setup
Your assistant must have:

1. **Phone Number**: A valid Twilio phone number assigned
2. **Twilio Credentials**: Account SID and Auth Token (can be at assistant or environment level)
3. **Active Status**: The assistant must be active (`is_active = true`)
4. **Billing**: Sufficient credits/within billing limits for your organization

### Twilio Requirements
- Valid Twilio account with phone numbers
- Sufficient account balance for outbound calls
- Proper webhook configuration (handled automatically)

### Billing Considerations
- Outbound calls count toward your monthly usage limits
- Billing is calculated based on call duration
- Failed calls (busy, no answer) may still incur minimal charges from Twilio

## Best Practices

### Agenda Writing
Write clear, specific agendas that help the AI stay focused:

**Good**:
```
"I'm calling to follow up on your recent inquiry about our AI chatbot service. I want to understand your current customer service challenges and explain how our solution could help reduce response times and improve customer satisfaction."
```

**Bad**:
```
"Talk about our service."
```

### Welcome Message Guidelines
- Keep it under 15 seconds when spoken
- Be clear about who's calling and why
- Set proper expectations
- Use a friendly, professional tone

### Phone Number Format
Always use E.164 format:
- ✅ `+1234567890` (US)
- ✅ `+447123456789` (UK)
- ✅ `+8612345678901` (China)
- ❌ `1234567890` (missing +)
- ❌ `(123) 456-7890` (wrong format)

### Error Handling
Always handle potential errors:
- Invalid phone numbers
- Assistant not found
- Billing limits exceeded
- Twilio API failures

## Integration Examples

### Python with httpx
```python
import httpx
import asyncio

async def make_outbound_call():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://your-domain.com/calls/initiate",
            json={
                "assistant_id": 1,
                "to_phone_number": "+1234567890",
                "welcome_message": "Hello from our AI assistant!",
                "agenda": "I'm calling to check on your recent order and ensure everything arrived safely."
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Call initiated: {result['call_sid']}")
        else:
            print(f"Error: {response.text}")

asyncio.run(make_outbound_call())
```

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function makeOutboundCall() {
  try {
    const response = await axios.post('https://your-domain.com/calls/initiate', {
      assistant_id: 1,
      to_phone_number: '+1234567890',
      welcome_message: 'Hello from our AI assistant!',
      agenda: 'I\'m calling to check on your recent order and ensure everything arrived safely.'
    });
    
    console.log('Call initiated:', response.data.call_sid);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

makeOutboundCall();
```

### PHP
```php
<?php
$data = [
    'assistant_id' => 1,
    'to_phone_number' => '+1234567890',
    'welcome_message' => 'Hello from our AI assistant!',
    'agenda' => 'I\'m calling to check on your recent order and ensure everything arrived safely.'
];

$options = [
    'http' => [
        'header' => "Content-Type: application/json\r\n",
        'method' => 'POST',
        'content' => json_encode($data)
    ]
];

$context = stream_context_create($options);
$result = file_get_contents('https://your-domain.com/calls/initiate', false, $context);

if ($result !== FALSE) {
    $response = json_decode($result, true);
    echo "Call initiated: " . $response['call_sid'];
} else {
    echo "Error initiating call";
}
?>
```

## Monitoring and Analytics

### Call Tracking
- All outbound calls are tracked in the database
- Use the returned `call_sid` to monitor call status
- Check call duration and outcome in your dashboard

### Webhooks
If your assistant has webhooks configured, you'll receive:
- Call started notifications
- Call ended notifications with recordings (if enabled)
- Transcript data
- Call analytics

### Billing Tracking
- Monitor usage through the billing dashboard
- Set up usage alerts to avoid unexpected charges
- Track ROI on outbound call campaigns

## Troubleshooting

### Common Issues

**Call not connecting**:
- Check phone number format (must be E.164)
- Verify Twilio credentials
- Ensure sufficient Twilio account balance
- Check if number is reachable (not blocked)

**Assistant not found**:
- Verify assistant ID exists
- Ensure assistant is active
- Check organization permissions

**Billing limits exceeded**:
- Check current usage in billing dashboard
- Upgrade plan or add top-up credits
- Contact support for assistance

**Invalid welcome message**:
- Keep message under 200 characters
- Avoid special characters that might break TTS
- Test with different messages

### Debugging
- Check server logs for detailed error messages
- Use the test script (`test_outbound_calls.py`) to validate setup
- Monitor Twilio console for call status and debugging info

## Security Considerations

- **Rate Limiting**: Implement rate limiting to prevent abuse
- **Authentication**: Secure your API endpoints with proper authentication
- **Phone Validation**: Always validate phone numbers before making calls
- **Compliance**: Ensure compliance with local telemarketing laws (TCPA, GDPR, etc.)
- **Consent**: Only call numbers that have given consent for automated calls

## Limitations

- Calls are limited by Twilio's concurrent call limits
- Subject to organization billing limits
- Agenda must be provided in text format (under 2000 characters recommended)
- Welcome message should be under 200 characters for best results
- Requires internet connectivity for real-time audio processing

## Support

For support with outbound calls:
1. Check this documentation first
2. Review server logs for error details
3. Test with the provided test script
4. Contact technical support with specific error messages and call SIDs 