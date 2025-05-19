# Diwaar

A system that uses AI to answer customer Calls.

## Features

- **Twilio Integration**: Handle phone calls through Twilio's Media Streams API
- **Speech-to-Text**: Real-time transcription using Deepgram Nova 3
- **Natural Language Processing**: Generate responses with OpenAI's
- **Text-to-Speech**: Convert responses to natural speech with ElevenLabs
- **Multi-Assistant Support**: 
  - Configure different assistants with unique personalities and settings
  - Assign assistants to different phone numbers
  - Store assistant configurations in database
- **Call Recording**:
  - Automatically record all calls
  - Store transcriptions in database
  - Track call metadata and outcomes
## Requirements

- Python 3.8+
- Twilio account with phone number
- Deepgram API key
- OpenAI API key or custom LLM API
- ElevenLabs API key
- PostgreSQL database

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Set up your environment variables in a `.env` file:
   ```
   # Server settings
   PORT=5678
   HOST=0.0.0.0

   # Database settings
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/volt_vapi

   # Twilio settings
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=your_phone_number

   # Deepgram settings
   DEEPGRAM_API_KEY=your_deepgram_api_key

   # OpenAI settings (optional if using custom LLM)
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4-turbo
   OPENAI_TEMPERATURE=0.7
   OPENAI_MAX_TOKENS=500


   # ElevenLabs settings
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_voice_id

   # Recording settings
   RECORDINGS_DIR=recordings
   ```

5. Initialize the database with Alembic:
   ```
   alembic upgrade head
   ```

## Running the Application

Start the server:

```
python -m app.main
```

The server will start on the configured port (default 5678) and will be available at http://localhost:5678

## Testing the Deepgram Transcription

You can test the Deepgram transcription service independently using the provided test script:

### Recording Test Audio

To record test audio for Deepgram testing:

```
python -m tests.record_audio --output recordings/test.wav --duration 10
```

This will record 10 seconds of audio from your microphone and save it to recordings/test.wav.

### Microphone Test
To test with your microphone input:

```
python -m tests.test_deepgram
```

### Audio File Test
To test with an audio file:

```
python -m tests.test_deepgram --file recordings/test.wav
```

Note: The audio file should be in the appropriate format (8kHz, mulaw encoding for Twilio compatibility).

## API Endpoints

### Assistant Management

- `GET /assistants` - List all assistants
- `POST /assistants` - Create a new assistant
- `GET /assistants/{assistant_id}` - Get a specific assistant
- `PUT /assistants/{assistant_id}` - Update an assistant
- `DELETE /assistants/{assistant_id}` - Delete an assistant
- `GET /assistants/by-phone/{phone_number}` - Get assistant by phone number

### Call Management

- `GET /calls` - List all calls
- `GET /calls/{call_id}` - Get a specific call
- `GET /calls/sid/{call_sid}` - Get a call by Twilio Call SID
- `GET /calls/{call_id}/transcripts` - Get transcripts for a call
- `GET /calls/{call_id}/recordings` - Get recordings for a call
- `GET /calls/sid/{call_sid}/transcripts` - Get transcripts by Call SID
- `GET /calls/sid/{call_sid}/recordings` - Get recordings by Call SID

### Twilio Integration

- `POST /twiml` - Generate TwiML for incoming calls
- `WebSocket /streams` - WebSocket endpoint for Twilio Media Streams

## Creating Assistants

You can create assistants through the API:

```bash
curl -X POST http://localhost:5678/assistants \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support Assistant",
    "phone_number": "+11234567890",
    "openai_model": "gpt-4-turbo",
    "system_prompt": "You are a helpful customer support agent...",
  }'
```