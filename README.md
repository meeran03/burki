# Burqi

A comprehensive AI-powered voice assistant system that handles customer calls with advanced features including multi-tenant support, billing management, and real-time audio processing.

## 🚀 Features

### Core Voice AI Capabilities
- **Twilio Integration**: Handle phone calls through Twilio's Media Streams API with WebSocket support
- **Speech-to-Text**: Real-time transcription using Deepgram Nova 3 with confidence scoring
- **Natural Language Processing**: Generate intelligent responses with OpenAI GPT models or custom LLM providers
- **Text-to-Speech**: Convert responses to natural speech with ElevenLabs
- **Audio Denoising**: Built-in RNNoise integration for crystal-clear audio quality
- **Call Recording**: Automatic recording with transcription storage and metadata tracking

### Multi-Assistant Management
- **Multiple AI Assistants**: Configure different assistants with unique personalities and settings
- **Phone Number Assignment**: Assign specific assistants to different phone numbers
- **Database-Driven Configuration**: Store and manage assistant configurations dynamically
- **Real-time Assistant Loading**: Hot-reload assistant configurations without restart

### Web Dashboard & Management
- **Modern Web Interface**: Beautiful, responsive dashboard for managing your voice AI system
- **Real-time Analytics**: Advanced call statistics, success rates, and performance metrics
- **Call Management**: View call history, transcripts, and recordings through the web interface
- **Assistant Configuration**: Create and manage assistants through an intuitive web UI
- **User Profile Management**: Comprehensive user and organization management

### Authentication & Security
- **Multi-tenant Architecture**: Support for multiple organizations with isolated data
- **Google OAuth Integration**: Seamless authentication with Google accounts
- **API Key Management**: Generate and manage API keys for programmatic access
- **Session Management**: Secure session handling with JWT tokens
- **Role-based Access Control**: User roles and permissions system

### Billing & Subscription Management
- **Stripe Integration**: Complete billing system with Stripe payment processing
- **Usage Tracking**: Automatic tracking of call minutes and usage metrics
- **Subscription Plans**: Starter (500 free minutes) and Pro ($30/month for 1000 minutes) plans
- **Auto Top-up**: Automatic balance top-up when minutes run low
- **Usage Analytics**: Detailed billing reports and usage summaries

### Advanced Audio Processing
- **Real-time Noise Reduction**: RNNoise integration for superior audio quality
- **Voice Activity Detection**: Smart silence detection and speech processing
- **Audio Format Support**: Support for various audio formats with automatic conversion
- **Recording Quality Enhancement**: Post-processing tools for improving recording quality

### API & Integration
- **RESTful API**: Comprehensive REST API for all system operations
- **WebSocket Support**: Real-time communication for live call handling
- **Webhook Support**: Configurable webhooks for call events and notifications
- **Custom LLM Support**: Integrate with various LLM providers beyond OpenAI

## 📋 Requirements

- Python 3.11+
- PostgreSQL database
- Twilio account with phone number
- Deepgram API key
- OpenAI API key (or custom LLM API)
- ElevenLabs API key
- Stripe account (for billing features)
- Google OAuth credentials (for authentication)

## 🛠 Installation

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd diwaar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Server settings
PORT=8000
HOST=localhost

# Database settings
DATABASE_URL=postgresql://user:password@localhost:5432/diwaar_db

# Twilio credentials
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=your_twilio_phone_number

# Deepgram settings
DEEPGRAM_API_KEY=your-deepgram-api-key

# OpenAI settings
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=500

# ElevenLabs settings
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL

# Authentication & Security
SECRET_KEY=your-secret-key-for-sessions-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-this-in-production

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# Stripe Configuration (for billing)
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_PRO_PLAN_PRICE_ID=your-stripe-price-id

# Voice detection settings
SILENCE_MIN_DURATION_MS=500
ENERGY_THRESHOLD=50
WAIT_AFTER_SPEECH_MS=700
NO_PUNCTUATION_WAIT_MS=300
VOICE_SECONDS_THRESHOLD=2
WORD_COUNT_THRESHOLD=5

# Call settings
IDLE_TIMEOUT=30
MAX_IDLE_MESSAGES=3
END_CALL_MESSAGE=Thank you for calling. Goodbye!

# File Storage
RECORDINGS_DIR=recordings
```

### 3. Database Setup

Initialize the database with Alembic migrations:

```bash
alembic upgrade head
```

### 4. Audio Processing Setup (Optional)

For enhanced audio quality, install RNNoise:

```bash
./scripts/build_rnnoise.sh
```

## 🚀 Running the Application

### Development Mode

```bash
python -m app.main
```

### Production Mode with Gunicorn

```bash
gunicorn app.main:app --bind 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker --workers 2
```

The application will be available at:
- **Web Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
docker build -t diwaar .
docker run -p 8000:8000 --env-file .env diwaar
```

### Railway Deployment

The project includes Railway deployment configuration. See `docs/RAILWAY_DEPLOYMENT.md` for detailed instructions.

## 🧪 Testing

### Test Deepgram Transcription

Record test audio:
```bash
python -m tests.record_audio --output recordings/test.wav --duration 10
```

Test with microphone:
```bash
python -m tests.test_deepgram
```

Test with audio file:
```bash
python -m tests.test_deepgram --file recordings/test.wav
```

## 📚 API Documentation

### Assistant Management
- `GET /assistants` - List all assistants
- `POST /assistants` - Create a new assistant
- `GET /assistants/{assistant_id}` - Get a specific assistant
- `PUT /assistants/{assistant_id}` - Update an assistant
- `DELETE /assistants/{assistant_id}` - Delete an assistant
- `GET /assistants/by-phone/{phone_number}` - Get assistant by phone number

### Call Management
- `GET /calls` - List all calls with filtering options
- `GET /calls/{call_id}` - Get a specific call
- `GET /calls/sid/{call_sid}` - Get a call by Twilio Call SID
- `GET /calls/{call_id}/transcripts` - Get transcripts for a call
- `GET /calls/{call_id}/recordings` - Get recordings for a call

### Billing & Usage
- `GET /billing/plans` - List available billing plans
- `GET /billing/usage` - Get current usage statistics
- `POST /billing/upgrade` - Upgrade to a higher plan
- `POST /billing/topup` - Add minutes to account

### Authentication
- `POST /auth/login` - Login with email/password
- `GET /auth/google` - Google OAuth login
- `POST /auth/logout` - Logout current session
- `GET /auth/profile` - Get current user profile

### Twilio Integration
- `POST /twiml` - Generate TwiML for incoming calls
- `WebSocket /streams` - WebSocket endpoint for Twilio Media Streams

## 🎯 Quick Start Guide

### 1. Create Your First Assistant

Through the web interface:
1. Navigate to http://localhost:8000/dashboard
2. Click "Create Assistant"
3. Configure your assistant's personality and settings
4. Assign a phone number

Or via API:
```bash
curl -X POST http://localhost:8000/assistants \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support Assistant",
    "phone_number": "+11234567890",
    "openai_model": "gpt-4o",
    "system_prompt": "You are a helpful customer support agent...",
    "voice_id": "EXAVITQu4vr4xnSDxMaL"
  }'
```

### 2. Configure Twilio Webhook

Set your Twilio webhook URL to:
```
https://your-domain.com/twiml
```

### 3. Start Receiving Calls

Your AI assistant is now ready to handle incoming calls!

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Audio Denoising Setup](docs/AUDIO_DENOISING.md)** - Configure RNNoise for better audio quality
- **[Authentication Setup](docs/AUTHENTICATION_SETUP.md)** - Configure Google OAuth and user management
- **[LLM Providers Guide](docs/LLM_PROVIDERS_GUIDE.md)** - Integrate with different LLM providers
- **[Railway Deployment](docs/RAILWAY_DEPLOYMENT.md)** - Deploy to Railway platform
- **[Recording Setup](docs/RECORDING_SETUP.md)** - Configure call recording and storage
- **[Design Philosophy](docs/DESIGN_PHILOSOPHY.md)** - System architecture and design decisions

## 🛠 Utilities & Scripts

- **`scripts/build_rnnoise.sh`** - Build RNNoise for audio denoising
- **`scripts/verify_rnnoise.sh`** - Verify RNNoise installation
- **`scripts/fix_garbled_recordings.py`** - Fix corrupted audio recordings

## 🏗 Architecture

Burqi follows a modular architecture with clear separation of concerns:

- **`app/core/`** - Core business logic and managers
- **`app/services/`** - Service layer for external integrations
- **`app/api/`** - REST API endpoints and web routes
- **`app/db/`** - Database models and migrations
- **`app/utils/`** - Utility functions and helpers
- **`app/templates/`** - Web interface templates
- **`app/static/`** - Static assets for web interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Open an issue on GitHub
- Review the API documentation at `/docs` endpoint

---

**Burqi** - Transforming customer communication with AI-powered voice assistants.