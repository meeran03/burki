# Burki

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

## 🛠 Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/meeran03/burki.git
cd burki
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
docker build -t burki .
docker run -p 8000:8000 --env-file .env burki
```

### Railway Deployment

The project includes Railway deployment configuration. See `docs/RAILWAY_DEPLOYMENT.md` for detailed instructions.

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Audio Denoising Setup](docs/AUDIO_DENOISING.md)** - Configure RNNoise for better audio quality
- **[Railway Deployment](docs/RAILWAY_DEPLOYMENT.md)** - Deploy to Railway platform
- **[Design Philosophy](docs/DESIGN_PHILOSOPHY.md)** - System architecture and design decisions

## 🛠 Utilities & Scripts

- **`scripts/build_rnnoise.sh`** - Build RNNoise for audio denoising
- **`scripts/verify_rnnoise.sh`** - Verify RNNoise installation

## 🏗 Architecture

Burki follows a modular architecture with clear separation of concerns:

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

**Burki** - Transforming customer communication with AI-powered voice assistants.