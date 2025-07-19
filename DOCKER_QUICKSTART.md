# 🚀 Docker Quick Start

Get Burki Voice AI running in **5 minutes** with Docker!

## Step 1: Prerequisites

Install these on your computer:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- [Git](https://git-scm.com/downloads)

## Step 2: Get API Keys

You need accounts with these services (all have free tiers):

1. **Twilio** → [Sign up](https://www.twilio.com/try-twilio) → Get Account SID, Auth Token, and Phone Number
2. **Deepgram** → [Sign up](https://console.deepgram.com/signup) → Get API Key  
3. **OpenAI** → [Sign up](https://platform.openai.com/signup) → Get API Key
4. **ElevenLabs** → [Sign up](https://elevenlabs.io/sign-up) → Get API Key

## Step 3: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/burki-voice-ai.git
cd burki-voice-ai

# Copy the environment template
cp .env.example .env
```

## Step 4: Add Your API Keys

Edit the `.env` file and add your API keys:

```env
# Replace with your actual values
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
SECRET_KEY=make-this-a-long-random-string
```

## Step 5: Deploy!

```bash
# Make script executable
chmod +x deploy.sh

# Deploy (this will take a few minutes the first time)
./deploy.sh
```

## Step 6: Test It

1. **Open your browser** → `http://localhost:8000`
2. **Create your first assistant** in the web dashboard
3. **Test with a phone call** to your Twilio number

## 🎉 That's it!

Your Burki Voice AI is now running! 

### Next Steps:
- Configure your assistant's personality and voice
- Set up Twilio webhooks to point to your server
- Add knowledge base documents for smarter responses

### Need Help?
- Check the [full Docker guide](README-DOCKER.md) for troubleshooting
- View logs: `docker-compose logs -f`
- Stop services: `docker-compose down`

---

**Pro Tip**: For production deployment, see our [Railway deployment guide](docs/RAILWAY_DEPLOYMENT.md) for one-click cloud hosting! 