# 🐳 Docker Deployment Guide

This guide shows you how to deploy Burki Voice AI using Docker Compose with our **one-click deployment system**.

## 🚀 Quick Start (One-Click Deploy)

### 1. Prerequisites

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Git** for cloning the repository
- **API Keys** from required providers (see below)

### 2. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/meeran03/burki.git
cd burki

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (see Configuration section below)
nano .env  # or use your preferred editor
```

### 3. One-Click Deploy

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Deploy in production mode (default)
./deploy.sh

# Or deploy in development mode
./deploy.sh --dev
```

That's it! 🎉 Your Burki Voice AI will be running at `http://localhost:8000`

---

## 📋 Required Configuration

### Essential API Keys

Before deploying, you **must** configure these in your `.env` file:

| Provider | Required | Get From |
|----------|----------|----------|
| **Twilio** | ✅ Yes | [Twilio Console](https://console.twilio.com) |
| **Deepgram** | ✅ Yes | [Deepgram Console](https://console.deepgram.com) |
| **OpenAI** | ✅ Yes | [OpenAI Platform](https://platform.openai.com/api-keys) |
| **ElevenLabs** | ✅ Yes | [ElevenLabs API](https://elevenlabs.io/app/settings/api-keys) |

### Example .env Configuration

```env
# Required - Get from Twilio Console
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Required - Get from Deepgram Console
DEEPGRAM_API_KEY=your_deepgram_api_key

# Required - Get from OpenAI Platform
OPENAI_API_KEY=your_openai_api_key

# Required - Get from ElevenLabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Required - Change this to a secure random string
SECRET_KEY=your-very-secure-secret-key-here
```

---

## 🛠️ Manual Docker Commands

If you prefer manual control over the one-click script:

### Start Services

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### Run Database Migrations

```bash
# Run Alembic migrations
docker-compose exec burki alembic upgrade head

# Create a default assistant (optional)
docker-compose exec burki python -m app.db.init_db
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes all data)
docker-compose down -v
```

---

## 🏗️ Architecture Overview

Our Docker Compose setup includes:

| Service | Purpose | Port |
|---------|---------|------|
| **burki** | Main application | 8000 |
| **postgres** | Database | 5432 |
| **redis** | Caching & sessions | 6379 |

### Data Persistence

The following volumes ensure your data persists across container restarts:

- `postgres_data` - Database storage
- `redis_data` - Redis cache
- `app_data` - Application data
- `recordings` - Call recordings
- `logs` - Application logs

---

## 🔧 Development vs Production

### Development Mode

```bash
# Deploy with development settings
./deploy.sh --dev

# Or manually
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

**Development features:**
- Debug logging enabled
- Hot reload (if configured)
- Development-friendly settings

### Production Mode

```bash
# Deploy with production settings (default)
./deploy.sh --prod

# Or manually
docker-compose up --build -d
```

**Production features:**
- Optimized performance settings
- Security hardening
- Health checks enabled
- Automatic restarts

---

## 🌐 Exposing to the Internet

### For Twilio Webhooks

Twilio needs to send webhooks to your server. Here are your options:

#### Option 1: ngrok (Development)

```bash
# Install ngrok
npm install -g ngrok

# Expose port 8000
ngrok http 8000

# Use the ngrok URL in your Twilio webhook configuration
# Example: https://abc123.ngrok.io/api/twilio/webhook
```

#### Option 2: Cloud Deployment (Production)

Deploy to a cloud provider:

- **Railway**: Connect GitHub repo, set environment variables, deploy
- **DigitalOcean**: Use App Platform or Droplet with Docker
- **AWS**: Use ECS or EC2 with Docker
- **Google Cloud**: Use Cloud Run or Compute Engine

#### Option 3: Self-Hosted (Advanced)

```bash
# Use reverse proxy (nginx, Traefik, etc.)
# Configure SSL certificates
# Set up domain name
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Services Won't Start

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs burki
docker-compose logs postgres
```

#### 2. Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose exec postgres pg_isready -U burki_user -d burki

# Reset database (⚠️ deletes all data)
docker-compose down -v
docker-compose up --build -d
```

#### 3. Application Not Responding

```bash
# Check health endpoint
curl http://localhost:8000/health

# Restart application
docker-compose restart burki

# Check resource usage
docker stats
```

#### 4. Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x deploy.sh
```

### Environment Variable Issues

```bash
# Validate environment variables
./deploy.sh --skip-env-check  # Skip validation if needed

# Check what's loaded
docker-compose exec burki printenv | grep -E "(TWILIO|DEEPGRAM|OPENAI|ELEVENLABS)"
```

### Reset Everything

```bash
# Nuclear option - removes all data
docker-compose down -v --remove-orphans
docker system prune -a
./deploy.sh
```

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U burki_user -d burki

# Redis health
docker-compose exec redis redis-cli ping
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f burki

# Last 100 lines
docker-compose logs --tail=100 burki
```

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U burki_user burki > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U burki_user burki < backup.sql
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose up --build -d

# Run any new migrations
docker-compose exec burki alembic upgrade head
```

---

## 🔒 Security Considerations

### Production Security

1. **Change default passwords** in `.env`
2. **Use strong SECRET_KEY** (generate with `openssl rand -hex 32`)
3. **Enable firewall** on your server
4. **Use SSL/TLS** for public deployments
5. **Regularly update** Docker images
6. **Monitor logs** for suspicious activity

### Network Security

```bash
# Only expose necessary ports
# Use Docker networks for internal communication
# Configure firewall rules
```

---

## 🆘 Support

### Getting Help

1. **Check logs** first: `docker-compose logs`
2. **Review this guide** for common solutions
3. **Check GitHub issues** for similar problems
4. **Open a new issue** with:
   - Your Docker/Docker Compose versions
   - Relevant logs
   - Steps to reproduce

### Useful Commands

```bash
# System information
docker version
docker-compose version

# Resource usage
docker stats

# Clean up
docker system prune

# View networks
docker network ls
```

---

## 📈 Performance Tuning

### Resource Allocation

Edit `docker-compose.yml` to adjust resources:

```yaml
services:
  burki:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Scaling

```bash
# Scale application instances
docker-compose up -d --scale burki=3

# Use load balancer (nginx, HAProxy, etc.)
```

---

That's everything you need to know about deploying Burki Voice AI with Docker! 🎉

For more advanced configurations, check out our [Production Deployment Guide](deployment.md). 