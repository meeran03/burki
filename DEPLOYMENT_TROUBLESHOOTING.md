# Deployment Troubleshooting Guide

## App Runner Build Failures

### Common Issues and Solutions

#### 1. PyAudio Build Failures
**Problem**: `PyAudio==0.2.13` requires system dependencies that aren't available in App Runner build environment.

**Solution**: 
- PyAudio has been removed from `requirements.txt` for production deployments
- Use `requirements-dev.txt` for local development that includes PyAudio
- Audio recording functionality is only needed for local testing, not server deployment

#### 2. Database Connection Issues
**Problem**: App tries to connect to localhost database during build.

**Solution**:
- Database connection is now configured with proper timeouts and connection pooling
- App Runner should have database environment variables configured
- Database connections are lazy-loaded and won't fail during build

#### 3. Missing Environment Variables
**Problem**: App Runner build fails due to missing environment variables.

**Solution**:
Configure these environment variables in App Runner:
```
DB_HOST=your-rds-endpoint
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=your-db-name
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
DEEPGRAM_API_KEY=your-deepgram-key
OPENAI_API_KEY=your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
```

## Pre-deployment Testing

Run the debug script to test your application locally:

```bash
python debug_build.py
```

This will test:
- All critical imports
- Environment variable configuration
- Database connectivity

## App Runner Configuration

The `apprunner.yaml` is configured for:
- Python 3.11 runtime
- Gunicorn with Uvicorn workers
- 4 worker processes
- 120-second timeout
- Port 8000

## Local Development Setup

For local development with audio capabilities:

```bash
# Install system dependencies (macOS)
brew install portaudio

# Install development dependencies
pip install -r requirements-dev.txt
```

## Database Setup

Ensure your RDS/PostgreSQL database is:
1. Accessible from App Runner (security groups)
2. Has the correct database name created
3. User has proper permissions

## Monitoring and Logs

Check App Runner logs for:
- Import errors
- Database connection failures
- Missing environment variables
- Service initialization errors

## Quick Fixes

1. **Build failing on dependencies**: Remove problematic packages from requirements.txt
2. **Database connection timeout**: Check security groups and RDS accessibility
3. **Import errors**: Run `debug_build.py` locally to identify issues
4. **Environment variables**: Verify all required vars are set in App Runner console

## Contact

If issues persist, check:
1. App Runner build logs
2. Application runtime logs
3. Database connectivity from App Runner subnet 