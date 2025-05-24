# AWS App Runner Deployment Guide

This guide will help you deploy the Diwaar Voice AI application to AWS App Runner.

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured
3. GitHub repository with your code
4. PostgreSQL database (AWS RDS recommended)

## Pre-deployment Setup

### 1. Database Setup (AWS RDS)

Create a PostgreSQL database on AWS RDS:

```bash
# Example AWS CLI command (adjust parameters as needed)
aws rds create-db-instance \
    --db-instance-identifier diwaar-prod-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 14.9 \
    --master-username your_db_user \
    --master-user-password your_secure_password \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-xxxxxxxxx \
    --db-subnet-group-name your-subnet-group
```

### 2. Environment Variables and Secrets

Set up the following environment variables in AWS App Runner:

#### Runtime Environment Variables
```
PORT=8000
APP_ENV=production
DEBUG=false
WORKERS=2
WORKER_CONNECTIONS=1000
TIMEOUT=300
LOG_LEVEL=info
```

#### Secrets (Configure in AWS App Runner Console)
- `DATABASE_URL` - Your PostgreSQL connection string
- `TWILIO_ACCOUNT_SID` - Your Twilio Account SID
- `TWILIO_AUTH_TOKEN` - Your Twilio Auth Token
- `TWILIO_PHONE_NUMBER` - Your Twilio Phone Number
- `DEEPGRAM_API_KEY` - Your Deepgram API Key
- `OPENAI_API_KEY` - Your OpenAI API Key
- `ELEVENLABS_API_KEY` - Your ElevenLabs API Key
- `SECRET_KEY` - Generate a strong secret key
- `JWT_SECRET_KEY` - Generate a strong JWT secret
- `SESSION_SECRET` - Generate a strong session secret
- `GOOGLE_CLIENT_ID` - Your Google OAuth Client ID
- `GOOGLE_CLIENT_SECRET` - Your Google OAuth Client Secret

### 3. Generate Production Secrets

Use these commands to generate secure keys:

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate SESSION_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Deployment Steps

### Option 1: Deploy via AWS Console

1. **Open AWS App Runner Console**
   - Go to https://console.aws.amazon.com/apprunner/

2. **Create a new service**
   - Click "Create service"
   - Choose "Source code repository"
   - Connect your GitHub repository

3. **Configure deployment**
   - Repository: Select your repository
   - Branch: `main` or your production branch
   - Configuration: Use configuration file (`apprunner.yaml`)

4. **Configure service**
   - Service name: `diwaar-voice-ai`
   - Virtual CPU: 1 vCPU
   - Virtual memory: 2 GB
   - Environment variables: Add all the variables listed above

5. **Configure security**
   - Add your secrets as environment variables

6. **Review and create**
   - Review all settings and create the service

### Option 2: Deploy via AWS CLI

1. **Create App Runner service configuration**

```json
{
  "ServiceName": "diwaar-voice-ai",
  "SourceConfiguration": {
    "CodeRepository": {
      "RepositoryUrl": "https://github.com/your-username/your-repo",
      "SourceCodeVersion": {
        "Type": "BRANCH",
        "Value": "main"
      },
      "CodeConfiguration": {
        "ConfigurationSource": "REPOSITORY"
      }
    },
    "AutoDeploymentsEnabled": true
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB",
    "InstanceRoleArn": "arn:aws:iam::your-account:role/AppRunnerInstanceRole"
  }
}
```

2. **Deploy using AWS CLI**

```bash
aws apprunner create-service --cli-input-json file://apprunner-config.json
```

## Post-Deployment Configuration

### 1. Database Migrations

The application will automatically run database migrations on startup. Monitor the logs to ensure they complete successfully.

### 2. Health Check

Verify the health endpoint is working:
```bash
curl https://your-app-runner-url.region.awsapprunner.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "buraaq-voice-ai"
}
```

### 3. Update OAuth Redirect URLs

Update your Google OAuth configuration with the new App Runner URL:
- Authorized redirect URIs: `https://your-app-runner-url.region.awsapprunner.com/auth/google/callback`

### 4. Configure CORS

Update the `CORS_ORIGINS` environment variable with your frontend domain.

## Monitoring and Logs

### CloudWatch Integration

AWS App Runner automatically integrates with CloudWatch. You can view logs and metrics in the AWS Console:

1. Go to CloudWatch Console
2. Navigate to Logs > Log groups
3. Find `/aws/apprunner/diwaar-voice-ai/application`

### Custom Metrics

The application includes structured logging. Monitor these key metrics:
- Application startup time
- Database connection status
- API response times
- Error rates

## Scaling Configuration

App Runner will automatically scale based on traffic. To configure scaling:

1. **Minimum instances**: 1
2. **Maximum instances**: 10 (adjust based on your needs)
3. **Concurrency**: 100 (requests per instance)

## Security Best Practices

1. **Use AWS Secrets Manager** for sensitive data
2. **Enable VPC integration** if connecting to private resources
3. **Use least privilege IAM roles**
4. **Enable CloudTrail** for audit logging
5. **Regular security updates** via automated deployments

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify RDS security group allows connections from App Runner
   - Check DATABASE_URL format
   - Ensure RDS is in the same region

2. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt
   - Check Docker build logs

3. **Runtime Errors**
   - Check CloudWatch logs
   - Verify all environment variables are set
   - Test health endpoint

### Useful Commands

```bash
# Check service status
aws apprunner describe-service --service-arn your-service-arn

# View recent deployments
aws apprunner list-operations --service-arn your-service-arn

# Trigger manual deployment
aws apprunner start-deployment --service-arn your-service-arn
```

## Cost Optimization

1. **Right-size your instances** based on actual usage
2. **Use appropriate scaling settings**
3. **Monitor CloudWatch metrics** for optimization opportunities
4. **Consider using Reserved Capacity** for predictable workloads

## Support

For deployment issues:
1. Check CloudWatch logs first
2. Verify all configuration settings
3. Test individual components (database, APIs)
4. Contact AWS Support if needed

---

**Note**: Replace all placeholder values (like `your-username`, `your-repo`, etc.) with your actual values before deployment. 