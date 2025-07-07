#!/bin/bash

# =============================================================================
# Burki Voice AI - One-Click Docker Deployment Script
# =============================================================================
# This script automates the deployment of Burki Voice AI using Docker Compose
# Usage: ./deploy.sh [--dev|--prod] [--skip-env-check]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="prod"
SKIP_ENV_CHECK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            ENVIRONMENT="dev"
            shift
            ;;
        --prod)
            ENVIRONMENT="prod"
            shift
            ;;
        --skip-env-check)
            SKIP_ENV_CHECK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dev|--prod] [--skip-env-check]"
            echo "  --dev: Deploy in development mode"
            echo "  --prod: Deploy in production mode (default)"
            echo "  --skip-env-check: Skip environment variable validation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}🚀 Burki Voice AI Deployment${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo ""

# Check if Docker and Docker Compose are installed
check_dependencies() {
    echo -e "${BLUE}📋 Checking dependencies...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
}

# Check if .env file exists and validate required variables
check_environment() {
    if [[ "$SKIP_ENV_CHECK" == true ]]; then
        echo -e "${YELLOW}⚠️  Skipping environment variable validation${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking environment configuration...${NC}"
    
    if [[ ! -f .env ]]; then
        echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
        if [[ -f .env.example ]]; then
            cp .env.example .env
            echo -e "${RED}❌ Please edit .env file with your configuration before running again${NC}"
            exit 1
        else
            echo -e "${RED}❌ .env.example file not found. Please create .env file manually${NC}"
            exit 1
        fi
    fi
    
    # Check required variables
    REQUIRED_VARS=(
        "TWILIO_ACCOUNT_SID"
        "TWILIO_AUTH_TOKEN"
        "TWILIO_PHONE_NUMBER"
        "DEEPGRAM_API_KEY"
        "OPENAI_API_KEY"
        "ELEVENLABS_API_KEY"
        "SECRET_KEY"
    )
    
    missing_vars=()
    for var in "${REQUIRED_VARS[@]}"; do
        if ! grep -q "^$var=" .env || grep -q "^$var=your_" .env || grep -q "^$var=$" .env; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo -e "${RED}❌ Missing or invalid required environment variables:${NC}"
        for var in "${missing_vars[@]}"; do
            echo -e "${RED}   - $var${NC}"
        done
        echo -e "${YELLOW}Please update your .env file with valid values${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Environment configuration looks good${NC}"
}

# Build and start services
deploy_services() {
    echo -e "${BLUE}🏗️  Building and starting services...${NC}"
    
    # Stop any existing services
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build and start services
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        docker-compose up --build -d
    else
        docker-compose up --build -d --remove-orphans
    fi
    
    echo -e "${GREEN}✅ Services started successfully${NC}"
}

# Wait for services to be healthy
wait_for_services() {
    echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
    
    # Wait for database
    echo -e "${YELLOW}Waiting for PostgreSQL...${NC}"
    timeout=60
    while ! docker-compose exec -T postgres pg_isready -U burki_user -d burki 2>/dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            echo -e "${RED}❌ PostgreSQL failed to start within 60 seconds${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
    
    # Wait for Redis
    echo -e "${YELLOW}Waiting for Redis...${NC}"
    timeout=30
    while ! docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            echo -e "${RED}❌ Redis failed to start within 30 seconds${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}✅ Redis is ready${NC}"
    
    # Wait for application
    echo -e "${YELLOW}Waiting for Burki Voice AI...${NC}"
    timeout=120
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
        timeout=$((timeout - 5))
        if [[ $timeout -le 0 ]]; then
            echo -e "${RED}❌ Burki Voice AI failed to start within 120 seconds${NC}"
            echo -e "${YELLOW}Check logs with: docker-compose logs burki${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}✅ Burki Voice AI is ready${NC}"
}

# Run database migrations
run_migrations() {
    echo -e "${BLUE}🗄️  Running database migrations...${NC}"
    
    # Run Alembic migrations
    if docker-compose exec -T burki alembic upgrade head; then
        echo -e "${GREEN}✅ Database migrations completed${NC}"
    else
        echo -e "${RED}❌ Database migrations failed${NC}"
        exit 1
    fi
}

# Display success message and next steps
show_success() {
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo -e "${BLUE}📱 Your Burki Voice AI is now running:${NC}"
    echo -e "   🌐 Web Dashboard: ${YELLOW}http://localhost:8000${NC}"
    echo -e "   📚 API Documentation: ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "   💚 Health Check: ${YELLOW}http://localhost:8000/health${NC}"
    echo ""
    echo -e "${BLUE}🔧 Useful commands:${NC}"
    echo -e "   View logs: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "   Stop services: ${YELLOW}docker-compose down${NC}"
    echo -e "   Restart services: ${YELLOW}docker-compose restart${NC}"
    echo -e "   View service status: ${YELLOW}docker-compose ps${NC}"
    echo ""
    echo -e "${BLUE}📞 Next steps:${NC}"
    echo -e "   1. Configure your Twilio webhook to point to your server"
    echo -e "   2. Create your first assistant in the web dashboard"
    echo -e "   3. Test your voice AI by calling your Twilio number"
    echo ""
}

# Main deployment flow
main() {
    check_dependencies
    check_environment
    deploy_services
    wait_for_services
    run_migrations
    show_success
}

# Run main function
main "$@" 