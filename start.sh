#!/bin/bash

# Production startup script for Buraaq Voice AI API

echo "Starting Buraaq Voice AI API in production mode..."

# Set environment variables for production
export PYTHONPATH="${PYTHONPATH}:."
export PYTHONUNBUFFERED=1
export ENVIRONMENT=production

# Run database migrations (if needed)
echo "Running database migrations..."
alembic upgrade head

# Start the application with Gunicorn
echo "Starting FastAPI application with Gunicorn..."
exec gunicorn -w 4 \
    -k uvicorn.workers.UvicornWorker \
    app.main:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --preload \
    --access-logfile - \
    --error-logfile - 