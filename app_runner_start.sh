#!/bin/bash
set -e

echo "Starting Diwaar Application on App Runner..."

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Verify dependencies are installed
echo "Verifying FastAPI installation..."
python3 -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" || {
    echo "FastAPI not found, installing dependencies..."
    pip3 install -r requirements.txt
}

# Run database migrations if alembic is available
if [ -f "alembic.ini" ]; then
    echo "Running database migrations..."
    python3 -m alembic upgrade head || echo "Migration failed, continuing..."
fi

# Start the application with gunicorn
echo "Starting application with gunicorn..."
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WORKERS:-2} \
    --worker-connections ${WORKER_CONNECTIONS:-1000} \
    --max-requests ${MAX_REQUESTS:-1000} \
    --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
    --bind 0.0.0.0:${PORT:-8000} \
    --timeout ${TIMEOUT:-300} \
    --keep-alive ${KEEP_ALIVE:-2} \
    --log-level ${LOG_LEVEL:-info} \
    --access-logfile - \
    --error-logfile - 