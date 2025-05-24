#!/bin/bash
set -e

echo "Starting Buraaq Voice AI Application..."

# Wait for database to be ready
echo "Waiting for database connection..."
until pg_isready -h ${DATABASE_HOST:-localhost} -p ${DATABASE_PORT:-5432} -U ${DATABASE_USER:-postgres}; do
    echo "Database is unavailable - sleeping"
    sleep 2
done
echo "Database is ready!"

# Run database migrations
echo "Running database migrations..."
if [ -f "alembic.ini" ]; then
    alembic upgrade head
    echo "Database migrations completed"
else
    echo "No alembic.ini found, skipping migrations"
fi

# Start the FastAPI application with Gunicorn
echo "Starting FastAPI application..."
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WORKERS:-1} \
    --worker-connections ${WORKER_CONNECTIONS:-1000} \
    --max-requests ${MAX_REQUESTS:-1000} \
    --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
    --preload \
    --bind 0.0.0.0:${PORT:-8000} \
    --timeout ${TIMEOUT:-300} \
    --keep-alive ${KEEP_ALIVE:-2} \
    --log-level ${LOG_LEVEL:-info} \
    --access-logfile - \
    --error-logfile - 