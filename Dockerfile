# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default environment variables (Railway will override these)
ENV PORT=8000
ENV HOST=0.0.0.0
ENV DEBUG=false
ENV SERVER_TYPE=gunicorn
ENV APP_ENV=production

# LLM settings
ENV OPENAI_MODEL=gpt-4-turbo
ENV OPENAI_TEMPERATURE=0.7
ENV OPENAI_MAX_TOKENS=500

# TTS settings
ENV ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Server Performance Settings
ENV WORKERS=2
ENV WORKER_CONNECTIONS=1000
ENV TIMEOUT=300
ENV LOG_LEVEL=info
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=100
ENV KEEP_ALIVE=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    curl \
    wget \
    git \
    autoconf \
    automake \
    libtool \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install RNNoise for audio denoising
RUN mkdir -p /tmp/rnnoise && cd /tmp/rnnoise \
    && git clone https://github.com/xiph/rnnoise.git \
    && cd rnnoise \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cp examples/.libs/rnnoise_demo /usr/local/bin/ \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/rnnoise

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/recordings /app/logs

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Verify RNNoise installation
RUN /app/scripts/verify_rnnoise.sh

# Expose port (Railway will set this dynamically)
EXPOSE $PORT

# Start the application with gunicorn
CMD python3 -m gunicorn app.main:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker --workers $WORKERS --timeout $TIMEOUT --log-level $LOG_LEVEL --max-requests $MAX_REQUESTS --max-requests-jitter $MAX_REQUESTS_JITTER --keep-alive $KEEP_ALIVE 