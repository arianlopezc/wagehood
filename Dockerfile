# Wagehood Trading System - Production Docker Container
# Multi-stage build for optimized production deployment with security best practices

FROM python:3.9-slim as builder

# Security: Use non-root user for build
RUN groupadd -r builduser && useradd -r -g builduser builduser

# Install system dependencies for building with security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.9-slim

# Security: Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Redis server and runtime dependencies with security updates
RUN apt-get update && apt-get install -y \
    redis-server \
    procps \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security
RUN groupadd -r wagehood && useradd -r -g wagehood wagehood

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app ./

# Copy production service files
COPY start_production_service.py ./
COPY wagehood.service ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R wagehood:wagehood /app

# Copy Docker-specific entrypoint and health check
COPY docker-entrypoint.sh ./
COPY docker-healthcheck.py ./
RUN chmod +x docker-entrypoint.sh docker-healthcheck.py

# Create Redis configuration for container
RUN echo "# Redis configuration for Wagehood container" > /etc/redis/redis.conf && \
    echo "port 6379" >> /etc/redis/redis.conf && \
    echo "bind 0.0.0.0" >> /etc/redis/redis.conf && \
    echo "save 60 100" >> /etc/redis/redis.conf && \
    echo "maxmemory 256mb" >> /etc/redis/redis.conf && \
    echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf && \
    echo "appendonly yes" >> /etc/redis/redis.conf && \
    echo "dir /app/data" >> /etc/redis/redis.conf

# Comprehensive health check with Alpaca connectivity validation
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python docker-healthcheck.py || exit 1

# Switch to non-root user
USER wagehood

# Expose Redis port
EXPOSE 6379

# Environment variables
ENV PYTHONPATH=/app
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379
ENV WAGEHOOD_ENV=production

# Security: Python optimizations and security settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Credential validation: Ensure these are set at runtime
ENV ALPACA_API_KEY_REQUIRED=true
ENV ALPACA_SECRET_KEY_REQUIRED=true

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["production"]