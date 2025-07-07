# Wagehood Trading System - Docker Production Setup Guide

## Overview

This guide provides comprehensive instructions for deploying the Wagehood Trading System in a production Docker environment with proper Alpaca credentials validation and security best practices.

## Prerequisites

### Required Software
- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+
- Git (for cloning the repository)

### Required Credentials
- **Alpaca Markets API Key and Secret Key**
  - Get your free paper trading credentials from: https://app.alpaca.markets/
  - For live trading, upgrade to a funded account

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/wagehood.git
cd wagehood
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Environment Variables:**
```bash
# Alpaca Markets API Configuration (MANDATORY)
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# Trading Configuration
ALPACA_PAPER_TRADING=true
ALPACA_DATA_FEED=iex

# System Configuration
WAGEHOOD_ENV=production
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Build and Run

**Option A: Using Docker Compose (Recommended)**
```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

**Option B: Using Docker Run**
```bash
# Build the image
docker build -t wagehood:latest .

# Run with environment variables
docker run -d \
  --name wagehood-trading \
  -e ALPACA_API_KEY=your_key_here \
  -e ALPACA_SECRET_KEY=your_secret_here \
  -e ALPACA_PAPER_TRADING=true \
  -e ALPACA_DATA_FEED=iex \
  -p 6379:6379 \
  -v wagehood-data:/app/data \
  -v wagehood-logs:/app/logs \
  wagehood:latest
```

## Production Security Configuration

### 1. Credential Management

**DO NOT** hard-code credentials in Dockerfiles or docker-compose files.

**Recommended Methods:**

**Environment File (.env):**
```bash
# Create .env file (never commit to git)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

**Docker Secrets (Swarm Mode):**
```bash
# Create secrets
echo "your_api_key" | docker secret create alpaca_api_key -
echo "your_secret_key" | docker secret create alpaca_secret_key -

# Use in docker-compose.yml
services:
  wagehood:
    secrets:
      - alpaca_api_key
      - alpaca_secret_key
```

**External Secret Management:**
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault

### 2. Network Security

**Restrict Port Access:**
```yaml
# docker-compose.yml
services:
  wagehood:
    ports:
      - "127.0.0.1:6379:6379"  # Bind to localhost only
```

**Use Custom Networks:**
```yaml
networks:
  wagehood-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. Resource Limits

```yaml
# docker-compose.yml
services:
  wagehood:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

## Health Monitoring

### Built-in Health Check

The container includes a comprehensive health check that validates:
- Redis connectivity
- Alpaca API credentials
- Market data retrieval
- Core system functionality

**Check Health Status:**
```bash
# View health status
docker inspect wagehood-trading | grep -A 20 "Health"

# View health check logs
docker logs wagehood-trading 2>&1 | grep "HEALTH CHECK"
```

### External Monitoring

**Docker Compose with Monitoring:**
```yaml
services:
  wagehood:
    # ... existing configuration ...
    
  redis-insight:
    image: redislabs/redis-insight:latest
    ports:
      - "8001:8001"
    volumes:
      - redis-insight-data:/db
```

## Data Persistence

### Volume Configuration

**Important Volumes:**
- `/app/data` - Market data cache and system state
- `/app/logs` - Application logs
- `/app/config` - Custom configuration (optional)

**Docker Compose Example:**
```yaml
volumes:
  wagehood-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/wagehood/data
      
  wagehood-logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/wagehood/logs
```

## Troubleshooting

### Common Issues

**1. Credential Validation Failures**
```bash
# Check environment variables
docker exec wagehood-trading env | grep ALPACA

# Validate credentials manually
docker exec wagehood-trading python docker-healthcheck.py
```

**2. Redis Connection Issues**
```bash
# Check Redis status
docker exec wagehood-trading redis-cli ping

# Check Redis logs
docker exec wagehood-trading redis-cli monitor
```

**3. Market Data Issues**
```bash
# Test Alpaca connectivity
docker exec wagehood-trading python -c "
import asyncio
import os
from src.realtime.data_ingestion import MinimalAlpacaProvider

async def test():
    config = {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'paper': True,
        'feed': 'iex'
    }
    provider = MinimalAlpacaProvider(config)
    await provider.connect()
    print('Alpaca connection successful')

asyncio.run(test())
"
```

### Debug Mode

**Run in Debug Mode:**
```bash
# Start container in shell mode
docker run -it --rm \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  wagehood:latest shell

# Or exec into running container
docker exec -it wagehood-trading /bin/bash
```

### Log Analysis

**View Logs:**
```bash
# Application logs
docker exec wagehood-trading tail -f /app/logs/wagehood_production.log

# System logs
docker logs wagehood-trading

# Redis logs
docker exec wagehood-trading redis-cli --latency
```

## Production Deployment Checklist

### Pre-deployment
- [ ] Alpaca credentials configured and validated
- [ ] `.env` file created with production settings
- [ ] Volume mounts configured for data persistence
- [ ] Network security configured
- [ ] Resource limits set appropriately
- [ ] Health check endpoints accessible
- [ ] Monitoring configured

### Post-deployment
- [ ] Container starts successfully
- [ ] Health checks pass
- [ ] Market data is being received
- [ ] Redis is operational
- [ ] Logs are being written
- [ ] Performance metrics are normal
- [ ] Alerts configured

## Advanced Configuration

### Custom Configuration

**Mount custom config:**
```yaml
volumes:
  - ./custom-config.yaml:/app/config/config.yaml:ro
```

### Scaling

**Horizontal Scaling:**
```yaml
services:
  wagehood:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
```

### SSL/TLS

**With Reverse Proxy:**
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

## Support and Maintenance

### Regular Maintenance

**Weekly Tasks:**
- Check disk space usage
- Review application logs
- Verify market data accuracy
- Test health checks

**Monthly Tasks:**
- Update Docker images
- Review security configurations
- Backup configuration files
- Performance optimization

### Support

For issues and support:
1. Check the logs first
2. Run the health check script
3. Review this documentation
4. Create an issue on GitHub

## Security Best Practices Summary

1. **Never commit credentials to version control**
2. **Use environment variables or secrets management**
3. **Run containers as non-root users**
4. **Keep Docker images updated**
5. **Restrict network access**
6. **Monitor logs for security events**
7. **Use resource limits**
8. **Implement proper backup strategies**

---

**Note:** This system handles real financial data and trading operations. Always test thoroughly in paper trading mode before using with real funds.