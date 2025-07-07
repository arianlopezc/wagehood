# Wagehood Docker Deployment Guide

This guide covers deploying the Wagehood Trading System using Docker containers for production use.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- 2GB+ RAM available
- 1GB+ disk space
- **MANDATORY**: Valid Alpaca Markets account and API credentials

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/wagehood.git
cd wagehood

# REQUIRED: Set your Alpaca credentials
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"

# Build and start the service
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f wagehood
```

### 2. Build and Run with Docker CLI

```bash
# Build the image
docker build -t wagehood:latest .

# REQUIRED: Run with Alpaca credentials
docker run -d \
  --name wagehood-trading \
  --restart unless-stopped \
  -p 6379:6379 \
  -e ALPACA_API_KEY="your_alpaca_api_key" \
  -e ALPACA_SECRET_KEY="your_alpaca_secret_key" \
  -e ALPACA_PAPER_TRADING="true" \
  -v wagehood-data:/app/data \
  -v wagehood-logs:/app/logs \
  wagehood:latest
```

## 📋 Container Commands

The Docker container supports multiple run modes:

### Production Mode (Default)
```bash
docker run -d wagehood:latest production
```

### Test Mode
```bash
docker run --rm wagehood:latest test
```

### Interactive Shell
```bash
docker run -it --rm wagehood:latest shell
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WAGEHOOD_ENV` | Environment mode | `production` |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `ALPACA_API_KEY` | Alpaca API key | **REQUIRED** |
| `ALPACA_SECRET_KEY` | Alpaca secret key | **REQUIRED** |

### Docker Compose Environment

Create a `.env` file in the project root:

```env
# Alpaca Trading Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Optional: Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional: Environment
WAGEHOOD_ENV=production
```

## 📊 Monitoring

### Container Health Check

```bash
# Check container health
docker inspect wagehood-trading | grep -A 5 Health

# Manual health check
docker exec wagehood-trading python -c "from src.storage.cache import cache_manager; cache_manager.ping()"
```

### Log Monitoring

```bash
# View real-time logs
docker logs -f wagehood-trading

# View specific log files
docker exec wagehood-trading tail -f /app/logs/wagehood_production.log
```

### Redis Monitoring

```bash
# Check Redis stats
docker exec wagehood-trading redis-cli INFO stats

# Monitor Redis streams
docker exec wagehood-trading redis-cli XLEN market_data_stream
```

## 🔒 Security

### Production Security Checklist

- [ ] Use non-root user (already configured)
- [ ] Set resource limits in docker-compose.yml
- [ ] Use secrets management for API keys
- [ ] Enable Docker content trust
- [ ] Use specific image tags (not :latest)
- [ ] Configure log rotation
- [ ] Set up network policies
- [ ] Enable Docker security scanning

### Resource Limits

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 1G
    reservations:
      cpus: '0.5'
      memory: 512M
```

## 🗄️ Data Persistence

### Volume Configuration

```bash
# Create named volumes
docker volume create wagehood-data
docker volume create wagehood-logs

# Backup volumes
docker run --rm -v wagehood-data:/data -v $(pwd):/backup alpine tar czf /backup/wagehood-data.tar.gz /data
```

### Data Directories

- `/app/data` - Redis persistence and configuration
- `/app/logs` - Application logs
- `/app/config` - Configuration files (optional mount)

## 🚀 Production Deployment

### 1. Docker Swarm (Recommended)

```bash
# Initialize swarm
docker swarm init

# Create secrets
echo "your_api_key" | docker secret create alpaca_api_key -
echo "your_secret_key" | docker secret create alpaca_secret_key -

# Deploy stack
docker stack deploy -c docker-compose.yml wagehood
```

### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wagehood-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: wagehood
  template:
    metadata:
      labels:
        app: wagehood
    spec:
      containers:
      - name: wagehood
        image: wagehood:latest
        ports:
        - containerPort: 6379
        env:
        - name: ALPACA_API_KEY
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: api-key
        - name: ALPACA_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: wagehood-data
      - name: logs
        persistentVolumeClaim:
          claimName: wagehood-logs
```

### 3. Cloud Deployment

#### AWS ECS
```bash
# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create ECS service
aws ecs create-service --cluster default --service-name wagehood --task-definition wagehood:1 --desired-count 1
```

#### Google Cloud Run
```bash
# Build and push to GCR
docker build -t gcr.io/PROJECT_ID/wagehood:latest .
docker push gcr.io/PROJECT_ID/wagehood:latest

# Deploy to Cloud Run
gcloud run deploy wagehood \
  --image gcr.io/PROJECT_ID/wagehood:latest \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars WAGEHOOD_ENV=production
```

## 🔧 Troubleshooting

### Common Issues

1. **Container fails to start**
   ```bash
   docker logs wagehood-trading
   # Check for port conflicts, permission issues, or missing dependencies
   ```

2. **Redis connection issues**
   ```bash
   docker exec wagehood-trading redis-cli ping
   # Should return PONG
   ```

3. **Service not processing data**
   ```bash
   docker exec wagehood-trading python -c "from src.storage.cache import cache_manager; print(cache_manager.keys())"
   ```

4. **High memory usage**
   ```bash
   docker stats wagehood-trading
   # Check memory consumption and Redis memory usage
   ```

### Debug Mode

```bash
# Run container in debug mode
docker run -it --rm wagehood:latest shell

# Inside container
python -c "from src.realtime.data_ingestion import create_ingestion_service; print('OK')"
```

## 📈 Performance Optimization

### Memory Optimization

```yaml
# In docker-compose.yml
environment:
  - REDIS_MAXMEMORY=256mb
  - REDIS_MAXMEMORY_POLICY=allkeys-lru
```

### CPU Optimization

```yaml
# Multi-stage build optimization
deploy:
  resources:
    limits:
      cpus: '1.0'
  placement:
    constraints:
      - node.role == worker
```

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Build new version
docker build -t wagehood:v2.0.0 .

# Update with zero downtime
docker-compose up -d --no-deps wagehood

# Rollback if needed
docker-compose up -d --no-deps wagehood:v1.0.0
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec wagehood-trading redis-cli BGSAVE
docker run --rm -v wagehood-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/wagehood-$DATE.tar.gz /data
```

## 📞 Support

For Docker-specific issues:
- Check container logs: `docker logs wagehood-trading`
- Verify resource usage: `docker stats wagehood-trading`
- Test connectivity: `docker exec wagehood-trading redis-cli ping`

For application issues, refer to the main README.md file.