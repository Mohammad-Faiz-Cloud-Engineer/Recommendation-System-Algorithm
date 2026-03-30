# Deployment Guide

**Copyright © 2026 Mohammad Faiz**  
**Repository:** https://github.com/Mohammad-Faiz-Cloud-Engineer/Recommendation-System-Algorithm

This guide covers production deployment requirements for the recommendation system.

## Required Environment Variables

### Thunder Service (Kafka Consumer)

The Thunder service requires the following environment variables to be set:

```bash
# Kafka Authentication
export KAFKA_SASL_PASSWORD="your-sasl-password"
export KAFKA_PRODUCER_SASL_PASSWORD="your-producer-sasl-password"

# Kafka Topics (if using custom topics)
export KAFKA_TWEET_EVENT_TOPIC="your-tweet-event-topic"
export KAFKA_TWEET_EVENT_DEST="your-tweet-event-destination"
export KAFKA_IN_NETWORK_EVENTS_TOPIC="your-in-network-events-topic"
export KAFKA_IN_NETWORK_EVENTS_DEST="your-in-network-events-destination"
```

### Security Best Practices

1. **Never pass passwords via command-line arguments** - Always use environment variables
2. **Use secret management systems** - Consider using HashiCorp Vault, AWS Secrets Manager, or similar
3. **Rotate credentials regularly** - Implement a credential rotation policy
4. **Limit access** - Use principle of least privilege for service accounts

## Configuration Validation

The services will fail fast at startup if required environment variables are not set. This is intentional to prevent misconfiguration in production.

### Validation Checklist

Before deploying to production:

- [ ] All required environment variables are set
- [ ] Kafka topics exist and are accessible
- [ ] Service accounts have appropriate permissions
- [ ] TLS/SSL certificates are valid and not expired
- [ ] Network connectivity to Kafka brokers is verified
- [ ] Monitoring and alerting are configured

## Local Development

For local development, you can create a `.env.local` file (not committed to git):

```bash
# .env.local (add to .gitignore)
KAFKA_SASL_PASSWORD=dev-password
KAFKA_PRODUCER_SASL_PASSWORD=dev-password
```

Load it before running:

```bash
source .env.local
cargo run --bin thunder
```

## Docker Deployment

When deploying with Docker, pass environment variables securely:

```bash
docker run \
  -e KAFKA_SASL_PASSWORD \
  -e KAFKA_PRODUCER_SASL_PASSWORD \
  your-image:tag
```

Or use Docker secrets:

```yaml
# docker-compose.yml
services:
  thunder:
    image: your-image:tag
    secrets:
      - kafka_sasl_password
      - kafka_producer_sasl_password
    environment:
      KAFKA_SASL_PASSWORD_FILE: /run/secrets/kafka_sasl_password
      KAFKA_PRODUCER_SASL_PASSWORD_FILE: /run/secrets/kafka_producer_sasl_password

secrets:
  kafka_sasl_password:
    external: true
  kafka_producer_sasl_password:
    external: true
```

## Kubernetes Deployment

Use Kubernetes secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: kafka-credentials
type: Opaque
stringData:
  sasl-password: your-sasl-password
  producer-sasl-password: your-producer-sasl-password
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thunder
spec:
  template:
    spec:
      containers:
      - name: thunder
        env:
        - name: KAFKA_SASL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kafka-credentials
              key: sasl-password
        - name: KAFKA_PRODUCER_SASL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kafka-credentials
              key: producer-sasl-password
```

## Health Checks

The services expose health check endpoints:

- Thunder: `http://localhost:{http_port}/health`
- Home Mixer: `http://localhost:{metrics_port}/health`

Configure your orchestration platform to use these for liveness and readiness probes.

## Monitoring

Key metrics to monitor:

- Kafka consumer lag
- Request latency (p50, p95, p99)
- Error rates
- Memory and CPU usage
- Post store size and growth rate

All services expose Prometheus metrics on their metrics port.

## Troubleshooting

### Service fails to start with "Required environment variable not set"

**Solution:** Ensure all required environment variables are set. Check the error message for the specific variable name.

### Kafka connection errors

**Solution:** 
1. Verify network connectivity to Kafka brokers
2. Check that credentials are correct
3. Ensure topics exist and service account has permissions
4. Verify SSL/TLS configuration if using encrypted connections

### High memory usage in Thunder

**Solution:**
1. Check post retention settings (`--post-retention-seconds`)
2. Monitor the number of users and posts in the store
3. Consider scaling horizontally if needed
4. Review auto-trim interval (`--lag-monitor-interval-secs`)

## Performance Tuning

### Thunder Service

- `--kafka-num-threads`: Number of Kafka consumer threads (default: based on partitions)
- `--kafka-batch-size`: Batch size for processing (default: 1000)
- `--post-retention-seconds`: How long to keep posts in memory (default: 2 days)
- `--max-concurrent-requests`: Limit concurrent gRPC requests (default: 1000)

### Home Mixer

- Adjust `MAX_GRPC_MESSAGE_SIZE` in params if handling large requests
- Configure connection pooling for downstream services
- Tune timeout values based on SLA requirements

## License

This deployment guide is part of the Algorithm repository.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
