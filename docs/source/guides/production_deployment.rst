Production Deployment
=====================

Deploy LRS-Agents to production with monitoring, logging, and high availability.

Overview
--------

This guide covers:

* Production-ready architecture
* Monitoring and alerting
* Structured logging
* Scaling strategies
* High availability
* Security best practices

Architecture
------------

Recommended Stack
^^^^^^^^^^^^^^^^^

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │         Load Balancer (Nginx)           │
   └─────────────────┬───────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
   ┌─────▼──────┐         ┌─────▼──────┐
   │  LRS API   │         │  LRS API   │  (Multiple instances)
   │  Instance  │         │  Instance  │
   └─────┬──────┘         └─────┬──────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
   ┌─────▼──────┐         ┌─────▼──────┐
   │ PostgreSQL │         │   Redis    │
   │  Database  │         │   Cache    │
   └────────────┘         └────────────┘

Components:

* **Load Balancer**: Distributes traffic across instances
* **LRS API Instances**: Stateless agent execution servers
* **PostgreSQL**: Persistent storage for execution history
* **Redis**: Caching and job queue

Docker Deployment
-----------------

Basic Docker Setup
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Build image
   docker build -t lrs-agents:latest -f docker/Dockerfile .

   # Run single container
   docker run -d \
     -p 8000:8000 \
     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
     -e DATABASE_URL=$DATABASE_URL \
     lrs-agents:latest

Docker Compose
^^^^^^^^^^^^^^

Use Docker Compose for local development and testing:

.. code-block:: bash

   cd docker
   docker-compose up -d

   # Services available:
   # - API: http://localhost:8000
   # - Dashboard: http://localhost:8501
   # - Database: localhost:5432

Production Docker Compose:

.. code-block:: yaml

   # docker-compose.prod.yml
   version: '3.8'

   services:
     lrs-api:
       image: lrsagents/lrs-agents:latest
       deploy:
         replicas: 3
         resources:
           limits:
             cpus: '2'
             memory: 4G
           reservations:
             cpus: '1'
             memory: 2G
       environment:
         - DATABASE_URL=postgresql://user:pass@postgres:5432/lrs
         - REDIS_URL=redis://redis:6379/0
         - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
         - LOG_LEVEL=INFO
       depends_on:
         - postgres
         - redis

     postgres:
       image: postgres:15-alpine
       volumes:
         - postgres_data:/var/lib/postgresql/data
       environment:
         POSTGRES_PASSWORD: ${DB_PASSWORD}

     redis:
       image: redis:7-alpine
       volumes:
         - redis_data:/var/lib/redis

     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - lrs-api

   volumes:
     postgres_data:
     redis_data:

Kubernetes Deployment
---------------------

Basic Deployment
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create namespace
   kubectl create namespace lrs-agents

   # Apply configurations
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/hpa.yaml

Verify deployment:

.. code-block:: bash

   # Check pods
   kubectl get pods -n lrs-agents

   # Check services
   kubectl get svc -n lrs-agents

   # View logs
   kubectl logs -f deployment/lrs-agents -n lrs-agents

Production Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # k8s/deployment-prod.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: lrs-agents
     namespace: lrs-agents
   spec:
     replicas: 5  # Start with 5 replicas
     selector:
       matchLabels:
         app: lrs-agents
     template:
       metadata:
         labels:
           app: lrs-agents
       spec:
         containers:
         - name: lrs-api
           image: lrsagents/lrs-agents:v0.2.0  # Pin version
           resources:
             requests:
               memory: "2Gi"
               cpu: "500m"
             limits:
               memory: "4Gi"
               cpu: "2000m"
           env:
           - name: DATABASE_URL
             valueFrom:
               secretKeyRef:
                 name: lrs-secrets
                 key: database-url
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 10
             periodSeconds: 5

Auto-scaling:

.. code-block:: yaml

   # k8s/hpa-prod.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: lrs-agents-hpa
     namespace: lrs-agents
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: lrs-agents
     minReplicas: 5
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80

Monitoring
----------

Structured Logging
^^^^^^^^^^^^^^^^^^

Set up structured logging for production:

.. code-block:: python

   from lrs.monitoring.structured_logging import create_logger_for_agent
   import logging

   # Create logger
   logger = create_logger_for_agent(
       agent_id="production_agent",
       log_file="/var/log/lrs/agent.jsonl",
       console=False,  # Disable console in production
       level=logging.INFO
   )

   # Log events
   logger.log_tool_execution(
       tool_name="fetch_api",
       success=True,
       execution_time=150.5,
       prediction_error=0.1
   )

   logger.log_adaptation_event(
       trigger="High prediction error",
       old_precision=0.6,
       new_precision=0.4,
       action="Explore alternatives"
   )

Log Aggregation
^^^^^^^^^^^^^^^

Send logs to centralized system:

**ELK Stack (Elasticsearch, Logstash, Kibana):**

.. code-block:: yaml

   # filebeat.yml
   filebeat.inputs:
   - type: log
     enabled: true
     paths:
       - /var/log/lrs/*.jsonl
     json.keys_under_root: true
     json.add_error_key: true

   output.elasticsearch:
     hosts: ["elasticsearch:9200"]

**Datadog:**

.. code-block:: python

   from datadog import initialize, statsd

   # Initialize Datadog
   initialize(api_key=os.getenv('DATADOG_API_KEY'))

   # Send metrics
   statsd.increment('lrs.agent.execution')
   statsd.histogram('lrs.precision', precision_value)
   statsd.gauge('lrs.tool.success_rate', success_rate)

Metrics and Alerting
^^^^^^^^^^^^^^^^^^^^

Expose Prometheus metrics:

.. code-block:: python

   from prometheus_client import Counter, Histogram, Gauge, start_http_server

   # Define metrics
   agent_runs = Counter('lrs_agent_runs_total', 'Total agent runs')
   tool_executions = Counter('lrs_tool_executions_total', 'Total tool executions', ['tool', 'status'])
   precision_value = Gauge('lrs_precision_value', 'Current precision', ['level'])
   execution_time = Histogram('lrs_execution_time_seconds', 'Execution time')

   # Record metrics
   agent_runs.inc()
   tool_executions.labels(tool='fetch_api', status='success').inc()
   precision_value.labels(level='execution').set(0.75)
   
   with execution_time.time():
       result = agent.run(task)

   # Start metrics server
   start_http_server(9090)

Prometheus configuration:

.. code-block:: yaml

   # prometheus.yml
   scrape_configs:
     - job_name: 'lrs-agents'
       static_configs:
         - targets: ['lrs-api:9090']
       scrape_interval: 15s

Alerting rules:

.. code-block:: yaml

   # alerts.yml
   groups:
   - name: lrs_agents
     rules:
     - alert: HighFailureRate
       expr: rate(lrs_tool_executions_total{status="failure"}[5m]) > 0.5
       for: 5m
       labels:
         severity: warning
       annotations:
         summary: "High tool failure rate"

     - alert: LowPrecision
       expr: lrs_precision_value{level="execution"} < 0.3
       for: 10m
       labels:
         severity: warning
       annotations:
         summary: "Agent precision consistently low"

     - alert: ServiceDown
       expr: up{job="lrs-agents"} == 0
       for: 2m
       labels:
         severity: critical
       annotations:
         summary: "LRS-Agents service is down"

Dashboard
^^^^^^^^^

Run Streamlit dashboard for real-time monitoring:

.. code-block:: bash

   # In separate container/pod
   streamlit run lrs/monitoring/dashboard.py --server.port=8501

Grafana dashboards:

.. code-block:: json

   {
     "dashboard": {
       "title": "LRS-Agents Monitoring",
       "panels": [
         {
           "title": "Precision Over Time",
           "targets": [
             {
               "expr": "lrs_precision_value{level=\"execution\"}"
             }
           ]
         },
         {
           "title": "Tool Success Rate",
           "targets": [
             {
               "expr": "rate(lrs_tool_executions_total{status=\"success\"}[5m]) / rate(lrs_tool_executions_total[5m])"
             }
           ]
         },
         {
           "title": "Adaptation Events",
           "targets": [
             {
               "expr": "rate(lrs_adaptation_events_total[5m])"
             }
           ]
         }
       ]
     }
   }

Database Management
-------------------

Schema Setup
^^^^^^^^^^^^

Initialize production database:

.. code-block:: bash

   # Run migrations
   psql $DATABASE_URL < docker/init.sql

   # Or use migration tool
   alembic upgrade head

Connection Pooling
^^^^^^^^^^^^^^^^^^

Configure connection pooling:

.. code-block:: python

   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool

   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=20,          # Connections per instance
       max_overflow=10,       # Additional connections
       pool_timeout=30,       # Wait timeout
       pool_recycle=3600,     # Recycle connections after 1 hour
       pool_pre_ping=True     # Verify connections before use
   )

Backup Strategy
^^^^^^^^^^^^^^^

Automated backups:

.. code-block:: bash

   #!/bin/bash
   # backup.sh

   DATE=$(date +%Y%m%d_%H%M%S)
   BACKUP_FILE="lrs_backup_$DATE.sql"

   # Create backup
   pg_dump $DATABASE_URL > $BACKUP_FILE

   # Compress
   gzip $BACKUP_FILE

   # Upload to S3
   aws s3 cp $BACKUP_FILE.gz s3://lrs-backups/

   # Cleanup old backups (keep last 30 days)
   find . -name "lrs_backup_*.sql.gz" -mtime +30 -delete

Schedule with cron:

.. code-block:: cron

   # Daily backups at 2 AM
   0 2 * * * /path/to/backup.sh

Security
--------

API Authentication
^^^^^^^^^^^^^^^^^^

Implement JWT authentication:

.. code-block:: python

   from fastapi import Depends, HTTPException
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   import jwt

   security = HTTPBearer()

   def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
       try:
           payload = jwt.decode(
               credentials.credentials,
               SECRET_KEY,
               algorithms=["HS256"]
           )
           return payload
       except jwt.InvalidTokenError:
           raise HTTPException(status_code=401, detail="Invalid token")

   @app.post("/api/agent/run")
   async def run_agent(task: str, token: dict = Depends(verify_token)):
       # Execute agent
       pass

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Securely manage secrets:

.. code-block:: bash

   # Never commit secrets to version control
   # Use environment variables or secret management

   # Development
   export ANTHROPIC_API_KEY="sk-ant-..."

   # Production - Use secret management
   # AWS Secrets Manager
   aws secretsmanager get-secret-value --secret-id lrs/api-keys

   # Kubernetes Secrets
   kubectl create secret generic lrs-secrets \
     --from-literal=anthropic-api-key=sk-ant-...

Rate Limiting
^^^^^^^^^^^^^

Implement rate limiting:

.. code-block:: python

   from fastapi import Request
   from slowapi import Limiter
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)

   @app.post("/api/agent/run")
   @limiter.limit("10/minute")  # 10 requests per minute
   async def run_agent(request: Request, task: str):
       # Execute agent
       pass

Performance Optimization
------------------------

Caching
^^^^^^^

Implement Redis caching:

.. code-block:: python

   import redis
   import hashlib
   import json

   redis_client = redis.Redis(host='redis', port=6379, db=0)

   def cache_agent_result(task: str, result: dict, ttl: int = 3600):
       """Cache agent execution result"""
       cache_key = hashlib.md5(task.encode()).hexdigest()
       redis_client.setex(cache_key, ttl, json.dumps(result))

   def get_cached_result(task: str):
       """Get cached result if available"""
       cache_key = hashlib.md5(task.encode()).hexdigest()
       cached = redis_client.get(cache_key)
       return json.loads(cached) if cached else None

   # Usage
   result = get_cached_result(task)
   if not result:
       result = agent.run(task)
       cache_agent_result(task, result)

Async Execution
^^^^^^^^^^^^^^^

Use async for better throughput:

.. code-block:: python

   import asyncio
   from concurrent.futures import ThreadPoolExecutor

   executor = ThreadPoolExecutor(max_workers=10)

   async def run_agent_async(task: str):
       """Run agent in thread pool"""
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(
           executor,
           agent.run,
           task
       )
       return result

   # Handle multiple requests concurrently
   tasks = [run_agent_async(t) for t in task_list]
   results = await asyncio.gather(*tasks)

Resource Limits
^^^^^^^^^^^^^^^

Set resource limits:

.. code-block:: python

   # Limit maximum iterations
   result = agent.run(task, max_iterations=50)

   # Timeout protection
   import signal

   def timeout_handler(signum, frame):
       raise TimeoutError("Agent execution timeout")

   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(300)  # 5 minute timeout

   try:
       result = agent.run(task)
   except TimeoutError:
       logger.error("Agent execution timed out")
   finally:
       signal.alarm(0)

Health Checks
-------------

Implement health check endpoint:

.. code-block:: python

   from fastapi import FastAPI, status
   from sqlalchemy import text

   app = FastAPI()

   @app.get("/health")
   async def health_check():
       """Health check endpoint"""
       health = {
           "status": "healthy",
           "version": "0.2.0",
           "checks": {}
       }

       # Check database
       try:
           with engine.connect() as conn:
               conn.execute(text("SELECT 1"))
           health["checks"]["database"] = "ok"
       except Exception as e:
           health["status"] = "unhealthy"
           health["checks"]["database"] = f"error: {str(e)}"

       # Check Redis
       try:
           redis_client.ping()
           health["checks"]["redis"] = "ok"
       except Exception as e:
           health["status"] = "unhealthy"
           health["checks"]["redis"] = f"error: {str(e)}"

       # Check API keys
       if not os.getenv("ANTHROPIC_API_KEY"):
           health["status"] = "unhealthy"
           health["checks"]["api_keys"] = "missing"
       else:
           health["checks"]["api_keys"] = "ok"

       status_code = (
           status.HTTP_200_OK if health["status"] == "healthy"
           else status.HTTP_503_SERVICE_UNAVAILABLE
       )

       return health, status_code

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**High Memory Usage:**

.. code-block:: bash

   # Check memory usage
   kubectl top pods -n lrs-agents

   # Increase memory limits
   # Update deployment.yaml and apply

**Database Connection Errors:**

.. code-block:: python

   # Enable connection pooling
   # Add pool_pre_ping=True
   # Increase pool_size

**Slow Response Times:**

.. code-block:: bash

   # Check logs for slow operations
   kubectl logs -f deployment/lrs-agents -n lrs-agents | grep "execution_time"

   # Enable caching
   # Scale horizontally

Debug Mode
^^^^^^^^^^

Enable debug logging:

.. code-block:: bash

   # Set environment variable
   export LOG_LEVEL=DEBUG

   # Or in Kubernetes
   kubectl set env deployment/lrs-agents LOG_LEVEL=DEBUG -n lrs-agents

Checklist
---------

Pre-deployment:

* [ ] API keys configured
* [ ] Database initialized
* [ ] Secrets properly managed
* [ ] Resource limits set
* [ ] Health checks implemented
* [ ] Monitoring configured
* [ ] Logging set up
* [ ] Backups automated
* [ ] Rate limiting enabled
* [ ] Load balancing configured

Post-deployment:

* [ ] Health checks passing
* [ ] Metrics being collected
* [ ] Logs aggregating correctly
* [ ] Alerts configured
* [ ] Dashboard accessible
* [ ] Performance acceptable
* [ ] Error rate within limits

Next Steps
----------

* Set up monitoring with Prometheus/Grafana
* Configure log aggregation (ELK, Datadog)
* Implement CI/CD pipeline
* Load test your deployment
* Document runbooks for common issues
* Set up on-call rotation

Further Reading
---------------

* :doc:`../api/monitoring` - Monitoring API reference
* :doc:`../tutorials/07_production_deployment` - Production tutorial
* Kubernetes documentation
* Docker best practices
* Prometheus operator guide

