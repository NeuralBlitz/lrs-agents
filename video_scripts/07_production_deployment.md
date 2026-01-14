# Video 7: "Deploying LRS Agents to Production" (12 minutes)

## Script

[OPENING - 0:00-0:30]
VISUAL: Development laptop vs production cluster
VOICEOVER:
"You've built an LRS agent. It works on your laptop. Now you need to deploy 
it to production—scaling to handle thousands of requests, surviving node 
failures, and monitoring everything. Today we're going from prototype to 
production-grade deployment using Docker and Kubernetes."

[ARCHITECTURE OVERVIEW - 0:30-1:30]
VISUAL: Production architecture diagram
VOICEOVER:
"Here's what we're building: LRS agents running in Kubernetes pods, 
auto-scaling based on CPU and precision metrics. PostgreSQL for state 
persistence. Redis for caching. A monitoring dashboard. And all of it 
behind a load balancer."

DIAGRAM:
                ┌─────────────────┐
                │  Load Balancer  │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │   K8s Service   │
                └────────┬────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌────▼────┐ ┌────▼────┐ ┌────▼────┐ │ LRS Pod │ │ LRS Pod │ │ LRS Pod │ │ (2-10) │ │ (Auto- │ │ (scale) │ └────┬────┘ └────┬────┘ └────┬────┘ │ │ │ └───────────────────┼────────────────────┘ │ ┌───────────┴───────────┐ │ │ ┌──────▼──────┐ ┌──────▼──────┐ │ PostgreSQL │ │ Redis │ │ (State) │ │ (Cache) │ └─────────────┘ └─────────────┘

VOICEOVER:
"Pods auto-scale from 2 to 10 based on load. State persists in Postgres. 
Cache in Redis. And the horizontal pod autoscaler adjusts replica count 
automatically."

[CONTAINERIZATION - 1:30-3:00]
VISUAL: Dockerfile on screen
VOICEOVER:
"First, containerization. We use a multi-stage Docker build to keep images 
small and secure."

DOCKERFILE:
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY lrs/ ./lrs/
RUN pip install build && python -m build

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Install built package
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Create non-root user
RUN useradd -m -u 1000 lrs
USER lrs

# Health check
HEALTHCHECK CMD python -c "import lrs; print('healthy')"

CMD ["python", "-m", "lrs.monitoring.dashboard"]
VOICEOVER: “The builder stage compiles the package. The runtime stage installs it and runs as non-root for security. Health checks ensure Kubernetes knows when pods are ready.”

TERMINAL:

# Build image
docker build -t lrs-agents:0.2.0 -f docker/Dockerfile .

# Test locally
docker run -p 8501:8501 lrs-agents:0.2.0
[KUBERNETES DEPLOYMENT - 3:00-5:00] VISUAL: K8s YAML file VOICEOVER: “Next, Kubernetes deployment. This defines how pods run in production.”

YAML:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: lrs-agent
spec:
  replicas: 3  # Start with 3 pods
  selector:
    matchLabels:
      app: lrs-agent
  template:
    spec:
      containers:
      - name: lrs-agent
        image: lrs-agents:0.2.0
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: lrs-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command: ["python", "-c", "import lrs"]
          initialDelaySeconds: 30
          periodSeconds: 30
VOICEOVER: “API keys come from Kubernetes secrets—never hardcoded. Resource requests ensure the scheduler allocates enough CPU and memory. Liveness probes restart failed pods automatically.”

[AUTO-SCALING - 5:00-6:30] VISUAL: HPA configuration and live scaling demo VOICEOVER: “Here’s where it gets interesting: auto-scaling. The horizontal pod autoscaler watches CPU and memory usage and scales pods automatically.”

HPA YAML:

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lrs-agent-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: lrs-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100  # Double pods every 30s when scaling up
        periodSeconds: 30
VOICEOVER: “Minimum 2 pods for redundancy. Maximum 10 for cost control. Scale up when CPU hits 70%. And here’s the key: scale up aggressively (double every 30 seconds), but scale down conservatively (5 minute stabilization window). This prevents thrashing.”

LIVE DEMO:

# Watch scaling in action
kubectl get hpa -w

# Simulate load
hey -z 60s -c 50 http://lrs-agent-service/

# Output:
NAME            REFERENCE            TARGETS   MINPODS   MAXPODS   REPLICAS
lrs-agent-hpa   Deployment/lrs-agent 45%/70%   2         10        2
lrs-agent-hpa   Deployment/lrs-agent 85%/70%   2         10        2
lrs-agent-hpa   Deployment/lrs-agent 82%/70%   2         10        4  ← Scaled up
lrs-agent-hpa   Deployment/lrs-agent 65%/70%   2         10        4
[STATE PERSISTENCE - 6:30-7:30] VISUAL: PostgreSQL schema diagram VOICEOVER: “For production, state needs to persist. We use PostgreSQL to store precision history, tool executions, and adaptation events.”

SCHEMA:

CREATE TABLE agent_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    start_time TIMESTAMP DEFAULT NOW(),
    status VARCHAR(50)
);

CREATE TABLE precision_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    level VARCHAR(50) NOT NULL,
    precision_value FLOAT NOT NULL,
    prediction_error FLOAT
);

CREATE TABLE tool_executions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    tool_name VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    execution_time_ms FLOAT,
    prediction_error FLOAT
);
VOICEOVER: “Sessions track entire agent runs. Precision history stores every update. Tool executions log every action. This gives you full auditability.”

[SECRETS MANAGEMENT - 7:30-8:15] VISUAL: Kubernetes secrets creation VOICEOVER: “Never put API keys in your code or Docker images. Use Kubernetes secrets.”

TERMINAL:

# Create secret from environment variables
kubectl create secret generic lrs-secrets \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=postgres-password=$POSTGRES_PASSWORD

# Verify (values are base64 encoded)
kubectl get secret lrs-secrets -o yaml
VOICEOVER: “Secrets are encrypted at rest in etcd. Pods mount them as environment variables. And they never appear in logs or container images.”

[MONITORING & LOGGING - 8:15-9:30] VISUAL: Dashboard + Grafana metrics VOICEOVER: “Production needs observability. We provide structured JSON logging and Prometheus metrics.”

LOGGING EXAMPLE:

{
  "timestamp": "2025-01-14T10:23:45Z",
  "agent_id": "prod-agent-1",
  "session_id": "session_123",
  "event_type": "adaptation",
  "data": {
    "trigger": "high_prediction_error",
    "old_precision": {"execution": 0.75},
    "new_precision": {"execution": 0.41},
    "action": "switched_to_cache_fetch"
  }
}
VOICEOVER: “Each log entry is JSON—easily parsed by log aggregators like ELK or Datadog. Filter by event type. Query by agent ID. Trace entire sessions.”

GRAFANA DASHBOARD:

Metrics exposed:
- lrs_precision_value{level="execution|planning|abstract"}
- lrs_tool_execution_total{tool="...", success="true|false"}
- lrs_adaptation_events_total
- lrs_g_value{policy_id="..."}
[DEPLOYMENT SCRIPT - 9:30-10:30] VISUAL: Running deployment script VOICEOVER: “We provide a deployment script that handles everything.”

TERMINAL:

# Deploy to Kubernetes
./deploy/deploy.sh k8s production

Output:
🚀 Deploying LRS-Agents (k8s - production)
☸️  Deploying to Kubernetes...
Applying ConfigMap... ✓
Applying Secrets... ✓
Deploying application... ✓
Creating services... ✓
Setting up autoscaling... ✓
⏳ Waiting for deployment...
✅ Deployment complete!

📊 Get service URL:
   kubectl get svc lrs-agent-service -n lrs-agents

🔍 Check pod status:
   kubectl get pods -n lrs-agents

NAME                        READY   STATUS    RESTARTS   AGE
lrs-agent-7d9c8b5f-abc12    1/1     Running   0          2m
lrs-agent-7d9c8b5f-def34    1/1     Running   0          2m
lrs-agent-7d9c8b5f-ghi56    1/1     Running   0          2m
[ROLLOUT STRATEGY - 10:30-11:15] VISUAL: Rolling update animation VOICEOVER: “Kubernetes handles zero-downtime deployments via rolling updates.”

ANIMATION:

Initial state: 3 pods running v0.1.0

Rolling update to v0.2.0:
1. Create 1 new pod (v0.2.0)     [Old: 3, New: 1]
2. Wait for health check          [Old: 3, New: 1 ready]
3. Terminate 1 old pod            [Old: 2, New: 1]
4. Create another new pod         [Old: 2, New: 2]
5. Repeat until complete          [Old: 0, New: 3]
VOICEOVER: “At no point are all pods down. Traffic gradually shifts to new versions. If health checks fail, the rollout stops automatically.”

TERMINAL:

# Deploy new version
kubectl set image deployment/lrs-agent lrs-agent=lrs-agents:0.3.0

# Watch rollout
kubectl rollout status deployment/lrs-agent

# Rollback if needed
kubectl rollout undo deployment/lrs-agent
[DISASTER RECOVERY - 11:15-11:45] VISUAL: Backup and restore process VOICEOVER: “What if your cluster fails? Database backups ensure you don’t lose state.”

TERMINAL:

# Automated daily backup
kubectl create cronjob lrs-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- pg_dump -h postgres-service -U lrs lrs_agents > /backups/backup.sql

# Restore from backup
kubectl exec -it postgres-pod -- \
  psql -U lrs lrs_agents < backup.sql
[CLOSING - 11:45-12:00] VISUAL: Full production dashboard VOICEOVER: “You now have a production-grade LRS deployment. Auto-scaling. Zero-downtime updates. Full observability. State persistence. And disaster recovery. From prototype to production in one script.”

CODE:

# Complete deployment
git clone https://github.com/lrs-org/lrs-agents
cd lrs-agents/deploy
./deploy.sh k8s production
[END SCREEN
