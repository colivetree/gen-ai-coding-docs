# 🔌 API-Only Service Template

## Project Type: API-Only Service
**Stack:** FastAPI + PostgreSQL + Redis
**Includes:** Authentication, Workflows, Monitoring (no frontend)

## Specific Configuration

### API Service (FastAPI)
```dockerfile
# Dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: redis://redis:6379
      SUPABASE_URL: ${SUPABASE_URL}
      SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY}
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
      HATCHET_SERVER_URL: http://hatchet-engine:7070
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      auth:
        condition: service_started
    networks:
      - api-network
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - api-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m app.worker
    environment:
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: redis://redis:6379
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
      HATCHET_SERVER_URL: http://hatchet-engine:7070
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      hatchet-engine:
        condition: service_started
    networks:
      - api-network
    volumes:
      - ./app:/app/app
    restart: unless-stopped

networks:
  api-network:
    driver: bridge

volumes:
  redis_data:
```

### Requirements.txt
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
redis==5.0.1
celery==5.3.4
supabase==2.0.0
hatchet-sdk==0.25.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
httpx==0.25.2
structlog==23.2.0
prometheus-client==0.19.0
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
```

### Main API Application
```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import structlog
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .auth.supabase_client import verify_jwt_token
from .workflows.hatchet_client import workflow_router
from .database import engine, Base
from .config import settings

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting API service", service=settings.PROJECT_NAME)
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Shutting down API service")

app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    description="Microservice API with authentication and workflows",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "prod" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "prod" else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(duration)
    
    return response

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        user = verify_jwt_token(credentials.credentials)
        return user
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication")

@app.get("/")
async def root():
    return {
        "service": f"{settings.PROJECT_NAME} API",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": f"{settings.PROJECT_NAME}-api",
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    logger.info("Protected route accessed", user_id=current_user.id)
    return {"message": f"Hello {current_user.email}", "user": current_user}

# Include routers
app.include_router(workflow_router, prefix="/workflows", tags=["workflows"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Configuration Management
```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "api-service")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://redis:6379"
    
    # Authentication
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    JWT_SECRET: str
    
    # Hatchet
    HATCHET_CLIENT_TOKEN: str
    HATCHET_SERVER_URL: str = "http://hatchet-engine:7070"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Background Worker
```python
# app/worker.py
import asyncio
import structlog
from hatchet_sdk import Hatchet
from .config import settings

logger = structlog.get_logger()

hatchet = Hatchet(
    debug=settings.ENVIRONMENT != "prod",
    server_url=settings.HATCHET_SERVER_URL,
    token=settings.HATCHET_CLIENT_TOKEN
)

@hatchet.workflow(name="data-processing")
class DataProcessingWorkflow:
    @hatchet.step()
    def validate_data(self, context):
        logger.info("Validating data", workflow_run_id=context.workflow_run_id)
        # Data validation logic
        return {"status": "validated", "records": 100}
    
    @hatchet.step(parents=["validate_data"])
    def process_data(self, context):
        logger.info("Processing data", workflow_run_id=context.workflow_run_id)
        # Data processing logic
        return {"status": "processed", "output_file": "processed_data.json"}
    
    @hatchet.step(parents=["process_data"])
    def notify_completion(self, context):
        logger.info("Notifying completion", workflow_run_id=context.workflow_run_id)
        # Notification logic
        return {"status": "notified"}

@hatchet.workflow(name="api-task")
class ApiTaskWorkflow:
    @hatchet.step()
    def execute_task(self, context):
        task_data = context.workflow_input()
        logger.info("Executing API task", task_id=task_data.get("task_id"))
        
        # Task execution logic
        result = {
            "task_id": task_data.get("task_id"),
            "status": "completed",
            "result": "Task executed successfully"
        }
        
        return result

async def main():
    logger.info("Starting background worker")
    worker = hatchet.worker("api-worker", workflows=[
        DataProcessingWorkflow(),
        ApiTaskWorkflow()
    ])
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Database Models
```python
# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user_id = Column(String, index=True)
    workflow_run_id = Column(String, index=True, nullable=True)

class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, unique=True, index=True)
    name = Column(String)
    user_id = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
```

### API Routes
```python
# app/routes/tasks.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import structlog

from ..database import get_db
from ..models import Task
from ..auth.supabase_client import get_current_user
from ..workflows.hatchet_client import hatchet

router = APIRouter()
logger = structlog.get_logger()

@router.post("/tasks/")
async def create_task(
    task_data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create task in database
    task = Task(
        name=task_data["name"],
        description=task_data.get("description", ""),
        user_id=current_user.id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # Trigger workflow
    workflow_run = await hatchet.admin.run_workflow(
        "api-task",
        {
            "task_id": task.id,
            "user_id": current_user.id,
            **task_data
        }
    )
    
    # Update task with workflow run ID
    task.workflow_run_id = workflow_run.workflow_run_id
    db.commit()
    
    logger.info("Task created", task_id=task.id, workflow_run_id=workflow_run.workflow_run_id)
    
    return {
        "task_id": task.id,
        "workflow_run_id": workflow_run.workflow_run_id,
        "status": "created"
    }

@router.get("/tasks/")
async def list_tasks(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tasks = db.query(Task).filter(Task.user_id == current_user.id).all()
    return tasks

@router.get("/tasks/{task_id}")
async def get_task(
    task_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task
```

### Nginx Configuration
```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    upstream auth {
        server auth:9999;
    }

    upstream hatchet {
        server hatchet-dashboard:80;
    }

    upstream grafana {
        server grafana:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # API routes
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # API-specific headers
            proxy_set_header X-Request-ID $request_id;
            add_header X-Request-ID $request_id;
        }

        # Auth routes
        location /auth/ {
            proxy_pass http://auth/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Hatchet dashboard
        location /workflows/ {
            proxy_pass http://hatchet/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Grafana monitoring
        location /monitoring/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://api/health;
        }

        # Metrics endpoint
        location /metrics {
            proxy_pass http://api/metrics;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }

        # Default route for API documentation
        location / {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Environment Variables (.env.example)
```bash
# Project Configuration
PROJECT_NAME=my-api-service
ENVIRONMENT=dev
DOMAIN=localhost

# Database
POSTGRES_PASSWORD=your-secure-password
DATABASE_URL=postgres://postgres:${POSTGRES_PASSWORD}@postgres:5432/${PROJECT_NAME}

# Redis
REDIS_URL=redis://redis:6379

# Authentication (Supabase)
SUPABASE_URL=http://localhost:8000
SUPABASE_SERVICE_KEY=your-supabase-service-key
JWT_SECRET=your-jwt-secret
SITE_URL=http://localhost

# Hatchet Workflows
HATCHET_CLIENT_TOKEN=your-hatchet-token
HATCHET_SERVER_URL=http://hatchet-engine:7070

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin123

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=admin123

# Logging
LOG_LEVEL=INFO

# CORS
ALLOWED_ORIGINS=["*"]
```

### Bootstrap Command
```bash
#!/bin/bash
# Bootstrap API-only service

PROJECT_NAME=${1:-"my-api-service"}
ENVIRONMENT=${2:-"dev"}

echo "🔌 Bootstrapping API-Only Service: ${PROJECT_NAME}"

# Clone base infrastructure
git clone git@github.com:colivetree/ai-common-infra.git temp-infra
cp -r temp-infra/* .
rm -rf temp-infra

# Copy API-specific templates
cp project-type-templates/api-only-service-template.md ./
cp -r templates/api/* ./

# Generate environment file
cp .env.example .env
sed -i "s/{{PROJECT_NAME}}/${PROJECT_NAME}/g" .env
sed -i "s/{{ENVIRONMENT}}/${ENVIRONMENT}/g" .env

# Generate secure secrets
JWT_SECRET=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

sed -i "s/your-jwt-secret/${JWT_SECRET}/g" .env
sed -i "s/your-secure-password/${POSTGRES_PASSWORD}/g" .env
sed -i "s/admin123/${GRAFANA_PASSWORD}/g" .env

# Create project structure
mkdir -p {app,nginx,monitoring,scripts,docs,tests}
mkdir -p app/{auth,workflows,routes,models}
mkdir -p nginx/ssl
mkdir -p monitoring/grafana/dashboards
mkdir -p tests/{unit,integration}

# Create initial files
touch app/__init__.py
touch app/main.py
touch app/config.py
touch app/worker.py
touch requirements.txt

echo "✅ API-Only Service ${PROJECT_NAME} bootstrapped!"
echo "📝 Next steps:"
echo "   1. Configure Supabase keys in .env"
echo "   2. Configure Hatchet token in .env"
echo "   3. Run: docker-compose up -d"
echo "   4. Access API docs at: http://localhost/docs"
echo "   5. Access monitoring at: http://localhost/monitoring"
``` 