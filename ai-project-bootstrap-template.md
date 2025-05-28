<!-- ╔════════════════════════════════════════════════════════════╗
     ║  AI PROJECT BOOTSTRAP — ORG-WIDE STANDARD  · v1.0 · 2025-01-27 ║
     ╚════════════════════════════════════════════════════════════╝ -->
# 🚀 AI Project Bootstrap Preamble

## 1 · Mission
Bootstrap **production-ready** projects with standardized infrastructure, authentication, monitoring, and payment systems using our proven AI common infrastructure stack.

## 2 · Common Infrastructure Repository
| Component | Repository | Role |
|-----------|------------|------|
| ai-common-infra | git@github.com:colivetree/ai-common-infra.git | Core infrastructure services |

**Bootstrap first:** leverage existing infrastructure before building custom solutions.

## 3 · Core Infrastructure Stack

### Core Services
| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| Nginx | Reverse proxy & load balancer | 80/443 | `/health` |
| Supabase Auth | Authentication & authorization | 8000 | `/auth/v1/health` |
| Hatchet | Workflow orchestration | 8080 | `/api/v1/health` |
| RabbitMQ | Message broker | 5672/15672 | Management UI |

### Monitoring Services
| Service | Purpose | Port | Dashboard |
|---------|---------|------|-----------|
| Prometheus | Metrics collection | 9090 | `/graph` |
| Grafana | Metrics visualization | 3000 | `/login` |

### Payment & Billing
| Service | Purpose | Integration |
|---------|---------|-------------|
| Stripe | Payment processing | Webhook endpoints |

## 4 · Project Types & Templates

### 4.1 · Full-Stack Web Application
**Use case:** React/Next.js frontend with FastAPI backend
**Includes:** Auth, payments, workflows, monitoring

### 4.2 · API-Only Service
**Use case:** Microservice or API gateway
**Includes:** Auth, workflows, monitoring (no frontend)

### 4.3 · AI/ML Pipeline
**Use case:** Data processing, model training, inference
**Includes:** Hatchet workflows, monitoring, storage

### 4.4 · SaaS Application
**Use case:** Multi-tenant application with billing
**Includes:** Full stack + Stripe + tenant isolation

## 5 · Bootstrap Protocol
1. **Clone** ai-common-infra repository
2. **Configure** project-specific environment variables
3. **Customize** docker-compose overrides
4. **Initialize** database schemas and auth
5. **Deploy** and validate all services

## 6 · Environment Configuration

### Required Environment Variables
```bash
# Project Configuration
PROJECT_NAME={{PROJECT_NAME}}
ENVIRONMENT={{ENVIRONMENT}} # dev/staging/prod
DOMAIN={{DOMAIN}}

# Database
POSTGRES_PASSWORD={{POSTGRES_PASSWORD}}
DATABASE_URL=postgres://postgres:${POSTGRES_PASSWORD}@postgres:5432/${PROJECT_NAME}

# Authentication (Supabase)
SUPABASE_URL=http://localhost:8000
SUPABASE_ANON_KEY={{SUPABASE_ANON_KEY}}
SUPABASE_SERVICE_KEY={{SUPABASE_SERVICE_KEY}}
JWT_SECRET={{JWT_SECRET}}

# Hatchet Workflows
HATCHET_CLIENT_TOKEN={{HATCHET_CLIENT_TOKEN}}
HATCHET_SERVER_URL=http://hatchet-engine:7070

# Stripe Payments
STRIPE_PUBLISHABLE_KEY={{STRIPE_PUBLISHABLE_KEY}}
STRIPE_SECRET_KEY={{STRIPE_SECRET_KEY}}
STRIPE_WEBHOOK_SECRET={{STRIPE_WEBHOOK_SECRET}}

# Monitoring
GRAFANA_ADMIN_PASSWORD={{GRAFANA_ADMIN_PASSWORD}}
```

## 7 · Docker Compose Structure

### Base Infrastructure (from ai-common-infra)
```yaml
# docker-compose.base.yml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - auth
      - api

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${PROJECT_NAME}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Supabase Auth Stack
  auth:
    image: supabase/gotrue:latest
    environment:
      GOTRUE_API_HOST: 0.0.0.0
      GOTRUE_API_PORT: 9999
      GOTRUE_DB_DRIVER: postgres
      GOTRUE_DB_DATABASE_URL: ${DATABASE_URL}
      GOTRUE_JWT_SECRET: ${JWT_SECRET}
      GOTRUE_SITE_URL: ${SITE_URL}
    depends_on:
      postgres:
        condition: service_healthy

  # Hatchet Workflow Engine
  hatchet-engine:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-engine:latest
    environment:
      DATABASE_URL: ${DATABASE_URL}
      SERVER_GRPC_BIND_ADDRESS: "0.0.0.0"
    depends_on:
      postgres:
        condition: service_healthy

  # Message Broker
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    ports:
      - "15672:15672"

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

### Project-Specific Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      SUPABASE_URL: ${SUPABASE_URL}
      STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY}
    depends_on:
      - postgres
      - auth

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      NEXT_PUBLIC_SUPABASE_URL: ${SUPABASE_URL}
      NEXT_PUBLIC_SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY}
      NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: ${STRIPE_PUBLISHABLE_KEY}
    depends_on:
      - api
```

## 8 · Project Structure Template

```
{{PROJECT_NAME}}/
├── docker-compose.yml              # Main compose file
├── docker-compose.override.yml     # Project-specific services
├── .env                           # Environment variables
├── .env.example                   # Template for environment variables
├── backend/                       # API service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── auth/
│   │   ├── payments/
│   │   └── workflows/
│   └── tests/
├── frontend/                      # Web application (if applicable)
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   └── public/
├── nginx/                         # Reverse proxy configuration
│   ├── nginx.conf
│   └── ssl/
├── monitoring/                    # Observability configuration
│   ├── prometheus.yml
│   └── grafana/
├── scripts/                       # Deployment and utility scripts
│   ├── bootstrap.sh
│   ├── deploy.sh
│   └── backup.sh
├── docs/                         # Project documentation
│   ├── README.md
│   ├── API.md
│   └── DEPLOYMENT.md
└── tests/                        # Integration tests
    ├── e2e/
    └── load/
```

## 9 · Authentication Integration

### Supabase Auth Setup
```python
# backend/app/auth/supabase_client.py
from supabase import create_client, Client
import os

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def verify_jwt_token(token: str):
    try:
        user = supabase.auth.get_user(token)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Frontend Auth Integration
```typescript
// frontend/src/lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

## 10 · Payment Integration

### Stripe Setup
```python
# backend/app/payments/stripe_client.py
import stripe
import os

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

async def create_payment_intent(amount: int, currency: str = "usd"):
    return stripe.PaymentIntent.create(
        amount=amount,
        currency=currency,
        automatic_payment_methods={"enabled": True}
    )

async def handle_webhook(payload: str, sig_header: str):
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        # Handle event
        return event
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
```

## 11 · Workflow Integration

### Hatchet Workflow Setup
```python
# backend/app/workflows/hatchet_client.py
from hatchet_sdk import Hatchet
import os

hatchet = Hatchet(
    debug=True,
    server_url=os.getenv("HATCHET_SERVER_URL"),
    token=os.getenv("HATCHET_CLIENT_TOKEN")
)

@hatchet.workflow(name="process-payment")
class ProcessPaymentWorkflow:
    @hatchet.step()
    def validate_payment(self, context):
        # Validate payment details
        return {"status": "validated"}
    
    @hatchet.step(parents=["validate_payment"])
    def charge_customer(self, context):
        # Process payment with Stripe
        return {"status": "charged"}
    
    @hatchet.step(parents=["charge_customer"])
    def send_confirmation(self, context):
        # Send confirmation email
        return {"status": "confirmed"}
```

## 12 · Monitoring Configuration

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'hatchet'
    static_configs:
      - targets: ['hatchet-engine:7070']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "{{PROJECT_NAME}} Monitoring",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds{job=\"api\"}"
          }
        ]
      }
    ]
  }
}
```

## 13 · Bootstrap Scripts

### Main Bootstrap Script
```bash
#!/bin/bash
# scripts/bootstrap.sh

set -e

PROJECT_NAME=${1:-"my-project"}
ENVIRONMENT=${2:-"dev"}

echo "🚀 Bootstrapping ${PROJECT_NAME} for ${ENVIRONMENT} environment..."

# Clone common infrastructure
git clone git@github.com:colivetree/ai-common-infra.git temp-infra
cp -r temp-infra/* .
rm -rf temp-infra

# Generate environment file
cp .env.example .env
sed -i "s/{{PROJECT_NAME}}/${PROJECT_NAME}/g" .env
sed -i "s/{{ENVIRONMENT}}/${ENVIRONMENT}/g" .env

# Generate secure secrets
JWT_SECRET=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

sed -i "s/{{JWT_SECRET}}/${JWT_SECRET}/g" .env
sed -i "s/{{POSTGRES_PASSWORD}}/${POSTGRES_PASSWORD}/g" .env
sed -i "s/{{GRAFANA_ADMIN_PASSWORD}}/${GRAFANA_PASSWORD}/g" .env

# Initialize project structure
mkdir -p backend/app/{auth,payments,workflows}
mkdir -p frontend/src/{components,pages,lib}
mkdir -p nginx/ssl
mkdir -p monitoring/grafana
mkdir -p scripts
mkdir -p docs
mkdir -p tests/{e2e,load}

echo "✅ Project ${PROJECT_NAME} bootstrapped successfully!"
echo "📝 Next steps:"
echo "   1. Update .env with your specific configuration"
echo "   2. Configure Stripe keys"
echo "   3. Run: docker-compose up -d"
echo "   4. Access services:"
echo "      - Application: http://localhost"
echo "      - Grafana: http://localhost:3000"
echo "      - Hatchet: http://localhost:8080"
```

## 14 · Deployment Validation

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

echo "🔍 Checking service health..."

services=(
  "http://localhost/health:Nginx"
  "http://localhost:8000/auth/v1/health:Auth"
  "http://localhost:8080/api/v1/health:Hatchet"
  "http://localhost:3000/api/health:Grafana"
  "http://localhost:9090/-/healthy:Prometheus"
)

for service in "${services[@]}"; do
  url=$(echo $service | cut -d: -f1)
  name=$(echo $service | cut -d: -f2)
  
  if curl -f -s $url > /dev/null; then
    echo "✅ $name is healthy"
  else
    echo "❌ $name is not responding"
  fi
done
```

## 15 · Project Type Variants

### 15.1 · Full-Stack Web Application
```bash
# Generate full-stack project
./scripts/bootstrap.sh my-saas-app prod --type=fullstack
```

### 15.2 · API-Only Service
```bash
# Generate API-only project
./scripts/bootstrap.sh my-api-service prod --type=api
```

### 15.3 · AI/ML Pipeline
```bash
# Generate AI/ML project
./scripts/bootstrap.sh my-ml-pipeline prod --type=ml
```

### 15.4 · SaaS Application
```bash
# Generate SaaS project with multi-tenancy
./scripts/bootstrap.sh my-saas-platform prod --type=saas
```

## 16 · Extensibility Tokens
{{PROJECT_NAME}}, {{ENVIRONMENT}}, {{DOMAIN}}, {{POSTGRES_PASSWORD}}, {{JWT_SECRET}}, {{STRIPE_SECRET_KEY}}, {{HATCHET_CLIENT_TOKEN}}

----
# Project Bootstrap Command Template

**Project Name:** {{PROJECT_NAME}}
**Type:** {{PROJECT_TYPE}} # fullstack|api|ml|saas
**Environment:** {{ENVIRONMENT}} # dev|staging|prod

**Description:**
{{PROJECT_DESCRIPTION}}

**Required Integrations:**
- [ ] Authentication (Supabase)
- [ ] Payments (Stripe) 
- [ ] Workflows (Hatchet)
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Message Queue (RabbitMQ)

**Custom Requirements:**
{{CUSTOM_REQUIREMENTS}}

**Deployment Target:**
{{DEPLOYMENT_TARGET}} # local|aws|gcp|azure 