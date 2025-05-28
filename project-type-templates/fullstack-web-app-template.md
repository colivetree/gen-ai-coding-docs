# 🌐 Full-Stack Web Application Template

## Project Type: Full-Stack Web Application
**Stack:** React/Next.js + FastAPI + PostgreSQL
**Includes:** Authentication, Payments, Workflows, Monitoring

## Specific Configuration

### Frontend (Next.js)
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then yarn global add pnpm && pnpm i --frozen-lockfile; \
  else echo "Lockfile not found." && exit 1; \
  fi

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN yarn build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

### Backend (FastAPI)
```dockerfile
# backend/Dockerfile
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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      NEXT_PUBLIC_SUPABASE_URL: ${SUPABASE_URL}
      NEXT_PUBLIC_SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY}
      NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: ${STRIPE_PUBLISHABLE_KEY}
      NEXT_PUBLIC_API_URL: http://api:8000
    depends_on:
      - api
    networks:
      - app-network

  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      SUPABASE_URL: ${SUPABASE_URL}
      SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY}
      STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY}
      STRIPE_WEBHOOK_SECRET: ${STRIPE_WEBHOOK_SECRET}
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
      HATCHET_SERVER_URL: http://hatchet-engine:7070
    depends_on:
      postgres:
        condition: service_healthy
      auth:
        condition: service_started
    networks:
      - app-network
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"

networks:
  app-network:
    driver: bridge
```

### Frontend Package.json
```json
{
  "name": "{{PROJECT_NAME}}-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@supabase/supabase-js": "^2.38.0",
    "@stripe/stripe-js": "^2.1.0",
    "@stripe/react-stripe-js": "^2.3.0",
    "axios": "^1.5.0",
    "react-query": "^3.39.0",
    "react-hook-form": "^7.47.0",
    "tailwindcss": "^3.3.0",
    "@headlessui/react": "^1.7.0",
    "@heroicons/react": "^2.0.0",
    "clsx": "^2.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.8.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.2.0",
    "eslint": "^8.51.0",
    "eslint-config-next": "14.0.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}
```

### Backend Requirements.txt
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
supabase==2.0.0
stripe==7.6.0
hatchet-sdk==0.25.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
httpx==0.25.2
pytest==7.4.3
pytest-asyncio==0.21.1
```

### Frontend Components Structure
```typescript
// frontend/src/components/auth/AuthProvider.tsx
'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { User } from '@supabase/supabase-js'
import { supabase } from '@/lib/supabase'

interface AuthContextType {
  user: User | null
  loading: boolean
  signIn: (email: string, password: string) => Promise<void>
  signUp: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        setUser(session?.user ?? null)
        setLoading(false)
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const signIn = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) throw error
  }

  const signUp = async (email: string, password: string) => {
    const { error } = await supabase.auth.signUp({
      email,
      password,
    })
    if (error) throw error
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
  }

  return (
    <AuthContext.Provider value={{ user, loading, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
```

### Backend API Structure
```python
# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import os

from .auth.supabase_client import verify_jwt_token
from .payments.stripe_client import stripe_router
from .workflows.hatchet_client import workflow_router
from .database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    pass

app = FastAPI(
    title="{{PROJECT_NAME}} API",
    description="Full-stack web application API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        user = verify_jwt_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication")

@app.get("/")
async def root():
    return {"message": "{{PROJECT_NAME}} API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "{{PROJECT_NAME}}-api"}

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": f"Hello {current_user.email}", "user": current_user}

# Include routers
app.include_router(stripe_router, prefix="/payments", tags=["payments"])
app.include_router(workflow_router, prefix="/workflows", tags=["workflows"])
```

### Nginx Configuration
```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

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

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API routes
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
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
            return 200 'healthy\n';
            add_header Content-Type text/plain;
        }
    }
}
```

### Environment Variables (.env.example)
```bash
# Project Configuration
PROJECT_NAME=my-fullstack-app
ENVIRONMENT=dev
DOMAIN=localhost

# Database
POSTGRES_PASSWORD=your-secure-password
DATABASE_URL=postgres://postgres:${POSTGRES_PASSWORD}@postgres:5432/${PROJECT_NAME}

# Authentication (Supabase)
SUPABASE_URL=http://localhost:8000
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-key
JWT_SECRET=your-jwt-secret
SITE_URL=http://localhost

# Hatchet Workflows
HATCHET_CLIENT_TOKEN=your-hatchet-token
HATCHET_SERVER_URL=http://hatchet-engine:7070

# Stripe Payments
STRIPE_PUBLISHABLE_KEY=pk_test_your-publishable-key
STRIPE_SECRET_KEY=sk_test_your-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin123

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=admin123
```

### Bootstrap Command
```bash
#!/bin/bash
# Bootstrap full-stack web application

PROJECT_NAME=${1:-"my-fullstack-app"}
ENVIRONMENT=${2:-"dev"}

echo "🌐 Bootstrapping Full-Stack Web Application: ${PROJECT_NAME}"

# Clone base infrastructure
git clone git@github.com:colivetree/ai-common-infra.git temp-infra
cp -r temp-infra/* .
rm -rf temp-infra

# Copy full-stack specific templates
cp project-type-templates/fullstack-web-app-template.md ./
cp -r templates/fullstack/* ./

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
mkdir -p {frontend,backend,nginx,monitoring,scripts,docs,tests}
mkdir -p frontend/src/{components,pages,lib,styles}
mkdir -p backend/app/{auth,payments,workflows,models}
mkdir -p nginx/ssl
mkdir -p monitoring/grafana/dashboards

echo "✅ Full-Stack Web Application ${PROJECT_NAME} bootstrapped!"
echo "📝 Next steps:"
echo "   1. Configure Supabase keys in .env"
echo "   2. Configure Stripe keys in .env"
echo "   3. Run: docker-compose up -d"
echo "   4. Access your app at: http://localhost"
``` 