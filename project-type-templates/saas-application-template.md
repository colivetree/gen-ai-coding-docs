# 💼 SaaS Application Template

## Project Type: SaaS Application
**Stack:** Next.js + FastAPI + PostgreSQL + Stripe + Multi-tenancy
**Includes:** Authentication, billing, subscriptions, tenant isolation, monitoring

## Specific Configuration

### Multi-tenant Architecture
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  saas-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      NEXT_PUBLIC_SUPABASE_URL: ${SUPABASE_URL}
      NEXT_PUBLIC_SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY}
      NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: ${STRIPE_PUBLISHABLE_KEY}
      NEXT_PUBLIC_API_URL: http://api:8000
    depends_on:
      - saas-api
    networks:
      - saas-network
    ports:
      - "3000:3000"

  saas-api:
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
      TENANT_ISOLATION_MODE: ${TENANT_ISOLATION_MODE:-schema}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - saas-network
    ports:
      - "8000:8000"

  billing-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: python -m saas_app.workers.billing_worker
    environment:
      DATABASE_URL: ${DATABASE_URL}
      STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY}
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - saas-network

networks:
  saas-network:
    driver: bridge
```

### Multi-tenant Database Models
```python
# backend/saas_app/models/tenant.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    subdomain = Column(String, unique=True, nullable=False)
    custom_domain = Column(String, unique=True, nullable=True)
    plan = Column(String, default="free")
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Billing
    stripe_customer_id = Column(String, unique=True, nullable=True)
    subscription_id = Column(String, unique=True, nullable=True)
    subscription_status = Column(String, nullable=True)
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    
    # Settings
    settings = Column(Text, nullable=True)  # JSON
    
    # Relationships
    users = relationship("TenantUser", back_populates="tenant")

class TenantUser(Base):
    __tablename__ = "tenant_users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String, nullable=False)  # Supabase user ID
    role = Column(String, default="member")  # owner, admin, member
    status = Column(String, default="active")
    invited_by = Column(String, nullable=True)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="users")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    stripe_subscription_id = Column(String, unique=True, nullable=False)
    stripe_customer_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    plan_id = Column(String, nullable=False)
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    cancel_at_period_end = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### Tenant Middleware
```python
# backend/saas_app/middleware/tenant.py
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from ..models.tenant import Tenant
from ..database import get_db
import re

class TenantMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Extract tenant from subdomain or custom domain
            host = request.headers.get("host", "")
            tenant = await self.resolve_tenant(host)
            
            if tenant:
                scope["tenant"] = tenant
            
        await self.app(scope, receive, send)
    
    async def resolve_tenant(self, host: str) -> Tenant:
        """Resolve tenant from host header"""
        # Remove port if present
        host = host.split(":")[0]
        
        # Check for subdomain pattern (tenant.domain.com)
        subdomain_match = re.match(r"^([^.]+)\.(.+)$", host)
        if subdomain_match:
            subdomain = subdomain_match.group(1)
            if subdomain != "www":
                # Look up tenant by subdomain
                db = next(get_db())
                tenant = db.query(Tenant).filter(
                    Tenant.subdomain == subdomain,
                    Tenant.status == "active"
                ).first()
                return tenant
        
        # Check for custom domain
        db = next(get_db())
        tenant = db.query(Tenant).filter(
            Tenant.custom_domain == host,
            Tenant.status == "active"
        ).first()
        
        return tenant

def get_current_tenant(request: Request) -> Tenant:
    """Get current tenant from request"""
    tenant = getattr(request.scope, "tenant", None)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return tenant
```

### Billing Integration
```python
# backend/saas_app/billing/stripe_service.py
import stripe
from typing import Dict, Any
from ..models.tenant import Tenant, Subscription
from ..config import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

class BillingService:
    def __init__(self):
        self.plans = {
            "free": {"price_id": None, "features": ["basic"]},
            "pro": {"price_id": "price_pro", "features": ["basic", "advanced"]},
            "enterprise": {"price_id": "price_enterprise", "features": ["basic", "advanced", "premium"]}
        }
    
    async def create_customer(self, tenant: Tenant, user_email: str) -> str:
        """Create Stripe customer for tenant"""
        customer = stripe.Customer.create(
            email=user_email,
            name=tenant.name,
            metadata={
                "tenant_id": tenant.id,
                "tenant_name": tenant.name
            }
        )
        return customer.id
    
    async def create_subscription(self, tenant: Tenant, plan: str) -> Dict[str, Any]:
        """Create subscription for tenant"""
        if plan not in self.plans:
            raise ValueError(f"Invalid plan: {plan}")
        
        plan_config = self.plans[plan]
        if not plan_config["price_id"]:
            raise ValueError("Cannot create subscription for free plan")
        
        subscription = stripe.Subscription.create(
            customer=tenant.stripe_customer_id,
            items=[{"price": plan_config["price_id"]}],
            trial_period_days=14,
            metadata={
                "tenant_id": tenant.id,
                "plan": plan
            }
        )
        
        return {
            "subscription_id": subscription.id,
            "status": subscription.status,
            "current_period_start": subscription.current_period_start,
            "current_period_end": subscription.current_period_end,
            "trial_end": subscription.trial_end
        }
    
    async def handle_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
            
            if event["type"] == "customer.subscription.created":
                await self._handle_subscription_created(event["data"]["object"])
            elif event["type"] == "customer.subscription.updated":
                await self._handle_subscription_updated(event["data"]["object"])
            elif event["type"] == "customer.subscription.deleted":
                await self._handle_subscription_deleted(event["data"]["object"])
            elif event["type"] == "invoice.payment_succeeded":
                await self._handle_payment_succeeded(event["data"]["object"])
            elif event["type"] == "invoice.payment_failed":
                await self._handle_payment_failed(event["data"]["object"])
            
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
```

### Frontend Multi-tenancy
```typescript
// frontend/src/lib/tenant.ts
import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'

interface Tenant {
  id: string
  name: string
  subdomain: string
  plan: string
  status: string
}

export function useTenant() {
  const [tenant, setTenant] = useState<Tenant | null>(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    const fetchTenant = async () => {
      try {
        // Extract tenant from current domain
        const host = window.location.host
        const response = await fetch(`/api/tenant/resolve?host=${host}`)
        
        if (response.ok) {
          const tenantData = await response.json()
          setTenant(tenantData)
        }
      } catch (error) {
        console.error('Failed to fetch tenant:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchTenant()
  }, [])
  
  return { tenant, loading }
}

// Tenant-aware API client
export class TenantApiClient {
  private baseUrl: string
  private tenantId: string
  
  constructor(tenantId: string) {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    this.tenantId = tenantId
  }
  
  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseUrl}${endpoint}`
    
    const headers = {
      'Content-Type': 'application/json',
      'X-Tenant-ID': this.tenantId,
      ...options.headers,
    }
    
    return fetch(url, {
      ...options,
      headers,
    })
  }
  
  async get(endpoint: string) {
    return this.request(endpoint, { method: 'GET' })
  }
  
  async post(endpoint: string, data: any) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }
}
```

### Subscription Management UI
```typescript
// frontend/src/components/billing/SubscriptionManager.tsx
import { useState, useEffect } from 'react'
import { useTenant } from '@/lib/tenant'
import { loadStripe } from '@stripe/stripe-js'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

interface Plan {
  id: string
  name: string
  price: number
  features: string[]
}

export function SubscriptionManager() {
  const { tenant } = useTenant()
  const [plans, setPlans] = useState<Plan[]>([])
  const [currentPlan, setCurrentPlan] = useState<string>('')
  const [loading, setLoading] = useState(false)
  
  useEffect(() => {
    fetchPlans()
    if (tenant) {
      setCurrentPlan(tenant.plan)
    }
  }, [tenant])
  
  const fetchPlans = async () => {
    try {
      const response = await fetch('/api/billing/plans')
      const data = await response.json()
      setPlans(data.plans)
    } catch (error) {
      console.error('Failed to fetch plans:', error)
    }
  }
  
  const handleUpgrade = async (planId: string) => {
    setLoading(true)
    
    try {
      const response = await fetch('/api/billing/create-checkout-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planId }),
      })
      
      const { sessionId } = await response.json()
      
      const stripe = await stripePromise
      await stripe?.redirectToCheckout({ sessionId })
    } catch (error) {
      console.error('Failed to create checkout session:', error)
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Subscription Plans</h2>
      
      <div className="grid md:grid-cols-3 gap-6">
        {plans.map((plan) => (
          <div
            key={plan.id}
            className={`border rounded-lg p-6 ${
              currentPlan === plan.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
            }`}
          >
            <h3 className="text-xl font-semibold mb-2">{plan.name}</h3>
            <p className="text-3xl font-bold mb-4">
              ${plan.price}
              <span className="text-sm font-normal text-gray-600">/month</span>
            </p>
            
            <ul className="mb-6 space-y-2">
              {plan.features.map((feature, index) => (
                <li key={index} className="flex items-center">
                  <svg className="w-4 h-4 text-green-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {feature}
                </li>
              ))}
            </ul>
            
            {currentPlan === plan.id ? (
              <button
                disabled
                className="w-full bg-gray-300 text-gray-500 py-2 px-4 rounded cursor-not-allowed"
              >
                Current Plan
              </button>
            ) : (
              <button
                onClick={() => handleUpgrade(plan.id)}
                disabled={loading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? 'Processing...' : 'Upgrade'}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

### Environment Variables (.env.example)
```bash
# Project Configuration
PROJECT_NAME=my-saas-app
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

# Stripe Billing
STRIPE_PUBLISHABLE_KEY=pk_test_your-publishable-key
STRIPE_SECRET_KEY=sk_test_your-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret

# Hatchet Workflows
HATCHET_CLIENT_TOKEN=your-hatchet-token
HATCHET_SERVER_URL=http://hatchet-engine:7070

# Multi-tenancy
TENANT_ISOLATION_MODE=schema  # schema, database, or row_level

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin123

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=admin123
```

### Bootstrap Command
```bash
#!/bin/bash
# Bootstrap SaaS application

PROJECT_NAME=${1:-"my-saas-app"}
ENVIRONMENT=${2:-"dev"}

echo "💼 Bootstrapping SaaS Application: ${PROJECT_NAME}"

# Clone base infrastructure
git clone git@github.com:colivetree/ai-common-infra.git temp-infra
cp -r temp-infra/* .
rm -rf temp-infra

# Copy SaaS-specific templates
cp project-type-templates/saas-application-template.md ./

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
mkdir -p backend/saas_app/{models,middleware,billing,workers}
mkdir -p nginx/ssl
mkdir -p monitoring/grafana/dashboards

echo "✅ SaaS Application ${PROJECT_NAME} bootstrapped!"
echo "📝 Next steps:"
echo "   1. Configure Supabase keys in .env"
echo "   2. Configure Stripe keys in .env"
echo "   3. Configure Hatchet token in .env"
echo "   4. Run: docker-compose up -d"
echo "   5. Access your SaaS app at: http://localhost"
echo "   6. Set up tenant subdomains in your DNS/hosts file"
``` 