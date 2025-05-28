# 🤖 AI/ML Pipeline Template

## Project Type: AI/ML Pipeline
**Stack:** Python + MLflow + PostgreSQL + S3/MinIO + Jupyter
**Includes:** Data processing, model training, inference, monitoring, workflows

## Specific Configuration

### ML Pipeline Service
```dockerfile
# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ml_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
      HATCHET_SERVER_URL: http://hatchet-engine:7070
      SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_started
      minio:
        condition: service_healthy
    networks:
      - ml-network
    ports:
      - "8000:8000"
    volumes:
      - ./ml_pipeline:/app/ml_pipeline
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "
        pip install mlflow psycopg2-binary boto3 &&
        mlflow server 
        --backend-store-uri postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
        --host 0.0.0.0
        --port 5000
      "
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY}
      AWS_SECRET_ACCESS_KEY: ${MINIO_SECRET_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    networks:
      - ml-network
    ports:
      - "5000:5000"
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
    volumes:
      - minio_data:/data
    networks:
      - ml-network
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped

  jupyter:
    image: jupyter/scipy-notebook:latest
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      JUPYTER_TOKEN: ${JUPYTER_TOKEN}
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
      - ./models:/home/jovyan/models
    networks:
      - ml-network
    ports:
      - "8888:8888"
    restart: unless-stopped

  ml-worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m ml_pipeline.worker
    environment:
      DATABASE_URL: ${DATABASE_URL}
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
      HATCHET_CLIENT_TOKEN: ${HATCHET_CLIENT_TOKEN}
      HATCHET_SERVER_URL: http://hatchet-engine:7070
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_started
      minio:
        condition: service_healthy
      hatchet-engine:
        condition: service_started
    networks:
      - ml-network
    volumes:
      - ./ml_pipeline:/app/ml_pipeline
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

networks:
  ml-network:
    driver: bridge

volumes:
  minio_data:
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
supabase==2.0.0
hatchet-sdk==0.25.0
mlflow==2.8.1
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
boto3==1.34.0
minio==7.2.0
joblib==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
optuna==3.4.0
shap==0.43.0
evidently==0.4.11
great-expectations==0.18.5
dvc==3.32.0
python-multipart==0.0.6
structlog==23.2.0
prometheus-client==0.19.0
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
```

### Main ML API Application
```python
# ml_pipeline/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import structlog
import time
import pandas as pd
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from typing import Dict, Any, List

from .auth.supabase_client import verify_jwt_token
from .workflows.ml_workflows import MLWorkflowManager
from .models.ml_models import ModelManager
from .data.data_processor import DataProcessor
from .config import settings

# Metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total ML predictions', ['model_name', 'status'])
TRAINING_COUNT = Counter('ml_training_jobs_total', 'Total ML training jobs', ['status'])
PREDICTION_DURATION = Histogram('ml_prediction_duration_seconds', 'ML prediction duration')

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Pipeline service", service=settings.PROJECT_NAME)
    
    # Initialize ML components
    app.state.model_manager = ModelManager()
    app.state.data_processor = DataProcessor()
    app.state.workflow_manager = MLWorkflowManager()
    
    yield
    # Shutdown
    logger.info("Shutting down ML Pipeline service")

app = FastAPI(
    title=f"{settings.PROJECT_NAME} ML Pipeline",
    description="AI/ML Pipeline with training, inference, and monitoring",
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
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    if request.url.path.startswith("/predict"):
        PREDICTION_DURATION.observe(duration)
    
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
        "service": f"{settings.PROJECT_NAME} ML Pipeline",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "mlflow_uri": settings.MLFLOW_TRACKING_URI
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": f"{settings.PROJECT_NAME}-ml-pipeline",
        "timestamp": time.time(),
        "models_loaded": len(app.state.model_manager.loaded_models)
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Upload training data"""
    try:
        # Save uploaded file
        file_path = await app.state.data_processor.save_uploaded_file(file, current_user.id)
        
        # Validate data format
        validation_result = await app.state.data_processor.validate_data(file_path)
        
        logger.info("Data uploaded", user_id=current_user.id, file_path=file_path)
        
        return {
            "file_path": file_path,
            "validation": validation_result,
            "status": "uploaded"
        }
    except Exception as e:
        logger.error("Data upload failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/training/start")
async def start_training(
    training_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Start model training workflow"""
    try:
        # Trigger training workflow
        workflow_run = await app.state.workflow_manager.start_training_workflow(
            training_config, current_user.id
        )
        
        TRAINING_COUNT.labels(status="started").inc()
        
        logger.info("Training started", 
                   user_id=current_user.id, 
                   workflow_run_id=workflow_run.workflow_run_id)
        
        return {
            "workflow_run_id": workflow_run.workflow_run_id,
            "status": "training_started",
            "config": training_config
        }
    except Exception as e:
        TRAINING_COUNT.labels(status="failed").inc()
        logger.error("Training start failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/training/status/{workflow_run_id}")
async def get_training_status(
    workflow_run_id: str,
    current_user = Depends(get_current_user)
):
    """Get training job status"""
    try:
        status = await app.state.workflow_manager.get_workflow_status(workflow_run_id)
        return status
    except Exception as e:
        logger.error("Failed to get training status", error=str(e))
        raise HTTPException(status_code=404, detail="Training job not found")

@app.post("/predict/{model_name}")
async def predict(
    model_name: str,
    data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Make predictions using a trained model"""
    start_time = time.time()
    
    try:
        # Load model if not already loaded
        model = await app.state.model_manager.get_model(model_name)
        
        # Make prediction
        prediction = await app.state.model_manager.predict(model_name, data)
        
        PREDICTION_COUNT.labels(model_name=model_name, status="success").inc()
        
        logger.info("Prediction made", 
                   user_id=current_user.id, 
                   model_name=model_name,
                   duration=time.time() - start_time)
        
        return {
            "model_name": model_name,
            "prediction": prediction,
            "timestamp": time.time()
        }
    except Exception as e:
        PREDICTION_COUNT.labels(model_name=model_name, status="error").inc()
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models(current_user = Depends(get_current_user)):
    """List available models"""
    try:
        models = await app.state.model_manager.list_models(current_user.id)
        return {"models": models}
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.post("/models/{model_name}/deploy")
async def deploy_model(
    model_name: str,
    deployment_config: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Deploy a model for inference"""
    try:
        deployment = await app.state.model_manager.deploy_model(
            model_name, deployment_config, current_user.id
        )
        
        logger.info("Model deployed", 
                   user_id=current_user.id, 
                   model_name=model_name)
        
        return deployment
    except Exception as e:
        logger.error("Model deployment failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments")
async def list_experiments(current_user = Depends(get_current_user)):
    """List MLflow experiments"""
    try:
        experiments = await app.state.model_manager.list_experiments(current_user.id)
        return {"experiments": experiments}
    except Exception as e:
        logger.error("Failed to list experiments", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list experiments")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### ML Workflows
```python
# ml_pipeline/workflows/ml_workflows.py
import asyncio
import structlog
from hatchet_sdk import Hatchet
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from ..config import settings

logger = structlog.get_logger()

hatchet = Hatchet(
    debug=settings.ENVIRONMENT != "prod",
    server_url=settings.HATCHET_SERVER_URL,
    token=settings.HATCHET_CLIENT_TOKEN
)

class MLWorkflowManager:
    def __init__(self):
        self.hatchet = hatchet
    
    async def start_training_workflow(self, config, user_id):
        return await self.hatchet.admin.run_workflow(
            "ml-training-pipeline",
            {
                "config": config,
                "user_id": user_id
            }
        )
    
    async def get_workflow_status(self, workflow_run_id):
        # Implementation to get workflow status from Hatchet
        return {"status": "running", "workflow_run_id": workflow_run_id}

@hatchet.workflow(name="ml-training-pipeline")
class MLTrainingPipeline:
    @hatchet.step()
    def load_and_validate_data(self, context):
        """Load and validate training data"""
        config = context.workflow_input()["config"]
        user_id = context.workflow_input()["user_id"]
        
        logger.info("Loading data", user_id=user_id, config=config)
        
        try:
            # Load data from specified path
            data_path = config.get("data_path")
            data = pd.read_csv(data_path)
            
            # Basic validation
            if data.empty:
                raise ValueError("Dataset is empty")
            
            # Check for required columns
            required_columns = config.get("required_columns", [])
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            logger.info("Data loaded successfully", 
                       rows=len(data), 
                       columns=len(data.columns))
            
            return {
                "status": "data_loaded",
                "rows": len(data),
                "columns": len(data.columns),
                "data_path": data_path
            }
        except Exception as e:
            logger.error("Data loading failed", error=str(e))
            raise

    @hatchet.step(parents=["load_and_validate_data"])
    def preprocess_data(self, context):
        """Preprocess the data for training"""
        config = context.workflow_input()["config"]
        data_info = context.step_output("load_and_validate_data")
        
        logger.info("Preprocessing data")
        
        try:
            # Load data
            data = pd.read_csv(data_info["data_path"])
            
            # Apply preprocessing steps
            preprocessing_steps = config.get("preprocessing", {})
            
            # Handle missing values
            if preprocessing_steps.get("fill_missing"):
                data = data.fillna(data.mean(numeric_only=True))
            
            # Feature scaling
            if preprocessing_steps.get("scale_features"):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            
            # Save preprocessed data
            preprocessed_path = f"/app/data/preprocessed_{context.workflow_run_id}.csv"
            data.to_csv(preprocessed_path, index=False)
            
            logger.info("Data preprocessing completed")
            
            return {
                "status": "data_preprocessed",
                "preprocessed_path": preprocessed_path,
                "shape": data.shape
            }
        except Exception as e:
            logger.error("Data preprocessing failed", error=str(e))
            raise

    @hatchet.step(parents=["preprocess_data"])
    def train_model(self, context):
        """Train the ML model"""
        config = context.workflow_input()["config"]
        user_id = context.workflow_input()["user_id"]
        preprocess_info = context.step_output("preprocess_data")
        
        logger.info("Starting model training")
        
        try:
            # Set up MLflow experiment
            experiment_name = f"{user_id}_{config.get('experiment_name', 'default')}"
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Load preprocessed data
                data = pd.read_csv(preprocess_info["preprocessed_path"])
                
                # Prepare features and target
                target_column = config.get("target_column")
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Split data
                test_size = config.get("test_size", 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Train model
                model_type = config.get("model_type", "random_forest")
                model_params = config.get("model_params", {})
                
                if model_type == "random_forest":
                    model = RandomForestClassifier(**model_params)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log parameters and metrics
                mlflow.log_params(model_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("train_size", len(X_train))
                mlflow.log_metric("test_size", len(X_test))
                
                # Save model
                model_path = f"/app/models/model_{context.workflow_run_id}.joblib"
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                logger.info("Model training completed", 
                           accuracy=accuracy,
                           model_path=model_path)
                
                return {
                    "status": "model_trained",
                    "accuracy": accuracy,
                    "model_path": model_path,
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }
        except Exception as e:
            logger.error("Model training failed", error=str(e))
            raise

    @hatchet.step(parents=["train_model"])
    def validate_model(self, context):
        """Validate the trained model"""
        train_info = context.step_output("train_model")
        
        logger.info("Validating model")
        
        try:
            # Load model
            model = joblib.load(train_info["model_path"])
            
            # Perform additional validation
            # This could include cross-validation, bias testing, etc.
            
            validation_results = {
                "model_size_mb": os.path.getsize(train_info["model_path"]) / (1024 * 1024),
                "validation_passed": True
            }
            
            logger.info("Model validation completed", results=validation_results)
            
            return {
                "status": "model_validated",
                "validation_results": validation_results
            }
        except Exception as e:
            logger.error("Model validation failed", error=str(e))
            raise

@hatchet.workflow(name="batch-inference")
class BatchInferencePipeline:
    @hatchet.step()
    def load_inference_data(self, context):
        """Load data for batch inference"""
        config = context.workflow_input()
        
        logger.info("Loading inference data")
        
        try:
            data_path = config["data_path"]
            data = pd.read_csv(data_path)
            
            return {
                "status": "data_loaded",
                "rows": len(data),
                "data_path": data_path
            }
        except Exception as e:
            logger.error("Inference data loading failed", error=str(e))
            raise

    @hatchet.step(parents=["load_inference_data"])
    def run_batch_inference(self, context):
        """Run batch inference"""
        config = context.workflow_input()
        data_info = context.step_output("load_inference_data")
        
        logger.info("Running batch inference")
        
        try:
            # Load model and data
            model = joblib.load(config["model_path"])
            data = pd.read_csv(data_info["data_path"])
            
            # Make predictions
            predictions = model.predict(data)
            
            # Save results
            results_path = f"/app/data/predictions_{context.workflow_run_id}.csv"
            results_df = data.copy()
            results_df["prediction"] = predictions
            results_df.to_csv(results_path, index=False)
            
            logger.info("Batch inference completed", 
                       predictions_count=len(predictions))
            
            return {
                "status": "inference_completed",
                "results_path": results_path,
                "predictions_count": len(predictions)
            }
        except Exception as e:
            logger.error("Batch inference failed", error=str(e))
            raise
```

### Model Manager
```python
# ml_pipeline/models/ml_models.py
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import structlog
from ..config import settings

logger = structlog.get_logger()

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    async def get_model(self, model_name: str):
        """Get a model, loading it if necessary"""
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
        return self.loaded_models[model_name]
    
    async def load_model(self, model_name: str):
        """Load a model from MLflow"""
        try:
            # Load model from MLflow model registry
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            self.loaded_models[model_name] = model
            
            logger.info("Model loaded", model_name=model_name)
        except Exception as e:
            logger.error("Failed to load model", model_name=model_name, error=str(e))
            raise
    
    async def predict(self, model_name: str, data: Dict[str, Any]):
        """Make prediction using a loaded model"""
        try:
            model = await self.get_model(model_name)
            
            # Convert input data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            # Make prediction
            prediction = model.predict(df)
            
            # Convert numpy types to Python types for JSON serialization
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            return prediction
        except Exception as e:
            logger.error("Prediction failed", model_name=model_name, error=str(e))
            raise
    
    async def list_models(self, user_id: str) -> List[Dict[str, Any]]:
        """List available models for a user"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = []
            
            # Get registered models
            for model in client.search_registered_models():
                # Filter by user if needed
                models.append({
                    "name": model.name,
                    "latest_version": model.latest_versions[0].version if model.latest_versions else None,
                    "description": model.description
                })
            
            return models
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise
    
    async def deploy_model(self, model_name: str, config: Dict[str, Any], user_id: str):
        """Deploy a model for serving"""
        try:
            # Load the model to validate it works
            await self.load_model(model_name)
            
            deployment_info = {
                "model_name": model_name,
                "status": "deployed",
                "endpoint": f"/predict/{model_name}",
                "user_id": user_id,
                "config": config
            }
            
            logger.info("Model deployed", model_name=model_name, user_id=user_id)
            
            return deployment_info
        except Exception as e:
            logger.error("Model deployment failed", model_name=model_name, error=str(e))
            raise
    
    async def list_experiments(self, user_id: str) -> List[Dict[str, Any]]:
        """List MLflow experiments for a user"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = []
            
            for exp in client.search_experiments():
                if user_id in exp.name:  # Filter by user
                    experiments.append({
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "lifecycle_stage": exp.lifecycle_stage,
                        "artifact_location": exp.artifact_location
                    })
            
            return experiments
        except Exception as e:
            logger.error("Failed to list experiments", error=str(e))
            raise
```

### Data Processor
```python
# ml_pipeline/data/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import aiofiles
from fastapi import UploadFile
import structlog
from typing import Dict, Any
import os

logger = structlog.get_logger()

class DataProcessor:
    def __init__(self):
        self.data_dir = Path("/app/data")
        self.data_dir.mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile, user_id: str) -> str:
        """Save uploaded file to data directory"""
        try:
            # Create user-specific directory
            user_dir = self.data_dir / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Save file
            file_path = user_dir / file.filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info("File saved", file_path=str(file_path), user_id=user_id)
            
            return str(file_path)
        except Exception as e:
            logger.error("Failed to save file", error=str(e))
            raise
    
    async def validate_data(self, file_path: str) -> Dict[str, Any]:
        """Validate uploaded data"""
        try:
            # Load data
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Basic validation
            validation_results = {
                "rows": len(data),
                "columns": len(data.columns),
                "missing_values": data.isnull().sum().sum(),
                "data_types": data.dtypes.to_dict(),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
                "valid": True
            }
            
            # Check for issues
            if data.empty:
                validation_results["valid"] = False
                validation_results["issues"] = ["Dataset is empty"]
            
            logger.info("Data validation completed", results=validation_results)
            
            return validation_results
        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            return {
                "valid": False,
                "error": str(e)
            }
```

### Configuration
```python
# ml_pipeline/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "ml-pipeline")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    
    # Database
    DATABASE_URL: str
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    
    # MinIO/S3
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    
    # Authentication
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    JWT_SECRET: str
    
    # Hatchet
    HATCHET_CLIENT_TOKEN: str
    HATCHET_SERVER_URL: str = "http://hatchet-engine:7070"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Jupyter
    JUPYTER_TOKEN: str = "ml-pipeline-token"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Environment Variables (.env.example)
```bash
# Project Configuration
PROJECT_NAME=my-ml-pipeline
ENVIRONMENT=dev
DOMAIN=localhost

# Database
POSTGRES_PASSWORD=your-secure-password
DATABASE_URL=postgres://postgres:${POSTGRES_PASSWORD}@postgres:5432/${PROJECT_NAME}

# MinIO (S3-compatible storage)
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Authentication (Supabase)
SUPABASE_URL=http://localhost:8000
SUPABASE_SERVICE_KEY=your-supabase-service-key
JWT_SECRET=your-jwt-secret
SITE_URL=http://localhost

# Hatchet Workflows
HATCHET_CLIENT_TOKEN=your-hatchet-token
HATCHET_SERVER_URL=http://hatchet-engine:7070

# Jupyter
JUPYTER_TOKEN=ml-pipeline-token

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin123

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=admin123
```

### Bootstrap Command
```bash
#!/bin/bash
# Bootstrap AI/ML pipeline

PROJECT_NAME=${1:-"my-ml-pipeline"}
ENVIRONMENT=${2:-"dev"}

echo "🤖 Bootstrapping AI/ML Pipeline: ${PROJECT_NAME}"

# Clone base infrastructure
git clone git@github.com:colivetree/ai-common-infra.git temp-infra
cp -r temp-infra/* .
rm -rf temp-infra

# Copy ML-specific templates
cp project-type-templates/ai-ml-pipeline-template.md ./
cp -r templates/ml/* ./

# Generate environment file
cp .env.example .env
sed -i "s/{{PROJECT_NAME}}/${PROJECT_NAME}/g" .env
sed -i "s/{{ENVIRONMENT}}/${ENVIRONMENT}/g" .env

# Generate secure secrets
JWT_SECRET=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
MINIO_SECRET=$(openssl rand -base64 32)

sed -i "s/your-jwt-secret/${JWT_SECRET}/g" .env
sed -i "s/your-secure-password/${POSTGRES_PASSWORD}/g" .env
sed -i "s/admin123/${GRAFANA_PASSWORD}/g" .env
sed -i "s/minioadmin123/${MINIO_SECRET}/g" .env

# Create project structure
mkdir -p {ml_pipeline,data,models,notebooks,monitoring,scripts,docs,tests}
mkdir -p ml_pipeline/{auth,workflows,models,data}
mkdir -p data/{raw,processed,predictions}
mkdir -p models/{trained,deployed}
mkdir -p notebooks/{exploration,training,evaluation}
mkdir -p monitoring/grafana/dashboards
mkdir -p tests/{unit,integration}

# Create initial files
touch ml_pipeline/__init__.py
touch ml_pipeline/main.py
touch ml_pipeline/config.py
touch ml_pipeline/worker.py
touch requirements.txt

echo "✅ AI/ML Pipeline ${PROJECT_NAME} bootstrapped!"
echo "📝 Next steps:"
echo "   1. Configure Supabase keys in .env"
echo "   2. Configure Hatchet token in .env"
echo "   3. Run: docker-compose up -d"
echo "   4. Access services:"
echo "      - ML API: http://localhost:8000/docs"
echo "      - MLflow: http://localhost:5000"
echo "      - Jupyter: http://localhost:8888 (token: ml-pipeline-token)"
echo "      - MinIO: http://localhost:9001"
echo "      - Monitoring: http://localhost/monitoring"
``` 