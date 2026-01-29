from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Import services
from services.data_analysis_service import DataAnalysisService
from services.ml_models_service import MLModelsService
from services.ai_insights_service import AIInsightsService
from services.forecasting_service import ForecastingService

# MongoDB connection
mongo_url = os.environ['MONGO_URL'].strip()
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME'].strip()]

# Create the main app
app = FastAPI(title="Data Analysis API", version="2.0.0")

# Create routers
api_router = APIRouter(prefix="/api")
analysis_router = APIRouter(prefix="/api/analyze", tags=["Analysis"])
ml_router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])
ai_router = APIRouter(prefix="/api/ai", tags=["AI Insights"])
forecast_router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])

# Initialize services
ai_service = AIInsightsService()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Base Models ==============

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str


# ============== Analysis Models ==============

class DataRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None

class EDARequest(BaseModel):
    data: List[Dict[str, Any]]

class CorrelationRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None

class OutlierRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None
    method: str = "iqr"

class DistributionRequest(BaseModel):
    data: List[Dict[str, Any]]
    column: str


# ============== ML Models ==============

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_type: str = "auto"

class ClusteringRequest(BaseModel):
    data: List[Dict[str, Any]]
    feature_columns: Optional[List[str]] = None
    n_clusters: Optional[int] = None
    algorithm: str = "kmeans"

class AnomalyRequest(BaseModel):
    data: List[Dict[str, Any]]
    feature_columns: Optional[List[str]] = None
    contamination: float = 0.1


# ============== AI Models ==============

class InsightsRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    dataset_name: str = "Dataset"
    focus_areas: Optional[List[str]] = None

class QueryRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class ExplainRequest(BaseModel):
    analysis_type: str
    analysis_results: Dict[str, Any]
    data_context: str = ""

class RecommendationsRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    analysis_results: Dict[str, Any]
    business_context: str = ""


# ============== Forecast Models ==============

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    date_column: Optional[str] = None
    periods: int = 10
    method: str = "auto"

class MultiForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    periods: int = 10


# ============== Base Endpoints ==============

@api_router.get("/")
async def root():
    return {"message": "Data Analysis API v2.0", "status": "healthy"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "ml_service": "ready",
            "ai_service": "configured" if ai_service.api_key else "not_configured"
        }
    }

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks(limit: int = 100, skip: int = 0):
    """Get status checks with pagination."""
    status_checks = await db.status_checks.find().skip(skip).limit(min(limit, 100)).to_list(length=min(limit, 100))
    return [StatusCheck(**status_check) for status_check in status_checks]


# ============== Analysis Endpoints ==============

@analysis_router.post("/eda")
async def perform_eda(request: EDARequest):
    """Perform automated Exploratory Data Analysis."""
    try:
        result = DataAnalysisService.perform_eda(request.data)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"EDA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/correlations")
async def calculate_correlations(request: CorrelationRequest):
    """Calculate correlation matrix for numeric columns."""
    try:
        result = DataAnalysisService.calculate_correlations(request.data, request.columns)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Correlation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/outliers")
async def detect_outliers(request: OutlierRequest):
    """Detect outliers in numeric columns."""
    try:
        result = DataAnalysisService.detect_outliers(request.data, request.columns, request.method)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Outlier detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.post("/distribution")
async def analyze_distribution(request: DistributionRequest):
    """Analyze distribution of a specific column."""
    try:
        result = DataAnalysisService.get_distribution_analysis(request.data, request.column)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Distribution analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== ML Endpoints ==============

@ml_router.post("/predict")
async def train_prediction_model(request: PredictionRequest):
    """Train a prediction model (classification or regression)."""
    try:
        result = MLModelsService.train_prediction_model(
            data=request.data,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            model_type=request.model_type
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Prediction model error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.post("/cluster")
async def perform_clustering(request: ClusteringRequest):
    """Perform clustering analysis."""
    try:
        result = MLModelsService.perform_clustering(
            data=request.data,
            feature_columns=request.feature_columns,
            n_clusters=request.n_clusters,
            algorithm=request.algorithm
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.post("/anomaly")
async def detect_anomalies(request: AnomalyRequest):
    """Detect anomalies using Isolation Forest."""
    try:
        result = MLModelsService.detect_anomalies(
            data=request.data,
            feature_columns=request.feature_columns,
            contamination=request.contamination
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== AI Endpoints ==============

@ai_router.post("/insights")
async def generate_insights(request: InsightsRequest):
    """Generate AI-powered insights from the data."""
    try:
        result = await ai_service.generate_insights(
            data=request.data,
            columns=request.columns,
            dataset_name=request.dataset_name,
            focus_areas=request.focus_areas
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"AI Insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.post("/query")
async def answer_query(request: QueryRequest):
    """Answer natural language queries about the data."""
    try:
        result = await ai_service.answer_query(
            data=request.data,
            columns=request.columns,
            query=request.query,
            conversation_history=request.conversation_history
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.post("/explain")
async def explain_analysis(request: ExplainRequest):
    """Generate AI explanation for analysis results."""
    try:
        result = await ai_service.explain_analysis(
            analysis_type=request.analysis_type,
            analysis_results=request.analysis_results,
            data_context=request.data_context
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.post("/recommendations")
async def generate_recommendations(request: RecommendationsRequest):
    """Generate AI-powered recommendations based on analysis."""
    try:
        result = await ai_service.generate_recommendations(
            data=request.data,
            columns=request.columns,
            analysis_results=request.analysis_results,
            business_context=request.business_context
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Forecast Endpoints ==============

@forecast_router.post("/single")
async def forecast_single(request: ForecastRequest):
    """Perform time series forecasting on a single column."""
    try:
        result = ForecastingService.simple_forecast(
            data=request.data,
            value_column=request.value_column,
            date_column=request.date_column,
            periods=request.periods,
            method=request.method
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@forecast_router.post("/multi")
async def forecast_multiple(request: MultiForecastRequest):
    """Forecast multiple columns simultaneously."""
    try:
        result = ForecastingService.multi_column_forecast(
            data=request.data,
            columns=request.columns,
            periods=request.periods
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Multi-column forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Include Routers ==============

app.include_router(api_router)
app.include_router(analysis_router)
app.include_router(ml_router)
app.include_router(ai_router)
app.include_router(forecast_router)

# ============== Middleware ==============

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[origin.strip() for origin in os.environ.get('CORS_ORIGINS', '*').split(',')],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Shutdown Event ==============

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
