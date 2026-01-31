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

# Load environment variables for backend services
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Import services
from services.data_analysis_service import DataAnalysisService
from services.ml_models_service import MLModelsService
from services.ai_insights_service import AIInsightsService
from services.forecasting_service import ForecastingService
from services.nlp_engine import NLPEngine
from services.analysis_engine import AnalysisEngine
from services.ml_workbench import MLWorkbench
from services.ai_data_analyzer import AIDataAnalyzer

# MongoDB connection (used by routers; client is initialized on import)
_mongodb_uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URL")
if not _mongodb_uri:
    raise RuntimeError("MONGODB_URI environment variable is not set")

client = AsyncIOMotorClient(_mongodb_uri.strip())
db = client[os.environ["DB_NAME"].strip()]

# Create routers
api_router = APIRouter(prefix="/api")
analysis_router = APIRouter(prefix="/api/analyze", tags=["Analysis"])
ml_router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])
ai_router = APIRouter(prefix="/api/ai", tags=["AI Insights"])
forecast_router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])
advanced_router = APIRouter(prefix="/api/advanced", tags=["Advanced AI/ML"])

# Initialize services
ai_service = AIInsightsService()
nlp_engine = NLPEngine()
analysis_engine = AnalysisEngine()
ml_workbench = MLWorkbench()
ai_analyzer = AIDataAnalyzer()

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


# ============== Advanced AI/ML Models ==============

class SmartAnalyzeRequest(BaseModel):
    data: List[Dict[str, Any]]
    query: Optional[str] = None
    analysis_type: str = "auto"
    options: Optional[Dict[str, Any]] = None

class SmartQueryRequest(BaseModel):
    data: List[Dict[str, Any]]
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class DataQualityRequest(BaseModel):
    data: List[Dict[str, Any]]
    custom_rules: Optional[Dict[str, Any]] = None

class PatternDetectionRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None

class TrendAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    time_column: Optional[str] = None

class AutoMLRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    feature_columns: Optional[List[str]] = None
    problem_type: str = "auto"
    cv_folds: int = 5

class FeatureEngineeringRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: Optional[str] = None
    create_interactions: bool = True
    create_polynomials: bool = True

class EnsembleRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    feature_columns: Optional[List[str]] = None
    n_models: int = 3

class NLPAnalyzeRequest(BaseModel):
    query: str

class ReportRequest(BaseModel):
    data: List[Dict[str, Any]]
    report_type: str = "comprehensive"
    include_visualizations: bool = True


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


# ============== Advanced AI/ML Endpoints ==============

@advanced_router.post("/analyze")
async def smart_analyze(request: SmartAnalyzeRequest):
    """Intelligent data analysis with automatic approach selection."""
    try:
        result = await ai_analyzer.analyze(
            data=request.data,
            query=request.query,
            analysis_type=request.analysis_type,
            options=request.options
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Smart analyze error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/query")
async def smart_query(request: SmartQueryRequest):
    """Answer natural language questions about data."""
    try:
        result = await ai_analyzer.smart_query(
            data=request.data,
            query=request.query,
            conversation_history=request.conversation_history
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Smart query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/quality")
async def assess_data_quality(request: DataQualityRequest):
    """Comprehensive data quality assessment."""
    try:
        result = analysis_engine.comprehensive_data_quality_assessment(
            data=request.data,
            custom_rules=request.custom_rules
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Quality assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/patterns")
async def detect_patterns(request: PatternDetectionRequest):
    """Detect patterns in data."""
    try:
        result = analysis_engine.detect_patterns(
            data=request.data,
            columns=request.columns
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Pattern detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/trends")
async def analyze_trends(request: TrendAnalysisRequest):
    """Advanced trend analysis with change point detection."""
    try:
        result = analysis_engine.advanced_trend_analysis(
            data=request.data,
            value_column=request.value_column,
            time_column=request.time_column
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/automl")
async def run_automl(request: AutoMLRequest):
    """Run AutoML pipeline with model selection and comparison."""
    try:
        result = ml_workbench.auto_ml_pipeline(
            data=request.data,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            problem_type=request.problem_type,
            cv_folds=request.cv_folds
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"AutoML error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/features")
async def engineer_features(request: FeatureEngineeringRequest):
    """Intelligent feature engineering."""
    try:
        result = ml_workbench.intelligent_feature_engineering(
            data=request.data,
            target_column=request.target_column,
            create_interactions=request.create_interactions,
            create_polynomials=request.create_polynomials
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/ensemble")
async def create_ensemble(request: EnsembleRequest):
    """Create ensemble prediction model."""
    try:
        result = ml_workbench.ensemble_prediction(
            data=request.data,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            n_models=request.n_models
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Ensemble error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/nlp/analyze")
async def analyze_nlp(request: NLPAnalyzeRequest):
    """Analyze natural language query."""
    try:
        result = nlp_engine.analyze_query(request.query)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"NLP analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.post("/report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive analysis report."""
    try:
        result = await ai_analyzer.generate_report(
            data=request.data,
            report_type=request.report_type,
            include_visualizations=request.include_visualizations
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.get("/capabilities")
async def get_capabilities():
    """Get available analysis capabilities."""
    return ai_analyzer.get_analysis_capabilities()


# ============== Shutdown Helper ==============

async def shutdown_db_client() -> None:
    """Close the shared MongoDB client.

    This is intended to be called from the main FastAPI application
    on shutdown.
    """
    client.close()
