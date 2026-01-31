"""
AI Data Analyzer - Integrated AI-Powered Data Analysis Platform.

This module serves as the orchestration layer that brings together:
- NLP Engine for natural language understanding
- Analysis Engine for pattern detection and data quality
- ML Workbench for predictive modeling
- AI Insights Service for LLM-powered insights

Provides a unified interface for comprehensive, intelligent data analysis.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from services.nlp_engine import NLPEngine
from services.analysis_engine import AnalysisEngine
from services.ml_workbench import MLWorkbench
from services.ai_insights_service import AIInsightsService

load_dotenv()
logger = logging.getLogger(__name__)


class AIDataAnalyzer:
    """
    Integrated AI-powered data analysis platform.
    
    Orchestrates multiple analysis engines to provide comprehensive,
    intelligent insights from data through natural language interaction.
    """
    
    def __init__(self):
        self.nlp_engine = NLPEngine()
        self.analysis_engine = AnalysisEngine()
        self.ml_workbench = MLWorkbench()
        self.ai_service = AIInsightsService()
        
        self.analysis_history = []
        self.data_context_cache = {}
    
    async def analyze(
        self,
        data: List[Dict[str, Any]],
        query: Optional[str] = None,
        analysis_type: str = "auto",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for intelligent data analysis.
        
        Automatically determines the best analysis approach based on:
        - Natural language query (if provided)
        - Data characteristics
        - Analysis type requested
        
        Args:
            data: List of data records
            query: Natural language query/question (optional)
            analysis_type: Type of analysis ('auto', 'comprehensive', 'quick', 'predictive', 'quality')
            options: Additional analysis options
            
        Returns:
            Comprehensive analysis results with insights
        """
        try:
            start_time = datetime.utcnow()
            options = options or {}
            
            df = pd.DataFrame(data)
            columns = df.columns.tolist()
            
            # Extract data context
            data_context = self.nlp_engine.extract_data_context(data, columns)
            
            # Analyze query if provided
            query_analysis = None
            if query:
                query_analysis = self.nlp_engine.analyze_query(query)
            
            # Determine analysis approach
            analysis_plan = self._create_analysis_plan(
                data_context, query_analysis, analysis_type, options
            )
            
            # Execute analysis plan
            results = await self._execute_analysis_plan(
                data, columns, analysis_plan, query, data_context
            )
            
            # Generate unified insights
            unified_insights = await self._generate_unified_insights(
                results, query, data_context
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store in history
            self._store_analysis_history(query, analysis_type, unified_insights)
            
            return {
                "success": True,
                "query": query,
                "analysis_type": analysis_plan["type"],
                "data_summary": {
                    "rows": len(df),
                    "columns": len(columns),
                    "domain": data_context.get("context", {}).get("data_domain", "general")
                },
                "query_understanding": query_analysis if query else None,
                "results": results,
                "insights": unified_insights,
                "execution_time_seconds": round(execution_time, 3),
                "recommendations": self._generate_next_steps(results, unified_insights)
            }
            
        except Exception as e:
            logger.error(f"AI Data Analyzer error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_analysis_plan(
        self,
        data_context: Dict,
        query_analysis: Optional[Dict],
        analysis_type: str,
        options: Dict
    ) -> Dict[str, Any]:
        """Create an analysis plan based on inputs."""
        plan = {
            "type": analysis_type,
            "components": [],
            "priority_areas": []
        }
        
        if analysis_type == "auto":
            # Determine based on query and data
            if query_analysis:
                intent = query_analysis.get("primary_intent", "general")
                
                if intent in ["prediction", "forecast"]:
                    plan["type"] = "predictive"
                    plan["components"] = ["ml_analysis", "trend_analysis"]
                elif intent in ["anomaly", "outlier"]:
                    plan["type"] = "anomaly_detection"
                    plan["components"] = ["quality_assessment", "outlier_detection", "anomaly_ml"]
                elif intent in ["trend", "comparison"]:
                    plan["type"] = "trend_analysis"
                    plan["components"] = ["trend_analysis", "pattern_detection"]
                elif intent in ["correlation", "relationship"]:
                    plan["type"] = "correlation_analysis"
                    plan["components"] = ["correlation_analysis", "pattern_detection"]
                else:
                    plan["type"] = "comprehensive"
                    plan["components"] = ["quick_eda", "quality_assessment", "pattern_detection"]
            else:
                plan["type"] = "comprehensive"
                plan["components"] = ["quick_eda", "quality_assessment", "pattern_detection", "correlation_analysis"]
        
        elif analysis_type == "comprehensive":
            plan["components"] = [
                "quick_eda",
                "quality_assessment",
                "pattern_detection",
                "correlation_analysis",
                "trend_analysis",
                "statistical_tests"
            ]
        
        elif analysis_type == "quick":
            plan["components"] = ["quick_eda", "quality_assessment"]
        
        elif analysis_type == "predictive":
            plan["components"] = ["ml_analysis", "feature_engineering"]
            if options.get("target_column"):
                plan["target_column"] = options["target_column"]
        
        elif analysis_type == "quality":
            plan["components"] = ["quality_assessment", "pattern_detection"]
        
        # Add priority areas from data context
        context = data_context.get("context", {})
        if context.get("key_metrics"):
            plan["priority_areas"].extend(context["key_metrics"][:3])
        
        return plan
    
    async def _execute_analysis_plan(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        plan: Dict,
        query: Optional[str],
        data_context: Dict
    ) -> Dict[str, Any]:
        """Execute the analysis plan and collect results."""
        results = {}
        
        for component in plan["components"]:
            try:
                if component == "quick_eda":
                    from services.data_analysis_service import DataAnalysisService
                    results["eda"] = DataAnalysisService.perform_eda(data)
                
                elif component == "quality_assessment":
                    results["quality"] = self.analysis_engine.comprehensive_data_quality_assessment(data)
                
                elif component == "pattern_detection":
                    results["patterns"] = self.analysis_engine.detect_patterns(data, columns)
                
                elif component == "correlation_analysis":
                    from services.data_analysis_service import DataAnalysisService
                    results["correlations"] = DataAnalysisService.calculate_correlations(data)
                
                elif component == "trend_analysis":
                    # Find numeric columns for trend analysis
                    context = data_context.get("context", {})
                    metrics = context.get("key_metrics", [])
                    if metrics:
                        results["trends"] = self.analysis_engine.advanced_trend_analysis(
                            data, metrics[0]
                        )
                
                elif component == "statistical_tests":
                    context = data_context.get("context", {})
                    metrics = context.get("key_metrics", [])
                    if len(metrics) >= 2:
                        results["statistical_tests"] = self.analysis_engine.statistical_hypothesis_testing(
                            data, metrics[0], metrics[1]
                        )
                
                elif component == "ml_analysis":
                    target = plan.get("target_column")
                    if target:
                        results["ml"] = self.ml_workbench.auto_ml_pipeline(
                            data, target, cv_folds=3
                        )
                
                elif component == "feature_engineering":
                    results["features"] = self.ml_workbench.intelligent_feature_engineering(data)
                
                elif component == "outlier_detection":
                    from services.data_analysis_service import DataAnalysisService
                    results["outliers"] = DataAnalysisService.detect_outliers(data)
                
                elif component == "anomaly_ml":
                    from services.ml_models_service import MLModelsService
                    results["anomalies"] = MLModelsService.detect_anomalies(data)
                    
            except Exception as e:
                logger.warning(f"Component {component} failed: {str(e)}")
                results[component] = {"success": False, "error": str(e)}
        
        return results
    
    async def _generate_unified_insights(
        self,
        results: Dict,
        query: Optional[str],
        data_context: Dict
    ) -> Dict[str, Any]:
        """Generate unified insights from all analysis results."""
        insights = {
            "key_findings": [],
            "data_health": None,
            "patterns_discovered": [],
            "recommendations": [],
            "ai_summary": None
        }
        
        # Extract key findings from each result
        if "eda" in results and results["eda"].get("success"):
            eda = results["eda"]
            insights["key_findings"].append({
                "category": "data_overview",
                "finding": f"Dataset contains {eda['basic_info']['total_rows']} rows and {eda['basic_info']['total_columns']} columns",
                "importance": "info"
            })
            
            if eda.get("numeric_stats"):
                for stat in eda["numeric_stats"][:2]:
                    if stat.get("skewness", 0) > 1:
                        insights["key_findings"].append({
                            "category": "distribution",
                            "finding": f"Column '{stat['column']}' is significantly right-skewed (skewness: {stat['skewness']:.2f})",
                            "importance": "medium"
                        })
        
        if "quality" in results and results["quality"].get("success"):
            quality = results["quality"]
            insights["data_health"] = {
                "score": quality["overall_score"],
                "grade": quality["grade"],
                "critical_issues": len([i for i in quality.get("issues", []) if i.get("severity") == "critical"]),
                "total_issues": quality.get("total_issues", 0)
            }
            
            if quality["overall_score"] < 80:
                insights["key_findings"].append({
                    "category": "quality",
                    "finding": f"Data quality score is {quality['overall_score']:.1f}/100 - {quality['grade_description']}",
                    "importance": "high"
                })
        
        if "patterns" in results and results["patterns"].get("success"):
            patterns = results["patterns"]
            if patterns.get("patterns", {}).get("correlation_patterns"):
                for corr in patterns["patterns"]["correlation_patterns"][:2]:
                    insights["patterns_discovered"].append({
                        "type": "correlation",
                        "description": corr["description"],
                        "confidence": corr.get("confidence", 0.8)
                    })
            
            if patterns.get("patterns", {}).get("distribution_patterns"):
                for dist in patterns["patterns"]["distribution_patterns"][:2]:
                    if dist.get("distribution") in ["right_skewed", "left_skewed", "multimodal"]:
                        insights["patterns_discovered"].append({
                            "type": "distribution",
                            "description": f"Column '{dist['column']}' has {dist['description']}",
                            "confidence": dist.get("confidence", 0.7)
                        })
        
        if "correlations" in results and results["correlations"].get("success"):
            corr = results["correlations"]
            strong = corr.get("strong_correlations", [])
            if strong:
                insights["key_findings"].append({
                    "category": "relationships",
                    "finding": f"Found {len(strong)} strong correlation(s) between variables",
                    "importance": "high"
                })
        
        if "ml" in results and results["ml"].get("success"):
            ml = results["ml"]
            best = ml.get("best_model", {})
            insights["key_findings"].append({
                "category": "predictive",
                "finding": f"Best predictive model: {best.get('name', 'Unknown')} with score {list(best.get('metrics', {}).values())[0] if best.get('metrics') else 'N/A'}",
                "importance": "high"
            })
        
        # Generate AI summary if available
        if self.ai_service.api_key:
            try:
                # Prepare summary for AI
                summary_data = {
                    "findings": insights["key_findings"][:5],
                    "patterns": insights["patterns_discovered"][:3],
                    "health_score": insights.get("data_health", {}).get("score")
                }
                
                ai_result = await self.ai_service.explain_analysis(
                    "comprehensive_analysis",
                    summary_data,
                    f"Data domain: {data_context.get('context', {}).get('data_domain', 'general')}"
                )
                
                if ai_result.get("success"):
                    insights["ai_summary"] = ai_result.get("explanation")
                    
            except Exception as e:
                logger.warning(f"AI summary generation failed: {str(e)}")
        
        # Add recommendations
        if insights.get("data_health", {}).get("score", 100) < 80:
            insights["recommendations"].append({
                "priority": "high",
                "action": "Improve data quality",
                "reason": "Data quality score below acceptable threshold"
            })
        
        if len(insights["patterns_discovered"]) > 0:
            insights["recommendations"].append({
                "priority": "medium",
                "action": "Investigate discovered patterns",
                "reason": "Significant patterns found that may indicate business opportunities or issues"
            })
        
        return insights
    
    def _generate_next_steps(
        self,
        results: Dict,
        insights: Dict
    ) -> List[Dict[str, str]]:
        """Generate recommended next steps based on analysis."""
        next_steps = []
        
        # Based on quality
        if insights.get("data_health", {}).get("score", 100) < 70:
            next_steps.append({
                "step": "Data Cleaning",
                "description": "Address data quality issues before further analysis",
                "priority": "high"
            })
        
        # Based on patterns
        if insights.get("patterns_discovered"):
            next_steps.append({
                "step": "Deep Dive Analysis",
                "description": "Investigate discovered patterns with domain experts",
                "priority": "medium"
            })
        
        # Based on ML readiness
        if "ml" not in results and insights.get("data_health", {}).get("score", 0) > 80:
            next_steps.append({
                "step": "Predictive Modeling",
                "description": "Data quality supports building predictive models",
                "priority": "medium"
            })
        
        # General recommendations
        next_steps.append({
            "step": "Share Insights",
            "description": "Review findings with stakeholders for business context",
            "priority": "low"
        })
        
        return next_steps
    
    def _store_analysis_history(
        self,
        query: Optional[str],
        analysis_type: str,
        insights: Dict
    ):
        """Store analysis in history for context."""
        self.analysis_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "analysis_type": analysis_type,
            "key_findings_count": len(insights.get("key_findings", [])),
            "patterns_count": len(insights.get("patterns_discovered", []))
        })
        
        # Keep only last 20 analyses
        if len(self.analysis_history) > 20:
            self.analysis_history = self.analysis_history[-20:]
    
    async def smart_query(
        self,
        data: List[Dict[str, Any]],
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Answer natural language questions about the data.
        
        Combines NLP understanding with data analysis to provide
        intelligent, context-aware responses.
        """
        try:
            columns = list(data[0].keys()) if data else []
            
            # Analyze the query
            query_analysis = self.nlp_engine.analyze_query(query)
            
            # Get data context
            data_context = self.nlp_engine.extract_data_context(data, columns)
            
            # Generate structured query
            structured_query = await self.nlp_engine.generate_smart_query(
                query, data_context, columns
            )
            
            # Execute appropriate analysis based on intent
            intent = query_analysis.get("primary_intent", "general")
            
            analysis_results = {}
            
            if intent in ["aggregation", "summary", "grouping"]:
                # Perform aggregation analysis
                df = pd.DataFrame(data)
                
                metrics = data_context.get("context", {}).get("key_metrics", [])
                if metrics:
                    for metric in metrics[:3]:
                        if metric in df.columns:
                            numeric_data = pd.to_numeric(df[metric], errors='coerce')
                            analysis_results[metric] = {
                                "sum": float(numeric_data.sum()),
                                "mean": float(numeric_data.mean()),
                                "median": float(numeric_data.median()),
                                "min": float(numeric_data.min()),
                                "max": float(numeric_data.max())
                            }
            
            elif intent in ["trend", "comparison"]:
                # Perform trend analysis
                metrics = data_context.get("context", {}).get("key_metrics", [])
                if metrics:
                    analysis_results["trend"] = self.analysis_engine.advanced_trend_analysis(
                        data, metrics[0]
                    )
            
            elif intent in ["correlation", "relationship"]:
                # Perform correlation analysis
                from services.data_analysis_service import DataAnalysisService
                analysis_results["correlations"] = DataAnalysisService.calculate_correlations(data)
            
            elif intent in ["anomaly", "outlier"]:
                # Detect anomalies
                from services.ml_models_service import MLModelsService
                analysis_results["anomalies"] = MLModelsService.detect_anomalies(data)
            
            else:
                # General analysis
                from services.data_analysis_service import DataAnalysisService
                analysis_results["eda"] = DataAnalysisService.perform_eda(data)
            
            # Get AI-powered answer
            ai_answer = None
            if self.ai_service.api_key:
                ai_result = await self.ai_service.answer_query(
                    data, columns, query, conversation_history
                )
                if ai_result.get("success"):
                    ai_answer = ai_result.get("answer")
            
            return {
                "success": True,
                "query": query,
                "query_understanding": {
                    "intent": intent,
                    "entities": query_analysis.get("entities", {}),
                    "confidence": query_analysis.get("confidence", 0.5)
                },
                "structured_interpretation": structured_query.get("structured_query") if structured_query.get("success") else None,
                "analysis_results": analysis_results,
                "ai_answer": ai_answer,
                "suggested_followups": self._generate_followup_questions(intent, data_context)
            }
            
        except Exception as e:
            logger.error(f"Smart query error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_followup_questions(
        self,
        intent: str,
        data_context: Dict
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        followups = []
        context = data_context.get("context", {})
        metrics = context.get("key_metrics", [])
        dimensions = context.get("categorical_dimensions", [])
        
        if intent == "aggregation":
            if dimensions:
                followups.append(f"How does this vary by {dimensions[0]}?")
            followups.append("What's the trend over time?")
        
        elif intent == "trend":
            followups.append("Are there any anomalies in this trend?")
            followups.append("What factors might be driving this trend?")
        
        elif intent == "correlation":
            followups.append("Which variables have the strongest relationships?")
            followups.append("Can we predict one variable from another?")
        
        else:
            if metrics:
                followups.append(f"What's the average {metrics[0]}?")
            followups.append("What patterns exist in this data?")
            followups.append("Are there any data quality issues?")
        
        return followups[:3]
    
    async def generate_report(
        self,
        data: List[Dict[str, Any]],
        report_type: str = "comprehensive",
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            data: Data to analyze
            report_type: Type of report ('comprehensive', 'executive', 'technical')
            include_visualizations: Whether to include chart configurations
            
        Returns:
            Structured report with all analysis components
        """
        try:
            # Run comprehensive analysis
            analysis = await self.analyze(data, analysis_type="comprehensive")
            
            if not analysis.get("success"):
                return analysis
            
            report = {
                "success": True,
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": report_type,
                "sections": []
            }
            
            # Executive Summary
            if report_type in ["comprehensive", "executive"]:
                report["sections"].append({
                    "title": "Executive Summary",
                    "content": {
                        "data_overview": analysis["data_summary"],
                        "key_findings": analysis["insights"]["key_findings"][:5],
                        "health_score": analysis["insights"].get("data_health", {}).get("score"),
                        "ai_summary": analysis["insights"].get("ai_summary")
                    }
                })
            
            # Data Quality Section
            if analysis["results"].get("quality", {}).get("success"):
                quality = analysis["results"]["quality"]
                report["sections"].append({
                    "title": "Data Quality Assessment",
                    "content": {
                        "overall_score": quality["overall_score"],
                        "grade": quality["grade"],
                        "dimensions": quality["dimensions"],
                        "top_issues": quality.get("issues", [])[:5],
                        "recommendations": quality.get("recommendations", [])
                    }
                })
            
            # Statistical Analysis
            if report_type in ["comprehensive", "technical"]:
                if analysis["results"].get("eda", {}).get("success"):
                    eda = analysis["results"]["eda"]
                    report["sections"].append({
                        "title": "Statistical Analysis",
                        "content": {
                            "numeric_summary": eda.get("numeric_stats", [])[:10],
                            "categorical_summary": eda.get("categorical_stats", [])[:10],
                            "column_types": {
                                "numeric": len(eda.get("numeric_columns", [])),
                                "categorical": len(eda.get("categorical_columns", [])),
                                "datetime": len(eda.get("datetime_columns", []))
                            }
                        }
                    })
            
            # Pattern Discovery
            if analysis["results"].get("patterns", {}).get("success"):
                patterns = analysis["results"]["patterns"]
                report["sections"].append({
                    "title": "Pattern Discovery",
                    "content": {
                        "total_patterns": patterns["total_patterns_found"],
                        "distribution_patterns": patterns["patterns"].get("distribution_patterns", []),
                        "correlation_patterns": patterns["patterns"].get("correlation_patterns", []),
                        "insights": patterns.get("insights", [])
                    }
                })
            
            # Visualizations
            if include_visualizations:
                report["visualizations"] = self._generate_visualization_configs(analysis)
            
            # Recommendations
            report["sections"].append({
                "title": "Recommendations & Next Steps",
                "content": {
                    "recommendations": analysis["insights"].get("recommendations", []),
                    "next_steps": analysis.get("recommendations", [])
                }
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_visualization_configs(self, analysis: Dict) -> List[Dict]:
        """Generate visualization configurations based on analysis."""
        visualizations = []
        
        results = analysis.get("results", {})
        
        # Quality score gauge
        if results.get("quality", {}).get("success"):
            visualizations.append({
                "type": "gauge",
                "title": "Data Quality Score",
                "data": {
                    "value": results["quality"]["overall_score"],
                    "min": 0,
                    "max": 100,
                    "thresholds": [60, 80, 90]
                }
            })
        
        # Correlation heatmap
        if results.get("correlations", {}).get("success"):
            corr = results["correlations"]
            if corr.get("matrix"):
                visualizations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data": {
                        "matrix": corr["matrix"],
                        "labels": corr["columns"]
                    }
                })
        
        # Distribution charts
        if results.get("eda", {}).get("success"):
            eda = results["eda"]
            for stat in eda.get("numeric_stats", [])[:3]:
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution: {stat['column']}",
                    "data": {
                        "column": stat["column"],
                        "mean": stat["mean"],
                        "median": stat["median"],
                        "std": stat["std"]
                    }
                })
        
        return visualizations
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Return available analysis capabilities."""
        return {
            "analysis_types": [
                {
                    "type": "comprehensive",
                    "description": "Full analysis including EDA, quality assessment, patterns, and correlations"
                },
                {
                    "type": "quick",
                    "description": "Fast overview with basic statistics and quality check"
                },
                {
                    "type": "predictive",
                    "description": "Machine learning analysis with model comparison and predictions"
                },
                {
                    "type": "quality",
                    "description": "Deep data quality assessment with recommendations"
                },
                {
                    "type": "auto",
                    "description": "Automatically determines best analysis based on query and data"
                }
            ],
            "nlp_capabilities": [
                "Natural language query understanding",
                "Intent classification",
                "Entity extraction",
                "Sentiment analysis",
                "Query-to-analysis translation"
            ],
            "ml_capabilities": [
                "AutoML with model selection",
                "Feature engineering",
                "Ensemble methods",
                "Classification and regression",
                "Anomaly detection",
                "Clustering"
            ],
            "supported_intents": list(self.nlp_engine._intent_patterns.keys())
        }
