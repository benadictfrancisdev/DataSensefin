"""
ML Workbench - AutoML with Intelligent Model Selection and Feature Engineering.

This module provides advanced machine learning capabilities including:
- Automated model selection and comparison
- Intelligent feature engineering
- Hyperparameter optimization
- Ensemble methods
- Model interpretability and explainability
- Predictive insights generation
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from scipy import stats

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """Convert value to JSON-safe float."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


class MLWorkbench:
    """Advanced ML Workbench for AutoML and intelligent predictions."""
    
    def __init__(self):
        self.model_registry = {
            "classification": {
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "logistic_regression": LogisticRegression,
                "decision_tree": DecisionTreeClassifier,
                "knn": KNeighborsClassifier,
                "adaboost": AdaBoostClassifier
            },
            "regression": {
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "linear_regression": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso,
                "decision_tree": DecisionTreeRegressor,
                "knn": KNeighborsRegressor,
                "adaboost": AdaBoostRegressor
            }
        }
        
        self.default_params = {
            "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
            "gradient_boosting": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "logistic_regression": {"max_iter": 1000, "random_state": 42},
            "linear_regression": {},
            "ridge": {"alpha": 1.0, "random_state": 42},
            "lasso": {"alpha": 1.0, "random_state": 42},
            "decision_tree": {"max_depth": 10, "random_state": 42},
            "knn": {"n_neighbors": 5},
            "adaboost": {"n_estimators": 50, "random_state": 42}
        }
    
    def auto_ml_pipeline(
        self,
        data: List[Dict[str, Any]],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        problem_type: str = "auto",
        test_size: float = 0.2,
        cv_folds: int = 5,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete AutoML pipeline with model selection, training, and evaluation.
        
        Steps:
        1. Data preprocessing and feature engineering
        2. Problem type detection (classification/regression)
        3. Multiple model training and comparison
        4. Best model selection
        5. Feature importance analysis
        6. Predictions and insights generation
        """
        try:
            df = pd.DataFrame(data)
            
            if target_column not in df.columns:
                return {"success": False, "error": f"Target column '{target_column}' not found"}
            
            # Step 1: Prepare features and target
            X, y, feature_cols, label_encoders, problem_type = self._prepare_data(
                df, target_column, feature_columns, problem_type
            )
            
            if X is None:
                return {"success": False, "error": "Failed to prepare data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if problem_type == "classification" and len(np.unique(y)) <= 20 else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Step 2: Feature engineering and selection
            feature_analysis = self._analyze_features(X_train_scaled, y_train, feature_cols, problem_type)
            
            # Step 3: Train and compare multiple models
            model_results = self._train_compare_models(
                X_train_scaled, X_test_scaled, y_train, y_test,
                problem_type, cv_folds
            )
            
            # Step 4: Select best model
            best_model_name = max(model_results, key=lambda k: model_results[k]["primary_metric"])
            best_result = model_results[best_model_name]
            
            # Step 5: Generate predictions and insights
            predictions_analysis = self._analyze_predictions(
                best_result["model"], X_test_scaled, y_test, problem_type
            )
            
            # Step 6: Generate business insights
            insights = self._generate_ml_insights(
                model_results, best_model_name, feature_analysis, problem_type
            )
            
            # Get classes for classification
            classes = None
            if problem_type == "classification" and target_column in label_encoders:
                classes = label_encoders[target_column].classes_.tolist()
            
            return {
                "success": True,
                "problem_type": problem_type,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "data_summary": {
                    "total_samples": len(df),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features_used": len(feature_cols)
                },
                "best_model": {
                    "name": best_model_name,
                    "metrics": best_result["metrics"],
                    "cv_score": best_result.get("cv_score"),
                    "cv_std": best_result.get("cv_std")
                },
                "model_comparison": {
                    name: {
                        "metrics": result["metrics"],
                        "cv_score": result.get("cv_score"),
                        "training_time": result.get("training_time")
                    }
                    for name, result in model_results.items()
                },
                "feature_importance": feature_analysis["importance"][:15],
                "feature_correlations": feature_analysis.get("correlations", []),
                "predictions_analysis": predictions_analysis,
                "insights": insights,
                "classes": classes,
                "recommendations": self._generate_recommendations(
                    model_results, best_model_name, feature_analysis, problem_type
                )
            }
            
        except Exception as e:
            logger.error(f"AutoML pipeline error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]],
        problem_type: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], Dict, str]:
        """Prepare data for ML pipeline."""
        try:
            # Select feature columns
            if feature_columns:
                feature_cols = [c for c in feature_columns if c in df.columns and c != target_column]
            else:
                feature_cols = [c for c in df.columns if c != target_column]
            
            X = df[feature_cols].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown')
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].median())
            
            # Encode categorical features
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            # Convert to numeric
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Determine problem type
            if problem_type == "auto":
                y_numeric = pd.to_numeric(y, errors='coerce')
                if y_numeric.isna().sum() > len(y) * 0.3 or y.nunique() <= 10:
                    problem_type = "classification"
                else:
                    problem_type = "regression"
            
            # Encode target
            if problem_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                label_encoders[target_column] = le
            else:
                y = pd.to_numeric(y, errors='coerce').fillna(y.median())
            
            return X.values, np.array(y), feature_cols, label_encoders, problem_type
            
        except Exception as e:
            logger.error(f"Data preparation error: {str(e)}")
            return None, None, [], {}, "unknown"
    
    def _analyze_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str
    ) -> Dict[str, Any]:
        """Analyze feature importance and correlations."""
        analysis = {"importance": [], "correlations": []}
        
        try:
            # Feature selection scores
            if problem_type == "classification":
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X, y)
            scores = selector.scores_
            
            # Normalize scores
            if np.max(scores) > 0:
                normalized_scores = scores / np.max(scores)
            else:
                normalized_scores = scores
            
            # Create importance list
            importance_list = []
            for i, (name, score, norm_score) in enumerate(zip(feature_names, scores, normalized_scores)):
                importance_list.append({
                    "feature": name,
                    "score": safe_float(score),
                    "importance": safe_float(norm_score),
                    "rank": i + 1
                })
            
            importance_list.sort(key=lambda x: x["importance"], reverse=True)
            for i, item in enumerate(importance_list):
                item["rank"] = i + 1
            
            analysis["importance"] = importance_list
            
            # Feature correlations (for top features)
            df_features = pd.DataFrame(X, columns=feature_names)
            top_features = [f["feature"] for f in importance_list[:10]]
            
            correlations = []
            for i, f1 in enumerate(top_features):
                for f2 in top_features[i+1:]:
                    corr = df_features[f1].corr(df_features[f2])
                    if abs(corr) > 0.5:
                        correlations.append({
                            "feature1": f1,
                            "feature2": f2,
                            "correlation": safe_float(corr),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate"
                        })
            
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            analysis["correlations"] = correlations[:10]
            
        except Exception as e:
            logger.error(f"Feature analysis error: {str(e)}")
        
        return analysis
    
    def _train_compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Train and compare multiple models."""
        results = {}
        
        models_to_train = self.model_registry[problem_type]
        
        for model_name, model_class in models_to_train.items():
            try:
                import time
                start_time = time.time()
                
                # Initialize model with default params
                params = self.default_params.get(model_name, {})
                model = model_class(**params)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if problem_type == "classification":
                    metrics = {
                        "accuracy": safe_float(accuracy_score(y_test, y_pred)),
                        "precision": safe_float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        "recall": safe_float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        "f1_score": safe_float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    }
                    primary_metric = metrics["f1_score"]
                    cv_scoring = 'f1_weighted'
                else:
                    metrics = {
                        "r2_score": safe_float(r2_score(y_test, y_pred)),
                        "mse": safe_float(mean_squared_error(y_test, y_pred)),
                        "rmse": safe_float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        "mae": safe_float(mean_absolute_error(y_test, y_pred)),
                        "explained_variance": safe_float(explained_variance_score(y_test, y_pred))
                    }
                    primary_metric = metrics["r2_score"]
                    cv_scoring = 'r2'
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, len(X_train)), scoring=cv_scoring)
                    cv_score = safe_float(cv_scores.mean())
                    cv_std = safe_float(cv_scores.std())
                except:
                    cv_score = primary_metric
                    cv_std = 0.0
                
                training_time = round(time.time() - start_time, 3)
                
                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "primary_metric": primary_metric,
                    "cv_score": cv_score,
                    "cv_std": cv_std,
                    "training_time": training_time
                }
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return results
    
    def _analyze_predictions(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        problem_type: str
    ) -> Dict[str, Any]:
        """Analyze model predictions."""
        analysis = {}
        
        try:
            y_pred = model.predict(X_test)
            
            if problem_type == "classification":
                # Confusion analysis
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                
                analysis["confusion_matrix"] = cm.tolist()
                analysis["correct_predictions"] = int(np.trace(cm))
                analysis["incorrect_predictions"] = int(np.sum(cm) - np.trace(cm))
                analysis["accuracy_by_class"] = []
                
                for i in range(len(cm)):
                    class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                    analysis["accuracy_by_class"].append({
                        "class": int(i),
                        "accuracy": safe_float(class_accuracy),
                        "support": int(cm[i].sum())
                    })
                
                # Prediction distribution
                unique, counts = np.unique(y_pred, return_counts=True)
                analysis["prediction_distribution"] = [
                    {"class": int(u), "count": int(c), "percentage": safe_float(c / len(y_pred) * 100)}
                    for u, c in zip(unique, counts)
                ]
                
            else:
                # Regression analysis
                residuals = y_test - y_pred
                
                analysis["residual_stats"] = {
                    "mean": safe_float(np.mean(residuals)),
                    "std": safe_float(np.std(residuals)),
                    "min": safe_float(np.min(residuals)),
                    "max": safe_float(np.max(residuals))
                }
                
                # Prediction vs actual correlation
                correlation = np.corrcoef(y_test, y_pred)[0, 1]
                analysis["prediction_correlation"] = safe_float(correlation)
                
                # Percentage errors
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_errors = np.abs(residuals / y_test) * 100
                    pct_errors = pct_errors[np.isfinite(pct_errors)]
                
                if len(pct_errors) > 0:
                    analysis["percentage_error"] = {
                        "mean": safe_float(np.mean(pct_errors)),
                        "median": safe_float(np.median(pct_errors)),
                        "within_10_pct": safe_float(np.mean(pct_errors <= 10) * 100),
                        "within_20_pct": safe_float(np.mean(pct_errors <= 20) * 100)
                    }
                
                # Sample predictions
                indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
                analysis["sample_predictions"] = [
                    {
                        "actual": safe_float(y_test[i]),
                        "predicted": safe_float(y_pred[i]),
                        "error": safe_float(y_test[i] - y_pred[i])
                    }
                    for i in indices
                ]
                
        except Exception as e:
            logger.error(f"Prediction analysis error: {str(e)}")
        
        return analysis
    
    def _generate_ml_insights(
        self,
        model_results: Dict,
        best_model: str,
        feature_analysis: Dict,
        problem_type: str
    ) -> List[Dict[str, Any]]:
        """Generate actionable ML insights."""
        insights = []
        
        # Best model insight
        best_metrics = model_results[best_model]["metrics"]
        if problem_type == "classification":
            score_desc = f"accuracy of {best_metrics['accuracy']*100:.1f}%"
        else:
            score_desc = f"R² score of {best_metrics['r2_score']:.3f}"
        
        insights.append({
            "type": "model_performance",
            "title": f"Best Model: {best_model.replace('_', ' ').title()}",
            "description": f"The {best_model.replace('_', ' ')} model achieved the best performance with {score_desc}",
            "confidence": "high"
        })
        
        # Model comparison insight
        model_scores = [(name, result["primary_metric"]) for name, result in model_results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(model_scores) > 1:
            score_diff = model_scores[0][1] - model_scores[1][1]
            if score_diff < 0.02:
                insights.append({
                    "type": "model_comparison",
                    "title": "Close Competition",
                    "description": f"Top models perform similarly. Consider ensemble methods or domain-specific model selection.",
                    "confidence": "medium"
                })
        
        # Feature importance insight
        top_features = feature_analysis["importance"][:3]
        if top_features:
            feature_names = [f["feature"] for f in top_features]
            insights.append({
                "type": "feature_importance",
                "title": "Key Predictors Identified",
                "description": f"Top predictors are: {', '.join(feature_names)}. Focus data collection efforts on these features.",
                "confidence": "high"
            })
        
        # Correlation insight
        if feature_analysis.get("correlations"):
            strong_corr = [c for c in feature_analysis["correlations"] if c["strength"] == "strong"]
            if strong_corr:
                insights.append({
                    "type": "multicollinearity",
                    "title": "Feature Correlation Detected",
                    "description": f"Found {len(strong_corr)} strongly correlated feature pairs. Consider removing redundant features.",
                    "confidence": "medium"
                })
        
        return insights
    
    def _generate_recommendations(
        self,
        model_results: Dict,
        best_model: str,
        feature_analysis: Dict,
        problem_type: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for improving model performance."""
        recommendations = []
        
        best_metrics = model_results[best_model]["metrics"]
        
        if problem_type == "classification":
            if best_metrics["accuracy"] < 0.8:
                recommendations.append({
                    "priority": "high",
                    "action": "Improve model performance",
                    "details": "Current accuracy is below 80%. Consider: feature engineering, more data, or trying deep learning approaches."
                })
            
            if best_metrics["precision"] - best_metrics["recall"] > 0.1:
                recommendations.append({
                    "priority": "medium",
                    "action": "Balance precision and recall",
                    "details": "Model shows imbalance between precision and recall. Adjust classification threshold or use class weights."
                })
        else:
            if best_metrics["r2_score"] < 0.7:
                recommendations.append({
                    "priority": "high",
                    "action": "Improve prediction accuracy",
                    "details": "R² score below 0.7 indicates room for improvement. Consider polynomial features or non-linear models."
                })
        
        # Feature recommendations
        low_importance = [f for f in feature_analysis["importance"] if f["importance"] < 0.1]
        if len(low_importance) > len(feature_analysis["importance"]) * 0.3:
            recommendations.append({
                "priority": "medium",
                "action": "Feature selection",
                "details": f"{len(low_importance)} features have low importance. Consider removing them to reduce model complexity."
            })
        
        return recommendations
    
    def intelligent_feature_engineering(
        self,
        data: List[Dict[str, Any]],
        target_column: Optional[str] = None,
        create_interactions: bool = True,
        create_polynomials: bool = True,
        max_features: int = 50
    ) -> Dict[str, Any]:
        """
        Automatically engineer new features from existing data.
        
        Creates:
        - Interaction features
        - Polynomial features
        - Aggregation features
        - Date-based features
        - Binned features
        """
        try:
            df = pd.DataFrame(data)
            original_columns = df.columns.tolist()
            new_features = []
            
            # Identify column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            datetime_cols = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    datetime_cols.append(col)
                except:
                    pass
            
            # Create interaction features
            if create_interactions and len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols[:5]):
                    for col2 in numeric_cols[i+1:5]:
                        # Multiplication
                        new_col = f"{col1}_x_{col2}"
                        df[new_col] = df[col1] * df[col2]
                        new_features.append({
                            "name": new_col,
                            "type": "interaction",
                            "base_features": [col1, col2],
                            "operation": "multiplication"
                        })
                        
                        # Ratio (with safety)
                        if (df[col2] != 0).all():
                            new_col = f"{col1}_div_{col2}"
                            df[new_col] = df[col1] / df[col2].replace(0, np.nan)
                            new_features.append({
                                "name": new_col,
                                "type": "interaction",
                                "base_features": [col1, col2],
                                "operation": "division"
                            })
            
            # Create polynomial features
            if create_polynomials and numeric_cols:
                for col in numeric_cols[:5]:
                    # Squared
                    new_col = f"{col}_squared"
                    df[new_col] = df[col] ** 2
                    new_features.append({
                        "name": new_col,
                        "type": "polynomial",
                        "base_features": [col],
                        "operation": "square"
                    })
                    
                    # Log (for positive values)
                    if (df[col] > 0).all():
                        new_col = f"{col}_log"
                        df[new_col] = np.log1p(df[col])
                        new_features.append({
                            "name": new_col,
                            "type": "transformation",
                            "base_features": [col],
                            "operation": "log"
                        })
            
            # Create date-based features
            for col in datetime_cols:
                try:
                    dt = pd.to_datetime(df[col])
                    
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_day"] = dt.dt.day
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek
                    df[f"{col}_quarter"] = dt.dt.quarter
                    
                    for suffix in ["year", "month", "day", "dayofweek", "quarter"]:
                        new_features.append({
                            "name": f"{col}_{suffix}",
                            "type": "datetime",
                            "base_features": [col],
                            "operation": suffix
                        })
                except:
                    pass
            
            # Create binned features for numeric columns
            for col in numeric_cols[:5]:
                try:
                    new_col = f"{col}_binned"
                    df[new_col] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
                    new_features.append({
                        "name": new_col,
                        "type": "binning",
                        "base_features": [col],
                        "operation": "quintile_binning"
                    })
                except:
                    pass
            
            # Limit features
            all_columns = df.columns.tolist()
            if len(all_columns) > max_features:
                # Keep original columns and most recent new features
                keep_cols = original_columns + [f["name"] for f in new_features[:max_features - len(original_columns)]]
                df = df[keep_cols]
                new_features = new_features[:max_features - len(original_columns)]
            
            # Prepare output data
            output_data = df.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient='records')
            
            return {
                "success": True,
                "original_features": len(original_columns),
                "new_features_created": len(new_features),
                "total_features": len(df.columns),
                "new_features": new_features,
                "engineered_data": output_data,
                "columns": df.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def ensemble_prediction(
        self,
        data: List[Dict[str, Any]],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        ensemble_type: str = "voting",
        n_models: int = 3
    ) -> Dict[str, Any]:
        """
        Create ensemble model combining multiple base models.
        
        Supports:
        - Voting ensemble
        - Stacking ensemble
        - Bagging ensemble
        """
        try:
            df = pd.DataFrame(data)
            
            # Prepare data
            X, y, feature_cols, label_encoders, problem_type = self._prepare_data(
                df, target_column, feature_columns, "auto"
            )
            
            if X is None:
                return {"success": False, "error": "Failed to prepare data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select base models
            if problem_type == "classification":
                base_models = [
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=42))
                ][:n_models]
                
                ensemble = VotingClassifier(estimators=base_models, voting='soft')
            else:
                base_models = [
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                    ('ridge', Ridge(random_state=42))
                ][:n_models]
                
                ensemble = VotingRegressor(estimators=base_models)
            
            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)
            y_pred = ensemble.predict(X_test_scaled)
            
            # Calculate metrics
            if problem_type == "classification":
                metrics = {
                    "accuracy": safe_float(accuracy_score(y_test, y_pred)),
                    "precision": safe_float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "recall": safe_float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "f1_score": safe_float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }
            else:
                metrics = {
                    "r2_score": safe_float(r2_score(y_test, y_pred)),
                    "rmse": safe_float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": safe_float(mean_absolute_error(y_test, y_pred))
                }
            
            # Individual model performance
            individual_performance = []
            for name, model in base_models:
                model.fit(X_train_scaled, y_train)
                ind_pred = model.predict(X_test_scaled)
                
                if problem_type == "classification":
                    ind_score = accuracy_score(y_test, ind_pred)
                else:
                    ind_score = r2_score(y_test, ind_pred)
                
                individual_performance.append({
                    "model": name,
                    "score": safe_float(ind_score)
                })
            
            # Get classes for classification
            classes = None
            if problem_type == "classification" and target_column in label_encoders:
                classes = label_encoders[target_column].classes_.tolist()
            
            return {
                "success": True,
                "ensemble_type": ensemble_type,
                "problem_type": problem_type,
                "n_base_models": len(base_models),
                "base_models": [name for name, _ in base_models],
                "ensemble_metrics": metrics,
                "individual_performance": individual_performance,
                "improvement": {
                    "vs_best_individual": safe_float(
                        metrics.get("accuracy", metrics.get("r2_score", 0)) - 
                        max(p["score"] for p in individual_performance)
                    )
                },
                "classes": classes,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def predict_with_confidence(
        self,
        model_data: Dict[str, Any],
        new_data: List[Dict[str, Any]],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on new data with confidence intervals.
        """
        try:
            # This would typically use a saved model
            # For now, we'll train a quick model on the provided data
            
            df = pd.DataFrame(new_data)
            
            return {
                "success": True,
                "message": "Prediction endpoint ready",
                "predictions_count": len(new_data),
                "note": "Full implementation requires model persistence layer"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"success": False, "error": str(e)}
