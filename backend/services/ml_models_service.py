import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
import logging

logger = logging.getLogger(__name__)


class MLModelsService:
    """Service for machine learning models and predictions."""
    
    @staticmethod
    def train_prediction_model(
        data: List[Dict[str, Any]],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        model_type: str = "auto"
    ) -> Dict[str, Any]:
        """Train a prediction model (classification or regression)."""
        try:
            df = pd.DataFrame(data)
            
            if target_column not in df.columns:
                return {"success": False, "error": f"Target column '{target_column}' not found"}
            
            # Prepare features
            if feature_columns:
                feature_cols = [c for c in feature_columns if c in df.columns and c != target_column]
            else:
                feature_cols = [c for c in df.columns if c != target_column]
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown')
            
            # Encode categorical features
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            # Convert all to numeric
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Determine model type
            y_numeric = pd.to_numeric(y, errors='coerce')
            is_classification = y_numeric.isna().sum() > len(y) * 0.3 or y.nunique() <= 10
            
            if model_type == "auto":
                model_type = "classification" if is_classification else "regression"
            
            # Encode target for classification
            if model_type == "classification":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
                classes = target_encoder.classes_.tolist()
            else:
                y = pd.to_numeric(y, errors='coerce').fillna(y.mean())
                classes = None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                metrics = {
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                }
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                metrics["cv_accuracy"] = round(cv_scores.mean(), 4)
                metrics["cv_std"] = round(cv_scores.std(), 4)
                
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                metrics = {
                    "r2_score": round(r2_score(y_test, y_pred), 4),
                    "mse": round(mean_squared_error(y_test, y_pred), 4),
                    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    "mae": round(mean_absolute_error(y_test, y_pred), 4)
                }
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                metrics["cv_r2"] = round(cv_scores.mean(), 4)
                metrics["cv_std"] = round(cv_scores.std(), 4)
            
            # Feature importance
            feature_importance = sorted(
                zip(feature_cols, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
            feature_importance = [
                {"feature": f, "importance": round(i, 4)}
                for f, i in feature_importance[:15]
            ]
            
            # Sample predictions
            sample_predictions = []
            for i in range(min(10, len(y_test))):
                pred = {
                    "actual": classes[int(y_test.iloc[i])] if classes else round(float(y_test.iloc[i]), 4),
                    "predicted": classes[int(y_pred[i])] if classes else round(float(y_pred[i]), 4)
                }
                sample_predictions.append(pred)
            
            return {
                "success": True,
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "sample_predictions": sample_predictions,
                "classes": classes,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Prediction model error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def perform_clustering(
        data: List[Dict[str, Any]],
        feature_columns: Optional[List[str]] = None,
        n_clusters: Optional[int] = None,
        algorithm: str = "kmeans"
    ) -> Dict[str, Any]:
        """Perform clustering analysis."""
        try:
            df = pd.DataFrame(data)
            
            # Select numeric columns
            if feature_columns:
                cols = [c for c in feature_columns if c in df.columns]
            else:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not cols:
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                    cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(cols) < 2:
                return {"success": False, "error": "Need at least 2 numeric columns for clustering"}
            
            cols = cols[:10]  # Limit columns
            X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Auto-detect optimal clusters using elbow method
            if n_clusters is None and algorithm == "kmeans":
                inertias = []
                K_range = range(2, min(11, len(X) // 5 + 1))
                for k in K_range:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_temp.fit(X_scaled)
                    inertias.append(kmeans_temp.inertia_)
                
                # Find elbow point
                if len(inertias) > 2:
                    diffs = np.diff(inertias)
                    second_diffs = np.diff(diffs)
                    elbow_idx = np.argmax(second_diffs) + 2
                    n_clusters = max(2, min(elbow_idx + 2, 8))
                else:
                    n_clusters = 3
            elif n_clusters is None:
                n_clusters = 3
            
            # Perform clustering
            if algorithm == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                centroids = scaler.inverse_transform(model.cluster_centers_)
            else:  # DBSCAN
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                centroids = None
            
            # Calculate metrics
            if len(set(labels)) > 1:
                silhouette = round(silhouette_score(X_scaled, labels), 4)
                calinski = round(calinski_harabasz_score(X_scaled, labels), 4)
            else:
                silhouette = 0
                calinski = 0
            
            # Cluster statistics
            df['cluster'] = labels
            cluster_stats = []
            for cluster_id in sorted(set(labels)):
                if cluster_id == -1:
                    continue
                cluster_data = df[df['cluster'] == cluster_id]
                stats = {
                    "cluster_id": int(cluster_id),
                    "size": int(len(cluster_data)),
                    "percentage": float(round((len(cluster_data) / len(df)) * 100, 2)),
                    "centroid": {}
                }
                
                for col in cols:
                    col_vals = pd.to_numeric(cluster_data[col], errors='coerce')
                    stats["centroid"][col] = float(round(col_vals.mean(), 4)) if len(col_vals) > 0 else 0.0
                
                cluster_stats.append(stats)
            
            # Scatter data for visualization (using first 2 columns)
            scatter_data = []
            for i in range(min(500, len(X))):
                scatter_data.append({
                    "x": float(round(float(X.iloc[i, 0]), 4)),
                    "y": float(round(float(X.iloc[i, 1]), 4)) if len(cols) > 1 else 0.0,
                    "cluster": int(labels[i])
                })
            
            return {
                "success": True,
                "algorithm": algorithm,
                "n_clusters": int(n_clusters),
                "feature_columns": cols,
                "metrics": {
                    "silhouette_score": float(silhouette),
                    "calinski_harabasz_score": float(calinski)
                },
                "cluster_stats": cluster_stats,
                "scatter_data": scatter_data,
                "x_axis": cols[0],
                "y_axis": cols[1] if len(cols) > 1 else cols[0],
                "labels": [int(l) for l in labels]
            }
            
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def detect_anomalies(
        data: List[Dict[str, Any]],
        feature_columns: Optional[List[str]] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        try:
            df = pd.DataFrame(data)
            
            # Select numeric columns
            if feature_columns:
                cols = [c for c in feature_columns if c in df.columns]
            else:
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not cols:
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                    cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(cols) == 0:
                return {"success": False, "error": "No numeric columns found"}
            
            cols = cols[:10]
            X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            predictions = model.fit_predict(X_scaled)
            scores = model.score_samples(X_scaled)
            
            # Get anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            normal_indices = np.where(predictions == 1)[0]
            
            # Calculate severity based on anomaly scores
            anomalies = []
            for idx in anomaly_indices:
                score = scores[idx]
                # Normalize score to 0-1 range (lower score = more anomalous)
                severity_score = abs(score)
                
                if severity_score > 0.6:
                    severity = "critical"
                elif severity_score > 0.5:
                    severity = "high"
                elif severity_score > 0.4:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Find most anomalous features
                row = X.iloc[idx]
                mean_vals = X.mean()
                std_vals = X.std()
                
                deviations = []
                for col in cols:
                    z_score = abs((row[col] - mean_vals[col]) / std_vals[col]) if std_vals[col] > 0 else 0
                    if z_score > 1.5:
                        deviations.append({
                            "column": col,
                            "value": round(float(row[col]), 4),
                            "z_score": round(z_score, 2)
                        })
                
                deviations.sort(key=lambda x: x["z_score"], reverse=True)
                
                anomalies.append({
                    "index": int(idx),
                    "anomaly_score": round(float(score), 4),
                    "severity": severity,
                    "affected_columns": deviations[:5],
                    "row_data": {col: round(float(row[col]), 4) for col in cols[:5]}
                })
            
            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            anomalies.sort(key=lambda x: (severity_order[x["severity"]], x["anomaly_score"]))
            
            # Summary by severity
            severity_summary = {
                "critical": len([a for a in anomalies if a["severity"] == "critical"]),
                "high": len([a for a in anomalies if a["severity"] == "high"]),
                "medium": len([a for a in anomalies if a["severity"] == "medium"]),
                "low": len([a for a in anomalies if a["severity"] == "low"])
            }
            
            return {
                "success": True,
                "total_records": len(df),
                "anomaly_count": len(anomalies),
                "anomaly_rate": round((len(anomalies) / len(df)) * 100, 2),
                "normal_count": len(normal_indices),
                "feature_columns": cols,
                "contamination": contamination,
                "severity_summary": severity_summary,
                "anomalies": anomalies[:50],  # Limit to top 50
                "scores": [round(float(s), 4) for s in scores]
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {"success": False, "error": str(e)}
