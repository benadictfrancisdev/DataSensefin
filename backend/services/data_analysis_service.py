import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
import logging
import math

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """Convert value to JSON-safe float, handling NaN and Inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


class DataAnalysisService:
    """Service for comprehensive data analysis and EDA."""
    
    @staticmethod
    def perform_eda(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform automated Exploratory Data Analysis."""
        try:
            df = pd.DataFrame(data)
            
            # Basic info
            basic_info = {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "columns": list(df.columns),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "duplicate_rows": int(df.duplicated().sum())
            }
            
            # Column types analysis
            column_info = []
            numeric_columns = []
            categorical_columns = []
            datetime_columns = []
            
            for col in df.columns:
                col_data = df[col]
                missing_count = col_data.isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                unique_count = col_data.nunique()
                
                # Detect column type
                if pd.api.types.is_numeric_dtype(col_data):
                    col_type = "numeric"
                    numeric_columns.append(col)
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_type = "datetime"
                    datetime_columns.append(col)
                else:
                    # Try to convert to numeric
                    try:
                        numeric_vals = pd.to_numeric(col_data, errors='coerce')
                        if numeric_vals.notna().sum() / len(df) > 0.7:
                            col_type = "numeric"
                            numeric_columns.append(col)
                        else:
                            col_type = "categorical"
                            categorical_columns.append(col)
                    except:
                        col_type = "categorical"
                        categorical_columns.append(col)
                
                column_info.append({
                    "name": col,
                    "type": col_type,
                    "missing_count": int(missing_count),
                    "missing_pct": round(missing_pct, 2),
                    "unique_count": int(unique_count),
                    "unique_pct": round((unique_count / len(df)) * 100, 2)
                })
            
            # Numeric statistics
            numeric_stats = []
            for col in numeric_columns:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    numeric_stats.append({
                        "column": col,
                        "mean": safe_float(col_data.mean()),
                        "median": safe_float(col_data.median()),
                        "std": safe_float(col_data.std()),
                        "min": safe_float(col_data.min()),
                        "max": safe_float(col_data.max()),
                        "q1": safe_float(col_data.quantile(0.25)),
                        "q3": safe_float(col_data.quantile(0.75)),
                        "skewness": safe_float(col_data.skew()),
                        "kurtosis": safe_float(col_data.kurtosis())
                    })
            
            # Categorical statistics
            categorical_stats = []
            for col in categorical_columns[:10]:  # Limit to 10 columns
                value_counts = df[col].value_counts().head(10)
                categorical_stats.append({
                    "column": col,
                    "top_values": [
                        {"value": str(v), "count": int(c), "pct": round((c / len(df)) * 100, 2)}
                        for v, c in value_counts.items()
                    ]
                })
            
            # Data quality score
            total_cells = len(df) * len(df.columns)
            missing_cells = int(df.isna().sum().sum())
            quality_score = round(float((1 - missing_cells / total_cells) * 100), 2)
            
            return {
                "success": True,
                "basic_info": basic_info,
                "column_info": column_info,
                "numeric_stats": numeric_stats,
                "categorical_stats": categorical_stats,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "datetime_columns": datetime_columns,
                "data_quality_score": quality_score
            }
            
        except Exception as e:
            logger.error(f"EDA error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def calculate_correlations(data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate correlation matrix for numeric columns."""
        try:
            df = pd.DataFrame(data)
            
            # Get numeric columns
            if columns:
                numeric_cols = [c for c in columns if c in df.columns]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                # Try to convert columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {"success": False, "error": "Need at least 2 numeric columns"}
            
            # Limit columns for performance
            numeric_cols = numeric_cols[:15]
            df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Calculate correlations
            corr_matrix = df_numeric.corr()
            
            # Find top correlations
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        correlations.append({
                            "column1": numeric_cols[i],
                            "column2": numeric_cols[j],
                            "correlation": round(corr_val, 4),
                            "strength": "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.4 else "weak",
                            "direction": "positive" if corr_val > 0 else "negative"
                        })
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Build matrix for heatmap
            matrix = []
            for col in numeric_cols:
                row = []
                for col2 in numeric_cols:
                    val = corr_matrix.loc[col, col2]
                    row.append(round(val, 4) if not np.isnan(val) else 0)
                matrix.append(row)
            
            return {
                "success": True,
                "columns": numeric_cols,
                "matrix": matrix,
                "top_correlations": correlations[:20],
                "strong_correlations": [c for c in correlations if abs(c["correlation"]) > 0.7]
            }
            
        except Exception as e:
            logger.error(f"Correlation error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def detect_outliers(data: List[Dict[str, Any]], columns: Optional[List[str]] = None, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        try:
            df = pd.DataFrame(data)
            
            if columns:
                numeric_cols = [c for c in columns if c in df.columns]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outliers_by_column = []
            all_outlier_indices = set()
            
            for col in numeric_cols[:10]:
                col_data = pd.to_numeric(df[col], errors='coerce')
                
                if method == "iqr":
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                elif method == "zscore":
                    z_scores = np.abs(stats.zscore(col_data.dropna()))
                    outlier_mask = pd.Series(z_scores > 3, index=col_data.dropna().index)
                    outlier_mask = outlier_mask.reindex(col_data.index, fill_value=False)
                    lower_bound = col_data.mean() - 3 * col_data.std()
                    upper_bound = col_data.mean() + 3 * col_data.std()
                else:
                    continue
                
                outlier_indices = col_data[outlier_mask].index.tolist()
                outlier_values = col_data[outlier_mask].tolist()
                all_outlier_indices.update(outlier_indices)
                
                outliers_by_column.append({
                    "column": col,
                    "outlier_count": len(outlier_indices),
                    "outlier_pct": round((len(outlier_indices) / len(df)) * 100, 2),
                    "lower_bound": safe_float(lower_bound),
                    "upper_bound": safe_float(upper_bound),
                    "outlier_indices": outlier_indices[:50],
                    "outlier_values": [safe_float(v) for v in outlier_values[:50]]
                })
            
            return {
                "success": True,
                "method": method,
                "total_outliers": len(all_outlier_indices),
                "total_outlier_pct": round((len(all_outlier_indices) / len(df)) * 100, 2),
                "outliers_by_column": outliers_by_column,
                "all_outlier_rows": list(all_outlier_indices)[:100]
            }
            
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def get_distribution_analysis(data: List[Dict[str, Any]], column: str) -> Dict[str, Any]:
        """Analyze distribution of a specific column."""
        try:
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found"}
            
            col_data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(col_data) == 0:
                return {"success": False, "error": "No numeric data in column"}
            
            # Basic statistics
            statistics = {
                "count": len(col_data),
                "mean": round(col_data.mean(), 4),
                "median": round(col_data.median(), 4),
                "mode": round(col_data.mode().iloc[0], 4) if len(col_data.mode()) > 0 else None,
                "std": round(col_data.std(), 4),
                "variance": round(col_data.var(), 4),
                "min": round(col_data.min(), 4),
                "max": round(col_data.max(), 4),
                "range": round(col_data.max() - col_data.min(), 4),
                "skewness": round(col_data.skew(), 4),
                "kurtosis": round(col_data.kurtosis(), 4)
            }
            
            # Percentiles
            percentiles = {
                "p5": round(col_data.quantile(0.05), 4),
                "p25": round(col_data.quantile(0.25), 4),
                "p50": round(col_data.quantile(0.50), 4),
                "p75": round(col_data.quantile(0.75), 4),
                "p95": round(col_data.quantile(0.95), 4)
            }
            
            # Normality tests
            if len(col_data) >= 8:
                _, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                is_normal = shapiro_p > 0.05
            else:
                shapiro_p = None
                is_normal = None
            
            # Distribution type detection
            skew = col_data.skew()
            if abs(skew) < 0.5:
                distribution_type = "approximately normal"
            elif skew > 0.5:
                distribution_type = "right-skewed (positive)"
            else:
                distribution_type = "left-skewed (negative)"
            
            # Histogram data
            hist, bin_edges = np.histogram(col_data, bins=30)
            histogram = [
                {"bin_start": round(bin_edges[i], 4), "bin_end": round(bin_edges[i+1], 4), "count": int(hist[i])}
                for i in range(len(hist))
            ]
            
            return {
                "success": True,
                "column": column,
                "statistics": statistics,
                "percentiles": percentiles,
                "normality": {
                    "shapiro_p_value": round(shapiro_p, 4) if shapiro_p else None,
                    "is_normal": is_normal,
                    "distribution_type": distribution_type
                },
                "histogram": histogram
            }
            
        except Exception as e:
            logger.error(f"Distribution analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
