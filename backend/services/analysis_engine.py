"""
Analysis Engine - Advanced Pattern Detection and Data Quality Assessment.

This module provides sophisticated data analysis capabilities including:
- Deep pattern detection and recognition
- Comprehensive data quality scoring
- Advanced trend analysis and decomposition
- Statistical hypothesis testing
- Automated insight generation
- Data profiling and anomaly scoring
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def safe_value(value, default=0.0):
    """Convert value to JSON-safe format."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


class AnalysisEngine:
    """Advanced Analysis Engine for deep pattern detection and data quality assessment."""
    
    def __init__(self):
        self.quality_weights = {
            "completeness": 0.25,
            "uniqueness": 0.15,
            "validity": 0.20,
            "consistency": 0.20,
            "accuracy": 0.20
        }
    
    def comprehensive_data_quality_assessment(
        self,
        data: List[Dict[str, Any]],
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment with detailed scoring.
        
        Returns quality scores, issues, and recommendations.
        """
        try:
            df = pd.DataFrame(data)
            
            # Individual quality dimensions
            completeness = self._assess_completeness(df)
            uniqueness = self._assess_uniqueness(df)
            validity = self._assess_validity(df)
            consistency = self._assess_consistency(df)
            accuracy = self._assess_accuracy(df)
            
            # Calculate overall score
            overall_score = (
                completeness["score"] * self.quality_weights["completeness"] +
                uniqueness["score"] * self.quality_weights["uniqueness"] +
                validity["score"] * self.quality_weights["validity"] +
                consistency["score"] * self.quality_weights["consistency"] +
                accuracy["score"] * self.quality_weights["accuracy"]
            )
            
            # Grade assignment
            if overall_score >= 90:
                grade = "A"
                grade_description = "Excellent data quality"
            elif overall_score >= 80:
                grade = "B"
                grade_description = "Good data quality with minor issues"
            elif overall_score >= 70:
                grade = "C"
                grade_description = "Acceptable quality, improvements recommended"
            elif overall_score >= 60:
                grade = "D"
                grade_description = "Poor quality, significant issues present"
            else:
                grade = "F"
                grade_description = "Critical quality issues, immediate attention required"
            
            # Collect all issues
            all_issues = []
            all_issues.extend(completeness.get("issues", []))
            all_issues.extend(uniqueness.get("issues", []))
            all_issues.extend(validity.get("issues", []))
            all_issues.extend(consistency.get("issues", []))
            all_issues.extend(accuracy.get("issues", []))
            
            # Sort issues by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            all_issues.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 4))
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                completeness, uniqueness, validity, consistency, accuracy, all_issues
            )
            
            # Column-level quality
            column_quality = self._assess_column_quality(df)
            
            return {
                "success": True,
                "overall_score": round(overall_score, 2),
                "grade": grade,
                "grade_description": grade_description,
                "dimensions": {
                    "completeness": completeness,
                    "uniqueness": uniqueness,
                    "validity": validity,
                    "consistency": consistency,
                    "accuracy": accuracy
                },
                "total_issues": len(all_issues),
                "issues": all_issues[:20],
                "recommendations": recommendations,
                "column_quality": column_quality[:15],
                "summary": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "total_cells": len(df) * len(df.columns),
                    "missing_cells": int(df.isna().sum().sum()),
                    "duplicate_rows": int(df.duplicated().sum())
                }
            }
            
        except Exception as e:
            logger.error(f"Data quality assessment error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness dimension."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        issues = []
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                issues.append({
                    "type": "missing_data",
                    "severity": "critical",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "impact": "May significantly affect analysis reliability"
                })
            elif missing_pct > 20:
                issues.append({
                    "type": "missing_data",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "impact": "Could affect analysis accuracy"
                })
            elif missing_pct > 5:
                issues.append({
                    "type": "missing_data",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "impact": "Minor impact on analysis"
                })
        
        return {
            "score": round(completeness_score, 2),
            "missing_cells": int(missing_cells),
            "missing_percentage": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
            "issues": issues
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness dimension."""
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 100
        
        issues = []
        if duplicate_rows > len(df) * 0.1:
            issues.append({
                "type": "duplicate_rows",
                "severity": "high",
                "message": f"Dataset contains {duplicate_rows} duplicate rows ({(duplicate_rows/len(df)*100):.1f}%)",
                "impact": "May cause incorrect aggregations and counts"
            })
        elif duplicate_rows > 0:
            issues.append({
                "type": "duplicate_rows",
                "severity": "medium",
                "message": f"Dataset contains {duplicate_rows} duplicate rows",
                "impact": "Consider reviewing if duplicates are intentional"
            })
        
        # Check for potential ID columns with duplicates
        for col in df.columns:
            if any(kw in col.lower() for kw in ['id', 'key', 'code', 'uuid']):
                if df[col].duplicated().any():
                    dup_count = df[col].duplicated().sum()
                    issues.append({
                        "type": "duplicate_identifier",
                        "severity": "critical",
                        "column": col,
                        "message": f"Potential identifier column '{col}' has {dup_count} duplicate values",
                        "impact": "May indicate data integrity issues"
                    })
        
        return {
            "score": round(uniqueness_score, 2),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": round((duplicate_rows / len(df)) * 100, 2) if len(df) > 0 else 0,
            "issues": issues
        }
    
    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity dimension."""
        issues = []
        validity_scores = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for mixed types
            types = col_data.apply(type).unique()
            if len(types) > 1:
                issues.append({
                    "type": "mixed_types",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' contains mixed data types",
                    "impact": "May cause type conversion issues"
                })
                validity_scores.append(80)
            else:
                validity_scores.append(100)
            
            # Check numeric columns for invalid values
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            if numeric_data.notna().sum() > len(col_data) * 0.5:
                # Likely numeric column
                if (numeric_data < 0).any() and any(kw in col.lower() for kw in ['count', 'quantity', 'age', 'price']):
                    neg_count = (numeric_data < 0).sum()
                    issues.append({
                        "type": "invalid_values",
                        "severity": "high",
                        "column": col,
                        "message": f"Column '{col}' has {neg_count} negative values (unexpected for this field type)",
                        "impact": "May indicate data entry errors"
                    })
        
        validity_score = np.mean(validity_scores) if validity_scores else 100
        
        return {
            "score": round(validity_score, 2),
            "issues": issues
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency dimension."""
        issues = []
        consistency_scores = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check string columns for format inconsistencies
            if col_data.dtype == 'object':
                # Check case consistency
                has_upper = col_data.str.isupper().any() if hasattr(col_data.str, 'isupper') else False
                has_lower = col_data.str.islower().any() if hasattr(col_data.str, 'islower') else False
                has_title = col_data.str.istitle().any() if hasattr(col_data.str, 'istitle') else False
                
                case_variations = sum([has_upper, has_lower, has_title])
                if case_variations > 1:
                    issues.append({
                        "type": "inconsistent_format",
                        "severity": "low",
                        "column": col,
                        "message": f"Column '{col}' has inconsistent text casing",
                        "impact": "May affect grouping and matching operations"
                    })
                    consistency_scores.append(90)
                else:
                    consistency_scores.append(100)
                
                # Check for leading/trailing whitespace
                if col_data.str.strip().ne(col_data).any():
                    issues.append({
                        "type": "whitespace_issues",
                        "severity": "low",
                        "column": col,
                        "message": f"Column '{col}' has values with leading/trailing whitespace",
                        "impact": "May cause matching failures"
                    })
        
        consistency_score = np.mean(consistency_scores) if consistency_scores else 100
        
        return {
            "score": round(consistency_score, 2),
            "issues": issues
        }
    
    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data accuracy dimension using statistical methods."""
        issues = []
        accuracy_scores = []
        
        for col in df.columns:
            numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(numeric_data) < 10:
                continue
            
            # Detect statistical outliers using IQR
            q1, q3 = numeric_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (numeric_data < q1 - 3*iqr) | (numeric_data > q3 + 3*iqr)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > len(numeric_data) * 0.05:
                issues.append({
                    "type": "potential_outliers",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' has {outlier_count} extreme outliers ({(outlier_count/len(numeric_data)*100):.1f}%)",
                    "impact": "May indicate data entry errors or require investigation"
                })
                accuracy_scores.append(85)
            else:
                accuracy_scores.append(100)
        
        accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 100
        
        return {
            "score": round(accuracy_score, 2),
            "issues": issues
        }
    
    def _assess_column_quality(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Assess quality at column level."""
        column_quality = []
        
        for col in df.columns:
            col_data = df[col]
            missing_pct = (col_data.isna().sum() / len(df)) * 100
            unique_pct = (col_data.nunique() / len(df)) * 100
            
            # Calculate column score
            completeness = 100 - missing_pct
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                dtype = "datetime"
            else:
                dtype = "text"
            
            score = completeness * 0.7 + min(unique_pct, 100) * 0.3
            
            column_quality.append({
                "column": col,
                "quality_score": round(score, 2),
                "data_type": dtype,
                "completeness": round(completeness, 2),
                "unique_values": col_data.nunique(),
                "missing_count": int(col_data.isna().sum()),
                "missing_percentage": round(missing_pct, 2)
            })
        
        column_quality.sort(key=lambda x: x["quality_score"])
        return column_quality
    
    def _generate_quality_recommendations(
        self,
        completeness: Dict,
        uniqueness: Dict,
        validity: Dict,
        consistency: Dict,
        accuracy: Dict,
        issues: List
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []
        
        if completeness["score"] < 90:
            recommendations.append({
                "priority": "high",
                "category": "completeness",
                "recommendation": "Address missing data",
                "details": "Consider imputation strategies or investigate data collection process",
                "expected_impact": f"Could improve overall quality score by up to {min(10, 100-completeness['score']):.0f} points"
            })
        
        if uniqueness["duplicate_rows"] > 0:
            recommendations.append({
                "priority": "high" if uniqueness["score"] < 95 else "medium",
                "category": "uniqueness",
                "recommendation": "Review and remove duplicate records",
                "details": f"Found {uniqueness['duplicate_rows']} duplicate rows that may affect aggregations",
                "expected_impact": "Ensures accurate counts and aggregations"
            })
        
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            recommendations.append({
                "priority": "critical",
                "category": "data_integrity",
                "recommendation": "Address critical data issues immediately",
                "details": f"Found {len(critical_issues)} critical issues affecting data integrity",
                "expected_impact": "Prevents incorrect analysis results"
            })
        
        if consistency["score"] < 95:
            recommendations.append({
                "priority": "medium",
                "category": "consistency",
                "recommendation": "Standardize data formats",
                "details": "Implement consistent formatting for text fields and dates",
                "expected_impact": "Improves data matching and grouping accuracy"
            })
        
        return recommendations
    
    def detect_patterns(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect various patterns in the data including:
        - Temporal patterns (seasonality, trends)
        - Value distributions and clusters
        - Correlation patterns
        - Sequence patterns
        """
        try:
            df = pd.DataFrame(data)
            
            if columns:
                cols = [c for c in columns if c in df.columns]
            else:
                cols = df.columns.tolist()
            
            patterns = {
                "temporal_patterns": [],
                "distribution_patterns": [],
                "correlation_patterns": [],
                "value_patterns": [],
                "sequence_patterns": []
            }
            
            numeric_cols = []
            for col in cols:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                if numeric_data.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
                    
                    # Distribution patterns
                    dist_pattern = self._detect_distribution_pattern(numeric_data.dropna(), col)
                    if dist_pattern:
                        patterns["distribution_patterns"].append(dist_pattern)
                    
                    # Value patterns
                    value_pattern = self._detect_value_patterns(numeric_data.dropna(), col)
                    if value_pattern:
                        patterns["value_patterns"].append(value_pattern)
            
            # Correlation patterns
            if len(numeric_cols) >= 2:
                corr_patterns = self._detect_correlation_patterns(df, numeric_cols)
                patterns["correlation_patterns"] = corr_patterns
            
            # Categorical patterns
            cat_cols = [c for c in cols if c not in numeric_cols]
            for col in cat_cols[:5]:
                seq_pattern = self._detect_categorical_patterns(df[col], col)
                if seq_pattern:
                    patterns["sequence_patterns"].append(seq_pattern)
            
            # Summary
            total_patterns = sum(len(v) for v in patterns.values())
            
            return {
                "success": True,
                "total_patterns_found": total_patterns,
                "patterns": patterns,
                "columns_analyzed": len(cols),
                "insights": self._generate_pattern_insights(patterns)
            }
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _detect_distribution_pattern(self, data: pd.Series, col: str) -> Optional[Dict]:
        """Detect distribution patterns in numeric data."""
        if len(data) < 10:
            return None
        
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Determine distribution type
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            dist_type = "normal"
            description = "approximately normally distributed"
        elif skewness > 1:
            dist_type = "right_skewed"
            description = "right-skewed (long tail on right)"
        elif skewness < -1:
            dist_type = "left_skewed"
            description = "left-skewed (long tail on left)"
        elif kurtosis > 3:
            dist_type = "heavy_tailed"
            description = "heavy-tailed (more extreme values than normal)"
        elif kurtosis < -1:
            dist_type = "light_tailed"
            description = "light-tailed (fewer extreme values than normal)"
        else:
            dist_type = "moderate"
            description = "moderately distributed"
        
        # Check for multimodality using peaks
        hist, bin_edges = np.histogram(data, bins=20)
        peaks, _ = find_peaks(hist, height=len(data) * 0.05)
        
        if len(peaks) > 1:
            dist_type = "multimodal"
            description = f"multimodal distribution with {len(peaks)} apparent modes"
        
        return {
            "column": col,
            "pattern_type": "distribution",
            "distribution": dist_type,
            "description": description,
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "modes": len(peaks),
            "confidence": 0.85 if len(data) > 100 else 0.65
        }
    
    def _detect_value_patterns(self, data: pd.Series, col: str) -> Optional[Dict]:
        """Detect value patterns like concentration, gaps, etc."""
        if len(data) < 10:
            return None
        
        patterns_found = []
        
        # Check for value concentration
        top_5_pct = data.nlargest(int(len(data) * 0.05)).sum() / data.sum() if data.sum() > 0 else 0
        if top_5_pct > 0.5:
            patterns_found.append({
                "type": "concentration",
                "description": f"Top 5% of values account for {top_5_pct*100:.1f}% of total"
            })
        
        # Check for round number preference
        rounded = (data % 10 == 0).sum() / len(data)
        if rounded > 0.5:
            patterns_found.append({
                "type": "round_number_bias",
                "description": f"{rounded*100:.1f}% of values are round numbers (divisible by 10)"
            })
        
        # Check for boundary clustering
        near_min = ((data - data.min()) / (data.max() - data.min()) < 0.1).sum() / len(data) if (data.max() - data.min()) > 0 else 0
        near_max = ((data.max() - data) / (data.max() - data.min()) < 0.1).sum() / len(data) if (data.max() - data.min()) > 0 else 0
        
        if near_min > 0.3 or near_max > 0.3:
            patterns_found.append({
                "type": "boundary_clustering",
                "description": f"Values cluster near {'minimum' if near_min > near_max else 'maximum'}"
            })
        
        if patterns_found:
            return {
                "column": col,
                "pattern_type": "value_patterns",
                "patterns": patterns_found,
                "confidence": 0.8
            }
        return None
    
    def _detect_correlation_patterns(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
        """Detect correlation patterns between numeric columns."""
        patterns = []
        
        cols = numeric_cols[:10]  # Limit for performance
        df_numeric = df[cols].apply(pd.to_numeric, errors='coerce')
        corr_matrix = df_numeric.corr()
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if pd.notna(corr) and abs(corr) > 0.7:
                    patterns.append({
                        "columns": [col1, col2],
                        "pattern_type": "correlation",
                        "correlation": round(corr, 4),
                        "strength": "strong" if abs(corr) > 0.85 else "moderate",
                        "direction": "positive" if corr > 0 else "negative",
                        "description": f"{'Strong' if abs(corr) > 0.85 else 'Moderate'} {'positive' if corr > 0 else 'negative'} correlation between {col1} and {col2}",
                        "confidence": 0.9
                    })
        
        patterns.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return patterns[:10]
    
    def _detect_categorical_patterns(self, data: pd.Series, col: str) -> Optional[Dict]:
        """Detect patterns in categorical data."""
        data = data.dropna()
        if len(data) < 10:
            return None
        
        value_counts = data.value_counts()
        
        # Check for dominant category
        if len(value_counts) > 0:
            top_pct = value_counts.iloc[0] / len(data)
            if top_pct > 0.7:
                return {
                    "column": col,
                    "pattern_type": "dominant_category",
                    "dominant_value": str(value_counts.index[0]),
                    "percentage": round(top_pct * 100, 2),
                    "description": f"Category '{value_counts.index[0]}' dominates with {top_pct*100:.1f}% of values",
                    "confidence": 0.9
                }
        
        # Check for long tail distribution
        if len(value_counts) > 5:
            top_5_pct = value_counts.head(5).sum() / len(data)
            remaining_categories = len(value_counts) - 5
            if top_5_pct > 0.8 and remaining_categories > 10:
                return {
                    "column": col,
                    "pattern_type": "long_tail",
                    "top_5_percentage": round(top_5_pct * 100, 2),
                    "remaining_categories": remaining_categories,
                    "description": f"Long-tail distribution: top 5 categories cover {top_5_pct*100:.1f}%, with {remaining_categories} other categories",
                    "confidence": 0.85
                }
        
        return None
    
    def _generate_pattern_insights(self, patterns: Dict) -> List[str]:
        """Generate human-readable insights from detected patterns."""
        insights = []
        
        if patterns["distribution_patterns"]:
            skewed = [p for p in patterns["distribution_patterns"] if "skewed" in p.get("distribution", "")]
            if skewed:
                insights.append(f"Found {len(skewed)} skewed distributions that may benefit from transformation")
        
        if patterns["correlation_patterns"]:
            strong = [p for p in patterns["correlation_patterns"] if p.get("strength") == "strong"]
            if strong:
                insights.append(f"Identified {len(strong)} strong correlations that could indicate redundant features or causal relationships")
        
        if patterns["value_patterns"]:
            concentration = [p for p in patterns["value_patterns"] for pat in p.get("patterns", []) if pat.get("type") == "concentration"]
            if concentration:
                insights.append("Value concentration detected - consider investigating if this reflects real business patterns")
        
        return insights
    
    def advanced_trend_analysis(
        self,
        data: List[Dict[str, Any]],
        value_column: str,
        time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform advanced trend analysis with decomposition and change point detection.
        """
        try:
            df = pd.DataFrame(data)
            
            if value_column not in df.columns:
                return {"success": False, "error": f"Column '{value_column}' not found"}
            
            values = pd.to_numeric(df[value_column], errors='coerce').dropna()
            
            if len(values) < 10:
                return {"success": False, "error": "Insufficient data for trend analysis (need at least 10 points)"}
            
            # Basic trend using linear regression
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend direction
            if p_value < 0.05:
                if slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                trend_significant = True
            else:
                trend_direction = "no_significant_trend"
                trend_significant = False
            
            # Calculate trend strength
            trend_strength = abs(r_value)
            
            # Moving averages
            ma_5 = values.rolling(window=min(5, len(values)//2)).mean().dropna().tolist()
            ma_10 = values.rolling(window=min(10, len(values)//2)).mean().dropna().tolist() if len(values) >= 10 else []
            
            # Detect change points using CUSUM-like approach
            change_points = self._detect_change_points(values.values)
            
            # Volatility analysis
            volatility = values.std() / values.mean() if values.mean() != 0 else 0
            
            # Growth metrics
            if len(values) > 1 and values.iloc[0] != 0:
                total_change = ((values.iloc[-1] - values.iloc[0]) / abs(values.iloc[0])) * 100
                avg_change = values.pct_change().mean() * 100 if len(values) > 1 else 0
            else:
                total_change = 0
                avg_change = 0
            
            return {
                "success": True,
                "column": value_column,
                "data_points": len(values),
                "trend": {
                    "direction": trend_direction,
                    "significant": trend_significant,
                    "strength": round(trend_strength, 4),
                    "slope": round(slope, 6),
                    "p_value": round(p_value, 6),
                    "r_squared": round(r_value**2, 4)
                },
                "statistics": {
                    "mean": safe_value(values.mean()),
                    "median": safe_value(values.median()),
                    "std": safe_value(values.std()),
                    "min": safe_value(values.min()),
                    "max": safe_value(values.max()),
                    "volatility": round(volatility, 4)
                },
                "growth": {
                    "total_change_pct": round(total_change, 2),
                    "avg_period_change_pct": round(avg_change, 4)
                },
                "moving_averages": {
                    "ma_5": [safe_value(v) for v in ma_5[-20:]],
                    "ma_10": [safe_value(v) for v in ma_10[-20:]]
                },
                "change_points": change_points,
                "interpretation": self._interpret_trend(trend_direction, trend_strength, volatility, change_points)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _detect_change_points(self, values: np.ndarray) -> List[Dict]:
        """Detect significant change points in the data."""
        if len(values) < 10:
            return []
        
        change_points = []
        window = max(5, len(values) // 10)
        
        for i in range(window, len(values) - window):
            before = values[i-window:i]
            after = values[i:i+window]
            
            # Use t-test to detect significant changes
            t_stat, p_val = stats.ttest_ind(before, after)
            
            if p_val < 0.01:  # Significant change
                change_pct = ((np.mean(after) - np.mean(before)) / abs(np.mean(before))) * 100 if np.mean(before) != 0 else 0
                change_points.append({
                    "index": int(i),
                    "significance": round(1 - p_val, 4),
                    "change_percentage": round(change_pct, 2),
                    "direction": "increase" if np.mean(after) > np.mean(before) else "decrease"
                })
        
        # Filter to keep only the most significant change points
        change_points.sort(key=lambda x: x["significance"], reverse=True)
        return change_points[:5]
    
    def _interpret_trend(
        self,
        direction: str,
        strength: float,
        volatility: float,
        change_points: List
    ) -> str:
        """Generate human-readable trend interpretation."""
        interpretations = []
        
        if direction == "increasing":
            interpretations.append(f"The data shows an upward trend with {'strong' if strength > 0.7 else 'moderate' if strength > 0.4 else 'weak'} correlation (r={strength:.2f})")
        elif direction == "decreasing":
            interpretations.append(f"The data shows a downward trend with {'strong' if strength > 0.7 else 'moderate' if strength > 0.4 else 'weak'} correlation (r={strength:.2f})")
        else:
            interpretations.append("No statistically significant trend detected")
        
        if volatility > 0.5:
            interpretations.append("High volatility observed - values fluctuate significantly")
        elif volatility > 0.2:
            interpretations.append("Moderate volatility - some fluctuation in values")
        else:
            interpretations.append("Low volatility - values are relatively stable")
        
        if change_points:
            interpretations.append(f"Detected {len(change_points)} significant change point(s) in the data")
        
        return ". ".join(interpretations) + "."
    
    def statistical_hypothesis_testing(
        self,
        data: List[Dict[str, Any]],
        column1: str,
        column2: Optional[str] = None,
        test_type: str = "auto",
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform statistical hypothesis testing.
        
        Supports:
        - t-test (one-sample, two-sample, paired)
        - chi-square test
        - ANOVA
        - correlation test
        - normality test
        """
        try:
            df = pd.DataFrame(data)
            
            if column1 not in df.columns:
                return {"success": False, "error": f"Column '{column1}' not found"}
            
            data1 = pd.to_numeric(df[column1], errors='coerce').dropna()
            
            if len(data1) < 5:
                return {"success": False, "error": "Insufficient data for statistical testing"}
            
            results = {
                "success": True,
                "column1": column1,
                "tests_performed": []
            }
            
            # Normality test
            if len(data1) >= 8:
                stat, p_val = stats.shapiro(data1.sample(min(5000, len(data1))))
                results["normality_test"] = {
                    "test": "Shapiro-Wilk",
                    "statistic": round(stat, 4),
                    "p_value": round(p_val, 6),
                    "is_normal": p_val > 0.05,
                    "interpretation": "Data appears normally distributed" if p_val > 0.05 else "Data significantly deviates from normal distribution"
                }
                results["tests_performed"].append("normality")
            
            # Two-sample tests
            if column2 and column2 in df.columns:
                data2 = pd.to_numeric(df[column2], errors='coerce').dropna()
                
                if len(data2) >= 5:
                    # Two-sample t-test
                    t_stat, t_pval = stats.ttest_ind(data1, data2)
                    results["t_test"] = {
                        "test": "Independent Two-Sample t-test",
                        "statistic": round(t_stat, 4),
                        "p_value": round(t_pval, 6),
                        "significant": t_pval < 0.05,
                        "interpretation": f"The means are {'significantly different' if t_pval < 0.05 else 'not significantly different'} (α=0.05)"
                    }
                    results["tests_performed"].append("t_test")
                    
                    # Correlation test
                    corr, corr_pval = stats.pearsonr(data1[:min(len(data1), len(data2))], data2[:min(len(data1), len(data2))])
                    results["correlation_test"] = {
                        "test": "Pearson Correlation",
                        "correlation": round(corr, 4),
                        "p_value": round(corr_pval, 6),
                        "significant": corr_pval < 0.05,
                        "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak",
                        "interpretation": f"{'Significant' if corr_pval < 0.05 else 'No significant'} {'positive' if corr > 0 else 'negative'} correlation (r={corr:.3f})"
                    }
                    results["tests_performed"].append("correlation")
            
            # ANOVA for grouped data
            if group_column and group_column in df.columns:
                groups = df.groupby(group_column)[column1].apply(lambda x: pd.to_numeric(x, errors='coerce').dropna().tolist())
                groups = [g for g in groups if len(g) >= 3]
                
                if len(groups) >= 2:
                    f_stat, f_pval = stats.f_oneway(*groups)
                    results["anova"] = {
                        "test": "One-way ANOVA",
                        "f_statistic": round(f_stat, 4),
                        "p_value": round(f_pval, 6),
                        "significant": f_pval < 0.05,
                        "groups_tested": len(groups),
                        "interpretation": f"Group means are {'significantly different' if f_pval < 0.05 else 'not significantly different'} (α=0.05)"
                    }
                    results["tests_performed"].append("anova")
            
            return results
            
        except Exception as e:
            logger.error(f"Hypothesis testing error: {str(e)}")
            return {"success": False, "error": str(e)}
