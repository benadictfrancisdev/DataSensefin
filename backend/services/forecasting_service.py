import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ForecastingService:
    """Service for time series forecasting."""
    
    @staticmethod
    def simple_forecast(
        data: List[Dict[str, Any]],
        value_column: str,
        date_column: Optional[str] = None,
        periods: int = 10,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """Perform simple time series forecasting."""
        try:
            df = pd.DataFrame(data)
            
            if value_column not in df.columns:
                return {"success": False, "error": f"Column '{value_column}' not found"}
            
            # Get numeric values
            values = pd.to_numeric(df[value_column], errors='coerce').dropna().values
            
            if len(values) < 10:
                return {"success": False, "error": "Need at least 10 data points for forecasting"}
            
            # Calculate statistics
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            trend = (values[-1] - values[0]) / n if n > 1 else 0
            
            # Detect seasonality (simple autocorrelation)
            if n > 20:
                autocorr = np.correlate(values - mean, values - mean, mode='full')
                autocorr = autocorr[n-1:]
                autocorr = autocorr / autocorr[0]
                # Find peaks for seasonality
                peaks = []
                for i in range(2, min(len(autocorr) - 1, n // 2)):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                        peaks.append(i)
                seasonality_period = peaks[0] if peaks else None
            else:
                seasonality_period = None
            
            # Choose forecasting method
            if method == "auto":
                if seasonality_period:
                    method = "seasonal"
                elif abs(trend) > std * 0.1:
                    method = "linear"
                else:
                    method = "moving_average"
            
            # Generate forecasts
            forecasts = []
            confidence_intervals = []
            
            if method == "linear":
                # Linear regression forecast
                x = np.arange(n)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                for i in range(periods):
                    future_x = n + i
                    forecast_val = intercept + slope * future_x
                    forecasts.append(forecast_val)
                    
                    # Confidence interval
                    se = std_err * np.sqrt(1 + 1/n + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                    ci_lower = forecast_val - 1.96 * se
                    ci_upper = forecast_val + 1.96 * se
                    confidence_intervals.append({"lower": ci_lower, "upper": ci_upper})
                
                model_info = {
                    "method": "linear_regression",
                    "slope": round(slope, 4),
                    "intercept": round(intercept, 4),
                    "r_squared": round(r_value**2, 4)
                }
                
            elif method == "seasonal" and seasonality_period:
                # Seasonal decomposition forecast
                seasonal_component = []
                for i in range(seasonality_period):
                    seasonal_vals = values[i::seasonality_period]
                    seasonal_component.append(np.mean(seasonal_vals) - mean)
                
                deseasonalized = values.copy()
                for i in range(len(values)):
                    deseasonalized[i] = values[i] - seasonal_component[i % seasonality_period]
                
                # Trend on deseasonalized data
                x = np.arange(n)
                slope, intercept, _, _, _ = stats.linregress(x, deseasonalized)
                
                for i in range(periods):
                    future_x = n + i
                    trend_val = intercept + slope * future_x
                    seasonal_val = seasonal_component[(n + i) % seasonality_period]
                    forecast_val = trend_val + seasonal_val
                    forecasts.append(forecast_val)
                    
                    ci_lower = forecast_val - 1.96 * std
                    ci_upper = forecast_val + 1.96 * std
                    confidence_intervals.append({"lower": ci_lower, "upper": ci_upper})
                
                model_info = {
                    "method": "seasonal_decomposition",
                    "seasonality_period": seasonality_period,
                    "trend_slope": round(slope, 4)
                }
                
            else:  # Moving average
                # Exponential moving average forecast
                alpha = 0.3
                ema = values[0]
                for val in values[1:]:
                    ema = alpha * val + (1 - alpha) * ema
                
                # Simple trend
                recent_trend = (values[-1] - values[-min(5, len(values))]) / min(5, len(values))
                
                for i in range(periods):
                    forecast_val = ema + recent_trend * (i + 1)
                    forecasts.append(forecast_val)
                    
                    ci_range = std * np.sqrt(i + 1) * 0.5
                    confidence_intervals.append({
                        "lower": forecast_val - 1.96 * ci_range,
                        "upper": forecast_val + 1.96 * ci_range
                    })
                
                model_info = {
                    "method": "exponential_moving_average",
                    "alpha": alpha,
                    "recent_trend": round(recent_trend, 4)
                }
            
            # Calculate accuracy metrics on historical data
            if len(values) > 5:
                train_size = len(values) - 5
                train_values = values[:train_size]
                test_values = values[train_size:]
                
                # Simple forecast for test period
                if method == "linear":
                    x_train = np.arange(train_size)
                    slope, intercept, _, _, _ = stats.linregress(x_train, train_values)
                    test_forecasts = [intercept + slope * (train_size + i) for i in range(5)]
                else:
                    test_forecasts = [train_values[-1]] * 5
                
                mape = np.mean(np.abs((test_values - test_forecasts) / test_values)) * 100
                rmse = np.sqrt(np.mean((test_values - test_forecasts) ** 2))
            else:
                mape = None
                rmse = None
            
            # Prepare historical data for chart
            historical = [
                {"index": i, "value": round(float(values[i]), 4), "type": "historical"}
                for i in range(len(values))
            ]
            
            forecast_data = [
                {
                    "index": len(values) + i,
                    "value": round(float(forecasts[i]), 4),
                    "type": "forecast",
                    "ci_lower": round(float(confidence_intervals[i]["lower"]), 4),
                    "ci_upper": round(float(confidence_intervals[i]["upper"]), 4)
                }
                for i in range(periods)
            ]
            
            # Trend analysis
            if len(forecasts) > 0:
                forecast_change = ((forecasts[-1] - values[-1]) / values[-1]) * 100 if values[-1] != 0 else 0
                trend_direction = "increasing" if forecast_change > 2 else "decreasing" if forecast_change < -2 else "stable"
            else:
                forecast_change = 0
                trend_direction = "unknown"
            
            return {
                "success": True,
                "column": value_column,
                "periods": periods,
                "model_info": model_info,
                "accuracy_metrics": {
                    "mape": round(mape, 2) if mape else None,
                    "rmse": round(rmse, 4) if rmse else None
                },
                "historical_data": historical,
                "forecast_data": forecast_data,
                "summary": {
                    "current_value": round(float(values[-1]), 4),
                    "forecasted_end_value": round(float(forecasts[-1]), 4),
                    "forecast_change_pct": round(forecast_change, 2),
                    "trend_direction": trend_direction,
                    "seasonality_detected": seasonality_period is not None,
                    "seasonality_period": seasonality_period
                }
            }
            
        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def multi_column_forecast(
        data: List[Dict[str, Any]],
        columns: List[str],
        periods: int = 10
    ) -> Dict[str, Any]:
        """Forecast multiple columns simultaneously."""
        try:
            results = []
            
            for col in columns[:5]:  # Limit to 5 columns
                forecast_result = ForecastingService.simple_forecast(
                    data=data,
                    value_column=col,
                    periods=periods
                )
                
                if forecast_result.get("success"):
                    results.append({
                        "column": col,
                        "summary": forecast_result.get("summary"),
                        "model_info": forecast_result.get("model_info"),
                        "forecast_data": forecast_result.get("forecast_data")
                    })
            
            if not results:
                return {"success": False, "error": "No columns could be forecasted"}
            
            return {
                "success": True,
                "periods": periods,
                "forecasts": results,
                "columns_processed": len(results)
            }
            
        except Exception as e:
            logger.error(f"Multi-column forecast error: {str(e)}")
            return {"success": False, "error": str(e)}
