# app/agents/forecaster.py
from app.agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from app.core.api_clients import google_client
from app.config import get_settings
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import holidays
import json
from typing import Dict, List
from datetime import datetime, timedelta
from loguru import logger

settings = get_settings()


class ForecasterAgent(BaseAgent):
    """
    Advanced forecasting using Facebook Prophet
    """
    
    def __init__(self):
        super().__init__(
            name="Forecaster",
            model=settings.FORECASTER_MODEL,
            api_client=google_client
        )
        self.indian_holidays = holidays.India(years=range(2020, 2030))
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        try:
            # Validate input
            if "dataset" not in request.context:
                return AgentResponse(
                    agent_name=self.name,
                    success=False,
                    error="No dataset provided in context"
                )
            
            df = pd.DataFrame(request.context["dataset"])
            forecast_periods = request.parameters.get("periods", 30)
            
            logger.info(f"Starting forecast for {forecast_periods} periods")
            
            # Detect date and value columns
            date_col, value_cols = self._detect_columns(df)
            
            if not date_col or not value_cols:
                return AgentResponse(
                    agent_name=self.name,
                    success=False,
                    error="Could not detect date or numeric columns for forecasting"
                )
            
            # Prepare data for Prophet
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Generate forecasts for each numeric column
            forecasts_data = {}
            
            for col in value_cols[:3]:  # Limit to 3 columns for performance
                logger.info(f"Forecasting column: {col}")
                forecast_result = await self._forecast_column(
                    df, date_col, col, forecast_periods
                )
                forecasts_data[col] = forecast_result
            
            # Get LLM interpretation
            interpretation = await self._get_interpretation(
                forecasts_data, request.query
            )
            
            # Format response for frontend
            response_data = {
                "forecast_periods": forecast_periods,
                "forecasts": forecasts_data,
                "interpretation": interpretation,
                "metadata": {
                    "date_column": date_col,
                    "forecasted_columns": value_cols[:3],
                    "model": "Facebook Prophet",
                    "includes_holidays": True,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
            logger.info("Forecast completed successfully")
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Forecaster error: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e)
            )
    
    def _detect_columns(self, df: pd.DataFrame) -> tuple:
        """Detect date and numeric columns"""
        date_col = None
        value_cols = []
        
        # Find date column
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    date_col = col
                    break
                except:
                    continue
            elif 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        # Find numeric columns (exclude date and IDs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [
            col for col in numeric_cols 
            if col != date_col and 'id' not in col.lower() and 'price' not in col.lower()
        ]
        
        # Fallback if filtered too aggressively
        if not value_cols:
             value_cols = [col for col in numeric_cols if col != date_col]

        return date_col, value_cols
    
    async def _forecast_column(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        value_col: str, 
        periods: int
    ) -> Dict:
        """Generate Prophet forecast for a single column"""
        
        # --- FIX: Aggregate duplicates (Sum values per day) ---
        # This handles the case where you have multiple products per date
        grouped_df = df.groupby(date_col)[value_col].sum().reset_index()
        
        # Prepare Prophet dataframe
        prophet_df = pd.DataFrame({
            'ds': grouped_df[date_col],
            'y': grouped_df[value_col]
        })
        
        # Remove any NaN values
        prophet_df = prophet_df.dropna()
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self._create_holiday_df(prophet_df['ds'].min(), prophet_df['ds'].max()),
            interval_width=0.95
        )
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast data (only future periods)
        future_forecast = forecast.tail(periods)
        
        # Format predictions for frontend
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "value": float(row['yhat']),
                "lower": float(row['yhat_lower']),
                "upper": float(row['yhat_upper'])
            })
        
        # Extract seasonality components
        seasonality = {
            "weekly": self._extract_weekly_pattern(model, forecast),
            "yearly": self._extract_yearly_pattern(model, forecast)
        }
        
        # Calculate metrics
        # Use the AGGREGATED actuals vs predicted
        historical_actual = prophet_df['y'].values
        historical_predicted = forecast.head(len(prophet_df))['yhat'].values
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((historical_actual - historical_predicted) / historical_actual)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 0.0
        
        return {
            "predictions": predictions,
            "seasonality": seasonality,
            "confidence_score": float(1 - (mape / 100)) if mape < 100 else 0.0,
            "metrics": {
                "mape": float(mape),
                "trend": "increasing" if predictions[-1]["value"] > predictions[0]["value"] else "decreasing",
                "volatility": float(np.std([p["value"] for p in predictions]))
            }
        }
    
    def _create_holiday_df(self, start_date, end_date) -> pd.DataFrame:
        """Create holiday dataframe for Prophet"""
        holiday_list = []
        
        for date, name in self.indian_holidays.items():
            if start_date <= pd.Timestamp(date) <= end_date + timedelta(days=365):
                holiday_list.append({
                    'ds': pd.Timestamp(date),
                    'holiday': name
                })
        
        return pd.DataFrame(holiday_list) if holiday_list else pd.DataFrame(columns=['ds', 'holiday'])
    
    def _extract_weekly_pattern(self, model, forecast) -> Dict:
        """Extract weekly seasonality pattern"""
        if 'weekly' in forecast.columns:
            weekly = forecast['weekly'].values
            return {
                "pattern": "weekly",
                "peak_day": int(np.argmax(weekly[:7])),
                "low_day": int(np.argmin(weekly[:7])),
                "average_effect": float(np.mean(np.abs(weekly)))
            }
        return {}
    
    def _extract_yearly_pattern(self, model, forecast) -> Dict:
        """Extract yearly seasonality pattern"""
        if 'yearly' in forecast.columns:
            yearly = forecast['yearly'].values
            return {
                "pattern": "yearly",
                "peak_month": int(np.argmax(yearly[:12]) + 1),
                "low_month": int(np.argmin(yearly[:12]) + 1),
                "average_effect": float(np.mean(np.abs(yearly)))
            }
        return {}
    
    async def _get_interpretation(self, forecasts_data: Dict, query: str) -> str:
        """Get LLM interpretation of forecast results"""
        
        # Prepare summary for LLM
        summary = {
            "forecasted_metrics": {},
            "trends": {},
            "confidence": {}
        }
        
        for col, data in forecasts_data.items():
            first_pred = data["predictions"][0]["value"]
            last_pred = data["predictions"][-1]["value"]
            # Avoid division by zero
            if first_pred != 0:
                change = ((last_pred - first_pred) / first_pred) * 100
            else:
                change = 0.0
            
            summary["forecasted_metrics"][col] = {
                "start_value": round(first_pred, 2),
                "end_value": round(last_pred, 2),
                "change_percentage": round(change, 2)
            }
            summary["trends"][col] = data["metrics"]["trend"]
            summary["confidence"][col] = round(data["confidence_score"], 2)
        
        prompt = f"""Analyze these forecast results and provide business insights:

Forecast Data:
{json.dumps(summary, indent=2)}

User Query: {query}

Provide a clear, actionable interpretation covering:
1. Key trends and what's driving them
2. Confidence in predictions
3. Business recommendations
4. Potential risks or opportunities

Keep it concise and business-focused."""
        
        try:
            response = await self.api_client.generate_content(
                model_name=self.model,
                prompt=prompt,
                temperature=0.6,
                max_tokens=500
            )
            
            return response.get("text", "Forecast generated successfully")
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            return "Forecast completed. See detailed predictions above."