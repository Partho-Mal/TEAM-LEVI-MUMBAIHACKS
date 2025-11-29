from app.agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from app.core.api_clients import google_client
from app.config import get_settings
import pandas as pd
import numpy as np
from scipy import stats
from pytrends.request import TrendReq
import time
import json
from typing import Dict, List, Any
from loguru import logger

settings = get_settings()


class TrendAnalystAgent(BaseAgent):
    """
    Analyzes trends combining:
    - Internal data patterns (statistical analysis)
    - External market intelligence (Google Trends)
    - Seasonal patterns and anomalies
    """
    
    def __init__(self):
        super().__init__(
            name="TrendAnalyst",
            model=settings.TREND_ANALYST_MODEL,
            api_client=google_client
        )
        self.pytrends = None
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        try:
            if "dataset" not in request.context:
                return AgentResponse(
                    agent_name=self.name,
                    success=False,
                    error="No dataset provided for analysis"
                )
            
            df = pd.DataFrame(request.context["dataset"])
            logger.info(f"Analyzing trends for dataset: {df.shape}")
            
            # 1. Internal statistical analysis
            internal_trends = self._analyze_internal_trends(df)
            
            # 2. Extract product keywords for external search
            keywords = self._extract_keywords(df, request.query)
            
            # 3. Fetch external market trends
            external_trends = await self._analyze_external_trends(keywords)
            
            # 4. Combine and interpret
            combined_analysis = {
                "internal_trends": internal_trends,
                "external_trends": external_trends,
                "keywords_analyzed": keywords
            }
            
            # 5. Get LLM insights
            insights = await self._get_insights(combined_analysis, request.query)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={
                    "analysis": combined_analysis,
                    "insights": insights,
                    "metadata": {
                        "analysis_date": pd.Timestamp.now().isoformat(),
                        "data_points_analyzed": len(df),
                        "external_sources": ["Google Trends"] if external_trends else []
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Trend Analyst error: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e)
            )
    
    def _analyze_internal_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze internal data patterns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        trends = {}
        
        # --- HELPER: Sanitize NaN/Inf for JSON ---
        def safe_float(val: Any) -> float:
            try:
                if pd.isna(val) or np.isinf(val):
                    return 0.0
                return float(val)
            except:
                return 0.0

        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 2:
                continue
            
            # Linear regression for trend
            x = np.arange(len(series))
            # Catch warnings for constant data
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            except Exception:
                slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
            
            # Calculate additional metrics
            mean_val = series.mean()
            std_val = series.std()
            cv = (std_val / mean_val) * 100 if mean_val != 0 else 0  # Coefficient of variation
            
            # Detect anomalies (values > 2 std from mean)
            z_scores = np.abs(stats.zscore(series))
            # Handle NaN z-scores (constant data)
            if np.isnan(z_scores).all():
                anomalies = 0
            else:
                anomalies = (z_scores > 2).sum()
            
            # Growth rate (first to last)
            if len(series) > 1 and series.iloc[0] != 0:
                growth_rate = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
            else:
                growth_rate = 0
            
            trends[col] = {
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "slope": safe_float(slope),
                "r_squared": safe_float(r_value ** 2),
                "p_value": safe_float(p_value),
                "significance": "significant" if p_value < 0.05 else "not significant",
                "statistics": {
                    "mean": safe_float(mean_val),
                    "std": safe_float(std_val),
                    "coefficient_of_variation": safe_float(cv),
                    "min": safe_float(series.min()),
                    "max": safe_float(series.max()),
                    "growth_rate_pct": safe_float(growth_rate)
                },
                "anomalies_detected": int(anomalies),
                "volatility": "high" if cv > 30 else "medium" if cv > 15 else "low"
            }
        
        return trends
    
    def _extract_keywords(self, df: pd.DataFrame, query: str) -> List[str]:
        """Extract product/category keywords from data"""
        keywords = []
        
        # Look for product/category columns
        text_cols = ['product', 'category', 'item', 'name', 'sku', 'type']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in text_cols):
                if df[col].dtype == 'object':
                    # Get unique values (limit to top 3)
                    unique_values = df[col].value_counts().head(3).index.tolist()
                    keywords.extend(unique_values)
        
        # Extract from query
        query_words = query.lower().split()
        business_keywords = ['sales', 'revenue', 'demand', 'inventory', 'product', 'market']
        keywords.extend([w for w in query_words if w not in business_keywords and len(w) > 3])
        
        # Limit and clean
        keywords = list(set(keywords))[:5]  # Max 5 keywords
        keywords = [k for k in keywords if len(str(k)) > 2]
        
        return keywords if keywords else ["market trends"]
    
    async def _analyze_external_trends(self, keywords: List[str]) -> Dict:
        """Fetch Google Trends data"""
        if not keywords:
            return {}
        
        try:
            # Initialize pytrends
            if self.pytrends is None:
                self.pytrends = TrendReq(hl='en-US', tz=360)
            
            trends_data = {}
            
            for keyword in keywords[:3]:  # Limit to 3 keywords to avoid rate limits
                try:
                    logger.info(f"Fetching Google Trends for: {keyword}")
                    
                    # Build payload
                    self.pytrends.build_payload(
                        [str(keyword)],
                        cat=0,
                        timeframe='today 3-m',  # Last 3 months
                        geo='IN'  # India
                    )
                    
                    # Get interest over time
                    interest_df = self.pytrends.interest_over_time()
                    
                    if not interest_df.empty and str(keyword) in interest_df.columns:
                        series = interest_df[str(keyword)]
                        
                        # Calculate trend
                        current = series.iloc[-1]
                        previous = series.iloc[0]
                        change_pct = ((current - previous) / previous * 100) if previous != 0 else 0
                        
                        # Get related queries
                        try:
                            related = self.pytrends.related_queries()
                            top_queries = []
                            if str(keyword) in related and 'top' in related[str(keyword)]:
                                top_df = related[str(keyword)]['top']
                                if not top_df.empty:
                                    top_queries = top_df['query'].head(5).tolist()
                        except:
                            top_queries = []
                        
                        trends_data[str(keyword)] = {
                            "current_interest": int(current),
                            "trend": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable",
                            "change_percentage": float(change_pct),
                            "average_interest": float(series.mean()),
                            "peak_interest": int(series.max()),
                            "related_queries": top_queries
                        }
                    
                    # Rate limit: sleep 2 seconds between requests
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch trends for '{keyword}': {e}")
                    continue
            
            if trends_data:
                logger.info(f"Successfully fetched trends for {len(trends_data)} keywords")
            
            return trends_data
            
        except Exception as e:
            logger.warning(f"External trends analysis failed: {e}")
            return {}
    
    async def _get_insights(self, analysis: Dict, query: str) -> Dict:
        """Generate insights using LLM"""
        
        prompt = f"""Analyze these combined trend insights:

Internal Trends:
{json.dumps(analysis['internal_trends'], indent=2)}

External Market Trends (Google):
{json.dumps(analysis['external_trends'], indent=2)}

User Query: {query}

Provide insights in JSON format:
{{
    "key_findings": ["list of main discoveries"],
    "opportunities": [
        {{"type": "opportunity", "message": "...", "confidence": 0.0-1.0}}
    ],
    "risks": [
        {{"type": "risk", "message": "...", "confidence": 0.0-1.0}}
    ],
    "recommendations": ["actionable recommendations"],
    "market_sentiment": "positive/negative/neutral"
}}

Focus on actionable business insights."""
        
        try:
            response = await self.api_client.generate_content(
                model_name=self.model,
                prompt=prompt,
                temperature=0.6,
                max_tokens=800
            )
            
            content = response.get("text", "{}")
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"LLM insights failed: {e}")
            
            # Fallback insights
            return {
                "key_findings": ["Internal data shows clear trends", "Market analysis available"],
                "opportunities": [
                    {"type": "opportunity", "message": "Data ready for forecasting", "confidence": 0.8}
                ],
                "risks": [],
                "recommendations": ["Continue monitoring trends", "Consider seasonal patterns"],
                "market_sentiment": "neutral"
            }