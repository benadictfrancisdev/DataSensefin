import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)


class AIInsightsService:
    """Service for AI-powered data insights using LLM."""
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        self._chat = None
    
    async def _get_chat(self, session_id: str, system_message: str):
        """Initialize LLM chat."""
        from emergentintegrations.llm.chat import LlmChat
        
        chat = LlmChat(
            api_key=self.api_key,
            session_id=session_id,
            system_message=system_message
        )
        chat.with_model("openai", "gpt-5.2")
        return chat
    
    def _prepare_data_summary(self, data: List[Dict[str, Any]], columns: List[str]) -> str:
        """Prepare a summary of the data for the AI."""
        df = pd.DataFrame(data)
        
        summary = f"Dataset has {len(df)} rows and {len(columns)} columns.\n\n"
        summary += "Column Statistics:\n"
        
        for col in columns[:10]:  # Limit to 10 columns
            col_data = df[col]
            missing = col_data.isna().sum()
            
            if pd.api.types.is_numeric_dtype(col_data) or pd.to_numeric(col_data, errors='coerce').notna().sum() > len(df) * 0.7:
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    summary += f"- {col} (numeric): mean={numeric_data.mean():.2f}, min={numeric_data.min():.2f}, max={numeric_data.max():.2f}, missing={missing}\n"
            else:
                unique = col_data.nunique()
                top_val = col_data.value_counts().index[0] if len(col_data.value_counts()) > 0 else 'N/A'
                summary += f"- {col} (categorical): unique={unique}, top='{top_val}', missing={missing}\n"
        
        # Sample data
        summary += "\nSample Data (first 3 rows):\n"
        for i, row in enumerate(data[:3]):
            summary += f"Row {i+1}: {json.dumps({k: str(v)[:50] for k, v in list(row.items())[:5]})[:200]}\n"
        
        return summary
    
    async def generate_insights(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        dataset_name: str = "Dataset",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered insights from the data."""
        try:
            if not self.api_key:
                return {"success": False, "error": "AI service not configured"}
            
            from emergentintegrations.llm.chat import UserMessage
            
            data_summary = self._prepare_data_summary(data, columns)
            
            system_message = """You are an expert data analyst AI. Analyze the provided data and generate actionable insights.
Provide your response in a structured JSON format with these fields:
- key_findings: list of 3-5 most important findings
- trends: list of identified trends
- anomalies: any unusual patterns or outliers noted
- recommendations: 3-5 actionable recommendations
- data_quality_issues: any data quality concerns
- next_steps: suggested analysis steps

Be specific, quantitative, and business-focused in your insights."""
            
            chat = await self._get_chat(f"insights-{dataset_name}", system_message)
            
            focus_str = ""
            if focus_areas:
                focus_str = f"\nFocus particularly on: {', '.join(focus_areas)}"
            
            prompt = f"""Analyze this dataset '{dataset_name}' and provide comprehensive insights:

{data_summary}{focus_str}

Provide detailed, actionable insights in JSON format."""
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Try to parse JSON from response
            try:
                # Find JSON in response
                response_text = str(response)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    insights = json.loads(response_text[json_start:json_end])
                else:
                    insights = {"raw_insights": response_text}
            except json.JSONDecodeError:
                insights = {"raw_insights": str(response)}
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "insights": insights,
                "summary": data_summary[:500]
            }
            
        except Exception as e:
            logger.error(f"AI Insights error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def answer_query(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Answer natural language queries about the data."""
        try:
            if not self.api_key:
                return {"success": False, "error": "AI service not configured"}
            
            from emergentintegrations.llm.chat import UserMessage
            
            data_summary = self._prepare_data_summary(data, columns)
            
            system_message = """You are a friendly and helpful data analyst assistant. Help users understand their data in plain, clear language.

IMPORTANT FORMATTING RULES:
- Use simple, clean text without markdown symbols
- Do NOT use asterisks (*), hashtags (#), or other special formatting characters
- Write in natural conversational paragraphs
- Use bullet points with simple dashes (-) if listing items
- Keep explanations clear and easy to read
- Be specific with numbers and percentages when relevant

When answering:
1. Give a direct, clear answer first
2. Support with specific data points
3. Suggest what the user might want to look at next
4. Keep it conversational and helpful"""
            
            session_id = f"query-{hash(query) % 10000}"
            chat = await self._get_chat(session_id, system_message)
            
            prompt = f"""Based on this data:

{data_summary}

User Question: {query}

Please provide a helpful, clear response in plain conversational text. Do not use any markdown formatting like asterisks or hashtags."""
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Clean the response text - remove markdown symbols
            response_text = str(response)
            # Remove common markdown formatting
            response_text = response_text.replace('**', '')
            response_text = response_text.replace('__', '')
            response_text = response_text.replace('##', '')
            response_text = response_text.replace('###', '')
            response_text = response_text.replace('####', '')
            response_text = response_text.replace('`', '')
            # Clean up any double spaces
            while '  ' in response_text:
                response_text = response_text.replace('  ', ' ')
            response_text = response_text.strip()
            
            # Extract any suggested charts
            charts = []
            response_lower = response_text.lower()
            if 'bar chart' in response_lower or 'bar graph' in response_lower:
                charts.append({"type": "bar", "suggested": True})
            if 'line chart' in response_lower or 'line graph' in response_lower or 'trend' in response_lower:
                charts.append({"type": "line", "suggested": True})
            if 'pie chart' in response_lower:
                charts.append({"type": "pie", "suggested": True})
            if 'scatter' in response_lower or 'correlation' in response_lower:
                charts.append({"type": "scatter", "suggested": True})
            
            return {
                "success": True,
                "query": query,
                "answer": response_text,
                "suggested_charts": charts,
                "confidence": 0.95
            }
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def explain_analysis(
        self,
        analysis_type: str,
        analysis_results: Dict[str, Any],
        data_context: str = ""
    ) -> Dict[str, Any]:
        """Generate AI explanation for analysis results."""
        try:
            if not self.api_key:
                return {"success": False, "error": "AI service not configured"}
            
            from emergentintegrations.llm.chat import UserMessage
            
            system_message = """You are an expert data analyst explaining analysis results to business users.
Provide clear, jargon-free explanations that:
1. Explain what the analysis found
2. Why it matters for the business
3. What actions should be taken
4. Any caveats or limitations"""
            
            chat = await self._get_chat(f"explain-{analysis_type}", system_message)
            
            results_str = json.dumps(analysis_results, indent=2, default=str)[:2000]
            
            prompt = f"""Explain these {analysis_type} results in simple terms:

{results_str}

{data_context}

Provide a clear explanation suitable for non-technical stakeholders."""
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "explanation": str(response)
            }
            
        except Exception as e:
            logger.error(f"Explanation error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate_recommendations(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        analysis_results: Dict[str, Any],
        business_context: str = ""
    ) -> Dict[str, Any]:
        """Generate AI-powered recommendations based on analysis."""
        try:
            if not self.api_key:
                return {"success": False, "error": "AI service not configured"}
            
            from emergentintegrations.llm.chat import UserMessage
            
            data_summary = self._prepare_data_summary(data, columns)
            results_str = json.dumps(analysis_results, indent=2, default=str)[:1500]
            
            system_message = """You are a strategic data consultant providing actionable recommendations.
Provide recommendations that are:
1. Specific and actionable
2. Prioritized by impact and effort
3. Data-driven
4. Include expected outcomes

Format as JSON with:
- immediate_actions: things to do now
- short_term: actions for the next 1-3 months
- long_term: strategic initiatives
- metrics_to_track: KPIs to monitor"""
            
            chat = await self._get_chat("recommendations", system_message)
            
            prompt = f"""Based on this data analysis:

Data Summary:
{data_summary[:800]}

Analysis Results:
{results_str}

{f'Business Context: {business_context}' if business_context else ''}

Provide strategic recommendations in JSON format."""
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Try to parse JSON
            try:
                response_text = str(response)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    recommendations = json.loads(response_text[json_start:json_end])
                else:
                    recommendations = {"recommendations": response_text}
            except json.JSONDecodeError:
                recommendations = {"recommendations": str(response)}
            
            return {
                "success": True,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Recommendations error: {str(e)}")
            return {"success": False, "error": str(e)}
