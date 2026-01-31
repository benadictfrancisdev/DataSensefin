"""
NLP Engine - Enhanced Natural Language Processing with Deep Contextual Understanding.

This module provides advanced NLP capabilities including:
- Semantic text analysis and understanding
- Entity extraction and classification
- Sentiment analysis with confidence scoring
- Text similarity and clustering
- Query understanding and intent classification
- Contextual data interpretation
"""

import os
import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)


class NLPEngine:
    """Advanced NLP Engine for deep contextual understanding of data and queries."""
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        self._intent_patterns = self._build_intent_patterns()
        self._entity_patterns = self._build_entity_patterns()
        self._sentiment_lexicon = self._build_sentiment_lexicon()
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for intent classification."""
        return {
            "aggregation": ["sum", "total", "count", "average", "mean", "median", "max", "min", "aggregate"],
            "comparison": ["compare", "versus", "vs", "difference", "between", "higher", "lower", "more", "less"],
            "trend": ["trend", "over time", "growth", "decline", "change", "increase", "decrease", "forecast"],
            "distribution": ["distribution", "spread", "range", "variance", "histogram", "frequency"],
            "correlation": ["correlation", "relationship", "related", "associated", "impact", "affect", "influence"],
            "filtering": ["filter", "where", "only", "exclude", "include", "specific", "particular"],
            "grouping": ["group by", "per", "each", "by category", "breakdown", "segment"],
            "ranking": ["top", "bottom", "best", "worst", "highest", "lowest", "rank"],
            "anomaly": ["outlier", "anomaly", "unusual", "abnormal", "unexpected", "strange"],
            "prediction": ["predict", "forecast", "estimate", "project", "future", "expect"],
            "explanation": ["why", "explain", "reason", "cause", "understand", "insight"],
            "summary": ["summary", "overview", "describe", "tell me about", "what is"]
        }
    
    def _build_entity_patterns(self) -> Dict[str, str]:
        """Build regex patterns for entity extraction."""
        return {
            "date": r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s*\d{4})\b',
            "number": r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b',
            "percentage": r'\b(\d+(?:\.\d+)?)\s*%',
            "currency": r'[\$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "time_period": r'\b(daily|weekly|monthly|quarterly|yearly|annual|ytd|mtd|last\s+\d+\s+(?:days?|weeks?|months?|years?))\b',
            "comparison_operator": r'\b(greater than|less than|equal to|at least|at most|between|more than|fewer than)\b'
        }
    
    def _build_sentiment_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build sentiment lexicon for analysis."""
        return {
            "positive": {
                "increase": 0.6, "growth": 0.7, "profit": 0.8, "success": 0.9, "improvement": 0.7,
                "excellent": 0.9, "great": 0.8, "good": 0.6, "positive": 0.7, "gain": 0.6,
                "rise": 0.5, "up": 0.4, "higher": 0.5, "better": 0.6, "strong": 0.6,
                "efficient": 0.7, "effective": 0.7, "optimal": 0.8, "peak": 0.7, "best": 0.8
            },
            "negative": {
                "decrease": -0.6, "decline": -0.7, "loss": -0.8, "failure": -0.9, "drop": -0.6,
                "poor": -0.7, "bad": -0.7, "negative": -0.7, "down": -0.4, "lower": -0.5,
                "worse": -0.6, "weak": -0.6, "inefficient": -0.7, "problem": -0.6, "issue": -0.5,
                "risk": -0.5, "concern": -0.5, "warning": -0.6, "critical": -0.7, "worst": -0.8
            }
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a natural language query.
        
        Returns intent, entities, sentiment, and structured interpretation.
        """
        try:
            query_lower = query.lower().strip()
            
            # Extract intents
            intents = self._classify_intents(query_lower)
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(query_lower)
            
            # Extract column references (for data queries)
            column_hints = self._extract_column_hints(query_lower)
            
            # Determine query complexity
            complexity = self._assess_query_complexity(query_lower, intents, entities)
            
            # Generate structured interpretation
            interpretation = self._generate_interpretation(query, intents, entities, column_hints)
            
            return {
                "success": True,
                "original_query": query,
                "intents": intents,
                "primary_intent": intents[0] if intents else "general",
                "entities": entities,
                "sentiment": sentiment,
                "column_hints": column_hints,
                "complexity": complexity,
                "interpretation": interpretation,
                "confidence": self._calculate_confidence(intents, entities)
            }
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _classify_intents(self, query: str) -> List[str]:
        """Classify query intents based on patterns."""
        detected_intents = []
        intent_scores = {}
        
        for intent, keywords in self._intent_patterns.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                intent_scores[intent] = score
        
        # Sort by score and return top intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        detected_intents = [intent for intent, _ in sorted_intents[:3]]
        
        return detected_intents if detected_intents else ["general"]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {}
        
        for entity_type, pattern in self._entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        words = text.split()
        positive_score = 0
        negative_score = 0
        matched_words = {"positive": [], "negative": []}
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean in self._sentiment_lexicon["positive"]:
                positive_score += self._sentiment_lexicon["positive"][word_clean]
                matched_words["positive"].append(word_clean)
            elif word_clean in self._sentiment_lexicon["negative"]:
                negative_score += abs(self._sentiment_lexicon["negative"][word_clean])
                matched_words["negative"].append(word_clean)
        
        total_score = positive_score - negative_score
        
        if total_score > 0.3:
            sentiment_label = "positive"
        elif total_score < -0.3:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "label": sentiment_label,
            "score": round(total_score, 3),
            "positive_score": round(positive_score, 3),
            "negative_score": round(negative_score, 3),
            "matched_words": matched_words,
            "confidence": min(1.0, (abs(total_score) + 0.5) / 2)
        }
    
    def _extract_column_hints(self, query: str) -> List[str]:
        """Extract potential column name references from query."""
        # Common data field patterns
        hints = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for match in quoted:
            hints.extend([m for m in match if m])
        
        # Look for common column-like patterns
        column_patterns = [
            r'\b(\w+)_(\w+)\b',  # underscore_separated
            r'\b(\w+)(?:column|field|metric|measure|dimension)\b',
            r'\bthe\s+(\w+)\b',
            r'\bby\s+(\w+)\b',
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    hints.extend([m for m in match if m and len(m) > 2])
                elif len(match) > 2:
                    hints.append(match)
        
        return list(set(hints))
    
    def _assess_query_complexity(self, query: str, intents: List[str], entities: Dict) -> Dict[str, Any]:
        """Assess the complexity of a query."""
        word_count = len(query.split())
        entity_count = sum(len(v) for v in entities.values())
        intent_count = len(intents)
        
        # Calculate complexity score
        complexity_score = (
            min(word_count / 20, 1) * 0.3 +
            min(entity_count / 5, 1) * 0.3 +
            min(intent_count / 3, 1) * 0.4
        )
        
        if complexity_score < 0.3:
            level = "simple"
        elif complexity_score < 0.6:
            level = "moderate"
        else:
            level = "complex"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "word_count": word_count,
            "entity_count": entity_count,
            "intent_count": intent_count,
            "requires_aggregation": any(i in intents for i in ["aggregation", "grouping"]),
            "requires_comparison": "comparison" in intents,
            "requires_time_analysis": "trend" in intents or "time_period" in entities
        }
    
    def _generate_interpretation(
        self,
        query: str,
        intents: List[str],
        entities: Dict,
        column_hints: List[str]
    ) -> Dict[str, Any]:
        """Generate structured interpretation of the query."""
        interpretation = {
            "action": intents[0] if intents else "describe",
            "targets": column_hints,
            "filters": [],
            "groupings": [],
            "time_context": None,
            "comparison_context": None
        }
        
        # Extract time context
        if "time_period" in entities:
            interpretation["time_context"] = entities["time_period"][0]
        
        # Detect grouping requests
        group_patterns = [
            r'group(?:ed)?\s+by\s+(\w+)',
            r'per\s+(\w+)',
            r'by\s+(\w+)(?:\s+and\s+(\w+))?'
        ]
        for pattern in group_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if isinstance(match, tuple):
                    interpretation["groupings"].extend([m for m in match if m])
                else:
                    interpretation["groupings"].append(match)
        
        # Detect filter conditions
        filter_patterns = [
            r'where\s+(\w+)\s*(=|>|<|>=|<=|!=)\s*(\w+)',
            r'(\w+)\s+(?:is|equals?)\s+(\w+)',
            r'only\s+(\w+)'
        ]
        for pattern in filter_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                interpretation["filters"].extend(matches)
        
        return interpretation
    
    def _calculate_confidence(self, intents: List[str], entities: Dict) -> float:
        """Calculate overall confidence in query understanding."""
        base_confidence = 0.5
        
        # Add confidence for detected intents
        if intents and intents[0] != "general":
            base_confidence += 0.2
        
        # Add confidence for extracted entities
        if entities:
            base_confidence += min(len(entities) * 0.1, 0.3)
        
        return min(round(base_confidence, 2), 0.99)
    
    def extract_data_context(
        self,
        data: List[Dict[str, Any]],
        columns: List[str]
    ) -> Dict[str, Any]:
        """Extract semantic context from data for enhanced understanding."""
        try:
            df = pd.DataFrame(data)
            
            context = {
                "column_semantics": {},
                "data_domain": None,
                "key_metrics": [],
                "categorical_dimensions": [],
                "temporal_columns": [],
                "identifier_columns": []
            }
            
            for col in columns[:20]:  # Limit analysis
                col_data = df[col]
                col_lower = col.lower()
                
                semantics = {
                    "name": col,
                    "likely_type": self._infer_column_type(col_data, col_lower),
                    "semantic_category": self._infer_semantic_category(col_lower),
                    "sample_values": col_data.dropna().head(5).tolist()
                }
                
                context["column_semantics"][col] = semantics
                
                # Categorize columns
                if semantics["semantic_category"] == "metric":
                    context["key_metrics"].append(col)
                elif semantics["semantic_category"] == "dimension":
                    context["categorical_dimensions"].append(col)
                elif semantics["semantic_category"] == "temporal":
                    context["temporal_columns"].append(col)
                elif semantics["semantic_category"] == "identifier":
                    context["identifier_columns"].append(col)
            
            # Infer data domain
            context["data_domain"] = self._infer_data_domain(columns, context)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            logger.error(f"Data context extraction error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _infer_column_type(self, col_data: pd.Series, col_name: str) -> str:
        """Infer the semantic type of a column."""
        # Check for numeric
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        if numeric_data.notna().sum() / len(col_data) > 0.8:
            return "numeric"
        
        # Check for datetime
        try:
            pd.to_datetime(col_data, errors='coerce')
            if pd.to_datetime(col_data, errors='coerce').notna().sum() / len(col_data) > 0.5:
                return "datetime"
        except:
            pass
        
        # Check for boolean
        unique_vals = col_data.dropna().unique()
        if len(unique_vals) <= 2:
            return "boolean"
        
        # Check cardinality for categorical vs text
        if col_data.nunique() / len(col_data) < 0.1:
            return "categorical"
        
        return "text"
    
    def _infer_semantic_category(self, col_name: str) -> str:
        """Infer semantic category from column name."""
        metric_keywords = ["amount", "total", "sum", "count", "quantity", "price", "cost", "revenue", 
                         "profit", "sales", "value", "rate", "score", "percentage", "ratio"]
        dimension_keywords = ["category", "type", "status", "region", "country", "city", "segment",
                            "channel", "source", "brand", "product", "customer"]
        temporal_keywords = ["date", "time", "year", "month", "day", "week", "quarter", "period",
                           "created", "updated", "timestamp"]
        identifier_keywords = ["id", "code", "key", "number", "uuid", "sku", "reference"]
        
        col_lower = col_name.lower()
        
        if any(kw in col_lower for kw in identifier_keywords):
            return "identifier"
        if any(kw in col_lower for kw in temporal_keywords):
            return "temporal"
        if any(kw in col_lower for kw in metric_keywords):
            return "metric"
        if any(kw in col_lower for kw in dimension_keywords):
            return "dimension"
        
        return "unknown"
    
    def _infer_data_domain(self, columns: List[str], context: Dict) -> str:
        """Infer the business domain of the data."""
        columns_lower = " ".join(c.lower() for c in columns)
        
        domains = {
            "sales": ["sale", "revenue", "order", "customer", "product", "price", "discount"],
            "finance": ["transaction", "account", "balance", "payment", "invoice", "credit", "debit"],
            "marketing": ["campaign", "click", "impression", "conversion", "channel", "audience"],
            "hr": ["employee", "salary", "department", "hire", "performance", "attendance"],
            "inventory": ["stock", "inventory", "warehouse", "quantity", "sku", "supplier"],
            "healthcare": ["patient", "diagnosis", "treatment", "medical", "prescription"],
            "ecommerce": ["cart", "checkout", "product", "shipping", "return", "review"]
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for kw in keywords if kw in columns_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    async def generate_smart_query(
        self,
        natural_query: str,
        data_context: Dict[str, Any],
        columns: List[str]
    ) -> Dict[str, Any]:
        """Generate a structured analytical query from natural language."""
        try:
            if not self.api_key:
                return self._fallback_query_generation(natural_query, data_context, columns)
            
            from emergentintegrations.llm.chat import LlmChat, UserMessage
            
            system_message = """You are a data query translator. Convert natural language questions into structured analytical queries.
            
Output JSON with:
- operation: the main operation (aggregate, filter, group, compare, trend, etc.)
- target_columns: columns to analyze
- aggregations: list of {column, function} for aggregations
- filters: list of {column, operator, value} for filters
- groupings: columns to group by
- sort: {column, direction}
- limit: number of results

Be precise and use only columns that exist in the data."""
            
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"query-gen-{hashlib.md5(natural_query.encode()).hexdigest()[:8]}",
                system_message=system_message
            )
            chat.with_model("openai", "gpt-5.2")
            
            prompt = f"""Convert this question to a structured query:
Question: {natural_query}

Available columns: {', '.join(columns)}
Data domain: {data_context.get('context', {}).get('data_domain', 'general')}
Key metrics: {data_context.get('context', {}).get('key_metrics', [])}
Dimensions: {data_context.get('context', {}).get('categorical_dimensions', [])}

Return only valid JSON."""
            
            response = await chat.send_message(UserMessage(text=prompt))
            
            # Parse JSON response
            response_text = str(response)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                structured_query = json.loads(response_text[json_start:json_end])
            else:
                structured_query = self._fallback_query_generation(natural_query, data_context, columns)["query"]
            
            return {
                "success": True,
                "natural_query": natural_query,
                "structured_query": structured_query
            }
            
        except Exception as e:
            logger.error(f"Smart query generation error: {str(e)}")
            return self._fallback_query_generation(natural_query, data_context, columns)
    
    def _fallback_query_generation(
        self,
        query: str,
        data_context: Dict,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Fallback query generation without LLM."""
        analysis = self.analyze_query(query)
        
        structured = {
            "operation": analysis["primary_intent"],
            "target_columns": analysis.get("column_hints", columns[:3]),
            "aggregations": [],
            "filters": [],
            "groupings": analysis.get("interpretation", {}).get("groupings", []),
            "sort": None,
            "limit": 10
        }
        
        # Add aggregations based on intent
        if analysis["primary_intent"] in ["aggregation", "summary"]:
            metrics = data_context.get("context", {}).get("key_metrics", [])
            for metric in metrics[:3]:
                structured["aggregations"].append({"column": metric, "function": "sum"})
        
        return {"success": True, "natural_query": query, "query": structured}
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using TF-IDF approach."""
        def tokenize(text):
            return set(re.findall(r'\b\w+\b', text.lower()))
        
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        
        return round(jaccard, 4)
    
    def cluster_text_data(
        self,
        texts: List[str],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Cluster text data based on content similarity."""
        try:
            if len(texts) < n_clusters:
                n_clusters = max(2, len(texts) // 2)
            
            # Build simple TF representation
            all_words = set()
            text_tokens = []
            for text in texts:
                tokens = set(re.findall(r'\b\w+\b', str(text).lower()))
                text_tokens.append(tokens)
                all_words.update(tokens)
            
            all_words = list(all_words)
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            
            # Create feature matrix
            features = np.zeros((len(texts), len(all_words)))
            for i, tokens in enumerate(text_tokens):
                for token in tokens:
                    if token in word_to_idx:
                        features[i, word_to_idx[token]] = 1
            
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Get cluster info
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = {"texts": [], "indices": []}
                clusters[label]["texts"].append(texts[i][:100])
                clusters[label]["indices"].append(i)
            
            return {
                "success": True,
                "n_clusters": n_clusters,
                "clusters": [
                    {
                        "cluster_id": int(k),
                        "size": len(v["texts"]),
                        "sample_texts": v["texts"][:3],
                        "indices": v["indices"]
                    }
                    for k, v in clusters.items()
                ],
                "labels": [int(l) for l in labels]
            }
            
        except Exception as e:
            logger.error(f"Text clustering error: {str(e)}")
            return {"success": False, "error": str(e)}
