"""
Enhanced AI Assistant module for LogixSense platform.

This module implements a comprehensive NLP-powered intelligent assistant that provides
natural language interfaces for logistics data analysis, query processing,
insights generation, and data visualizations. It processes the entire database
for generating responses, not just similarity-based context.
"""

import os
import json
import httpx
import numpy as np
import pandas as pd
import asyncio
import logging
import traceback
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/assistant",
    tags=["AI Assistant"]
)

# Global service instances
ollama_service = None
vector_service = None
visualization_service = None
logistics_analyzer = None
data_processor = None

# Enhanced system prompt - improved to generate high-quality, structured responses
ENHANCED_SYSTEM_PROMPT = """You are LogixSense AI Assistant, a specialist in logistics and supply chain analytics.
Your goal is to provide detailed, accurate information about logistics operations based on shipping data.

RESPONSE STRUCTURE GUIDELINES:
1. Begin with a concise summary (2-3 sentences) directly answering the main question
2. Organize your response into clear, labeled sections with ## markdown headers
3. Use bullet points (•) for listing multiple items or data points
4. Include specific metrics with properly formatted numbers (use commas for thousands)
5. Bold key statistics or important insights using **bold formatting**
6. For numerical data, include percentage changes or comparisons when relevant

ANALYSIS GUIDELINES:
1. Focus on data-driven insights rather than general statements
2. Compare metrics across different dimensions (time, location, carriers, etc.)
3. Identify patterns, anomalies, or trends in the logistics data
4. Provide context for why certain patterns or anomalies might exist
5. Analyze both the "what" (observations) and the "why" (explanations)

ALWAYS END YOUR RESPONSE WITH:
1. 2-3 specific, actionable recommendations based on the data analysis
2. A suggestion for the most appropriate visualization type (bar chart, line chart, pie chart, etc.) 
   that would best illustrate the key insights, with a brief explanation of why

Keep your tone professional but conversational, and prioritize accuracy and specificity over generalization.
"""

# Enhanced user prompt template - provides better context and structures the request
def generate_enhanced_user_prompt(query, analysis_text, stats_text, context_text):
    return f"""I need a comprehensive logistics analysis based on the following query:

QUERY: {query}

COMPREHENSIVE DATABASE ANALYSIS RESULTS:
{analysis_text}

LOGISTICS DATABASE SUMMARY STATISTICS:
{stats_text}

ADDITIONAL RELEVANT CONTEXT:
{context_text}

Please provide a well-structured response with the following elements:
1. Start with a direct answer to my query
2. Organize your analysis into clear sections with ## markdown headers
3. Include specific numbers and metrics from the data (use commas for thousands)
4. Highlight key insights in **bold**
5. End with 2-3 actionable recommendations based on the data
6. Suggest the most appropriate visualization type that would best illustrate this data and explain why

Focus on providing data-driven insights that help me understand our logistics operations better. Be comprehensive but clear, focusing on the most important patterns and insights in the data."""

# Function to use in the assistant_query method to generate the response
def prepare_enhanced_prompts(query, analysis_text, stats_text, context_text):
    """Prepare enhanced system and user prompts for better quality responses"""
    system_prompt = ENHANCED_SYSTEM_PROMPT
    user_prompt = generate_enhanced_user_prompt(query, analysis_text, stats_text, context_text)
    
    return system_prompt, user_prompt

# Example usage in assistant_query() function:
"""
# Replace the existing system_prompt and user_prompt sections with:

system_prompt, user_prompt = prepare_enhanced_prompts(
    query=query,
    analysis_text=analysis_text,
    stats_text=stats_text,
    context_text=context_text
)

# Then continue with the existing code:
response = await ollama_service.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.7,
    max_tokens=2000
)
"""

# Automatic visualization suggestion extraction
def extract_visualization_recommendation(response):
    """Extract visualization recommendation from the response text"""
    import re
    
    # Look for visualization recommendation patterns
    patterns = [
        r"(?i)## Suggested Visualization:?\s*(.*?)(?:\n\n|\n##|$)",
        r"(?i)For visualization,?\s*(.*?)(?:\n\n|\n##|$)",
        r"(?i)I recommend (a|using) (bar|pie|line|chart|graph) (chart|graph).*?(?:\n\n|\n##|$)",
        r"(?i)A (bar|pie|line) (chart|graph) would (best|be ideal).*?(?:\n\n|\n##|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            recommendation = match.group(1) if "Suggested Visualization" in pattern else match.group(0)
            
            # Determine visualization type from recommendation
            viz_type = "bar"  # Default
            if "pie" in recommendation.lower():
                viz_type = "pie"
            elif "line" in recommendation.lower() or "trend" in recommendation.lower() or "time" in recommendation.lower():
                viz_type = "line"
                
            return {
                "recommendation": recommendation.strip(),
                "visualization_type": viz_type
            }
    
    # No explicit recommendation found
    return None

# Initialize services
def initialize_services(processor):
    """Initialize the assistant services"""
    global ollama_service, vector_service, visualization_service, logistics_analyzer, data_processor
    
    try:
        # Store data processor
        data_processor = processor
        logger.info(f"Data processor initialized with {len(processor.df) if hasattr(processor, 'df') else 0} records")
        
        # Initialize Ollama service
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")
        logger.info(f"Initializing Ollama service with URL {ollama_url} and model {ollama_model}")
        ollama_service = OllamaService(base_url=ollama_url, model=ollama_model)
        
        # Initialize vector service
        vector_service = FAISSVectorService()
        
        # Initialize logistics analyzer
        logger.info("Initializing LogisticsAnalyzer with data processor")
        logistics_analyzer = LogisticsAnalyzer(data_processor)
        
        # Try to load existing index
        vector_index_loaded = False
        if vector_service is not None:
            vector_index_loaded = vector_service.load_index()
        
        # Check Ollama connection
        import asyncio
        try:
            connection_check = asyncio.run(ollama_service.check_connection())
            if connection_check[0]:
                logger.info("Successfully connected to Ollama API")
            else:
                logger.warning(f"Ollama connection check failed: {connection_check[1]}")
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {str(e)}")
        
        logger.info(f"AI Assistant services initialized successfully. FAISS index loaded: {vector_index_loaded}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing AI Assistant services: {str(e)}")
        traceback.print_exc()
        return False
    
@router.get("/config")
async def get_assistant_config():
    """Get the assistant configuration and check connections to services"""
    global ollama_service, vector_service, logistics_analyzer, data_processor
    
    try:
        # Check Ollama connectivity
        ollama_status = "unknown"
        ollama_error = None
        if ollama_service:
            try:
                import requests
                response = requests.get(f"{ollama_service.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if any(m.get('name') == ollama_service.model for m in models):
                        ollama_status = "ready"
                    else:
                        ollama_status = "model_not_found"
                        ollama_error = f"Model {ollama_service.model} not found in Ollama"
                else:
                    ollama_status = "error"
                    ollama_error = f"Ollama returned status code {response.status_code}"
            except Exception as e:
                ollama_status = "connection_error"
                ollama_error = str(e)
        
        return {
            "ollama": {
                "status": ollama_status,
                "url": ollama_service.base_url if ollama_service else None,
                "model": ollama_service.model if ollama_service else None,
                "error": ollama_error
            },
            "vector_db": {
                "status": "online" if vector_service and vector_service.index else "offline",
                "document_count": len(vector_service.document_data) if vector_service and vector_service.document_data else 0
            },
            "logistics_analyzer": {
                "status": "online" if logistics_analyzer else "offline"
            },
            "data_processor": {
                "status": "online" if data_processor and hasattr(data_processor, 'df') else "offline",
                "record_count": len(data_processor.df) if data_processor and hasattr(data_processor, 'df') else 0
            },
            "environment": {
                "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
                "ollama_model": os.getenv("OLLAMA_MODEL", "mistral:7b")
            }
        }
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return {"error": str(e)}
    
@router.get("/test-ollama")
async def test_ollama_connection():
    """Test the connection to Ollama"""
    global ollama_service
    
    if ollama_service is None:
        return {"status": "error", "message": "Ollama service not initialized"}
    
    try:
        # Check connection to Ollama service
        connection_ok, error_msg = await ollama_service.check_connection()
        
        if connection_ok:
            # Try a simple generate call
            response = await ollama_service.generate(
                prompt="Respond with a single word: Hello",
                max_tokens=10
            )
            
            return {
                "status": "success" if "Hello" in response else "partial",
                "connection": "ok",
                "model": ollama_service.model,
                "url": ollama_service.base_url,
                "response": response
            }
        else:
            return {
                "status": "error",
                "connection": "failed",
                "model": ollama_service.model,
                "url": ollama_service.base_url,
                "error": error_msg
            }
    
    except Exception as e:
        logger.error(f"Error testing Ollama: {str(e)}")
        return {
            "status": "error",
            "connection": "exception",
            "model": ollama_service.model,
            "url": ollama_service.base_url,
            "error": str(e)
        }
    
# Request models
class AssistantQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class VisualizationRequest(BaseModel):
    query: str
    visualization_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# Response models
class AssistantResponse(BaseModel):
    answer: str
    context: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class SuggestionsResponse(BaseModel):
    suggestions: List[str]

class VisualizationResponse(BaseModel):
    visualization_type: str
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    description: str

# Services
class OllamaService:
    """Enhanced service for interacting with Ollama API with improved response quality"""
    
    def __init__(self, base_url="http://localhost:11434", model="mistral:7b"):
        """Initialize the Ollama service with base URL and model"""
        self.base_url = base_url
        self.model = model
        # Increase timeout to 3 minutes for more comprehensive responses
        self.client = httpx.AsyncClient(
            timeout=180.0,  
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        logger.info(f"Initialized Ollama service with model: {model}")
    
    async def check_connection(self):
        """Check if Ollama is accessible and the model is available"""
        try:
            # First check if the service is running at all
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Check if our model is available
            models_data = response.json()
            models = models_data.get("models", [])
            for model in models:
                if model.get("name") == self.model:
                    logger.info(f"Model {self.model} is available in Ollama")
                    return True, None
            
            # Model not found
            logger.warning(f"Model {self.model} not found in Ollama. Available models: {[m.get('name') for m in models]}")
            return False, f"Model {self.model} not found in Ollama. Available models: {[m.get('name') for m in models]}"
            
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            return False, f"Failed to connect to Ollama at {self.base_url}: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {str(e)}")
            return False, f"Error checking Ollama connection: {str(e)}"
    
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        """Generate text using Ollama API with improved timeout handling and response quality"""
        try:
            # Check connection first
            connection_ok, error_msg = await self.check_connection()
            if not connection_ok:
                logger.error(f"Cannot generate: {error_msg}")
                return f"I'm having trouble connecting to my AI backend: {error_msg}. Please ensure Ollama is running."
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.info(f"Sending request to Ollama API with prompt length: {len(prompt)} characters")
            
            # Add a retry mechanism with increasing timeouts
            max_retries = 3
            base_timeout = 180.0  # 3 minutes base timeout
            
            for attempt in range(max_retries):
                current_timeout = base_timeout * (attempt + 1)  # Increase timeout with each retry
                try:
                    logger.info(f"Attempt {attempt+1} with timeout {current_timeout}s")
                    
                    # Use a custom client for this request with adjusted timeout
                    async with httpx.AsyncClient(timeout=current_timeout) as client:
                        response = await client.post(
                            f"{self.base_url}/api/generate",
                            json=payload
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        if "response" in result:
                            # Post-process the response to improve formatting
                            processed_response = self._post_process_response(result["response"])
                            return processed_response
                        else:
                            logger.error(f"Unexpected Ollama response format: {result}")
                            return "I couldn't process that query. Please try again."
                    
                except asyncio.TimeoutError as e:
                    logger.warning(f"Request attempt {attempt + 1} timed out after {current_timeout}s: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        return "The request timed out. This query may be too complex for the current system constraints. Please try a simpler query."
                except httpx.RequestError as e:
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except httpx.RequestError as e:
            logger.error(f"Request error when connecting to Ollama: {str(e)}")
            return f"I'm having trouble connecting to the AI backend. Error: {str(e)}"
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama API: {e.response.status_code} - {e.response.text}")
            return f"The AI service returned an error: {e.response.status_code}. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I encountered an issue processing your query: {str(e)}. Please try again later."
    
    def _post_process_response(self, response):
        """Post-process the response to improve formatting and structure"""
        try:
            # Ensure sections have proper markdown formatting
            section_patterns = [
                (r'(?i)(\n|^)(summary:)', r'\1## Summary:\n'),
                (r'(?i)(\n|^)(key insights:)', r'\1## Key Insights:\n'),
                (r'(?i)(\n|^)(analysis:)', r'\1## Analysis:\n'),
                (r'(?i)(\n|^)(details:)', r'\1## Details:\n'),
                (r'(?i)(\n|^)(breakdown:)', r'\1## Breakdown:\n'),
                (r'(?i)(\n|^)(recommendations:)', r'\1## Recommendations:\n'),
                (r'(?i)(\n|^)(suggested visualization:)', r'\1## Suggested Visualization:\n')
            ]
            
            processed = response
            for pattern, replacement in section_patterns:
                import re
                processed = re.sub(pattern, replacement, processed)
            
            # Ensure numeric data has proper formatting
            # Format large numbers with commas if not already formatted
            number_pattern = r'(\b\d{4,}\b)(?![\d,])'
            
            def format_number(match):
                try:
                    num = int(match.group(1))
                    return f"{num:,}"
                except:
                    return match.group(1)
            
            processed = re.sub(number_pattern, format_number, processed)
            
            # Clean up any double newlines to make formatting more consistent
            processed = re.sub(r'\n{3,}', '\n\n', processed)
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error in post-processing response: {str(e)}")
            # Return original response if post-processing fails
            return response
    
    def generate_sync(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        """Generate text using Ollama API synchronously (as fallback)"""
        try:
            import requests
            
            # Check connection first
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    return f"Could not connect to Ollama at {self.base_url}. Please ensure the service is running."
                
                # Check if model exists
                models_data = response.json()
                models = models_data.get("models", [])
                model_exists = False
                for model in models:
                    if model.get("name") == self.model:
                        model_exists = True
                        break
                
                if not model_exists:
                    return f"Model {self.model} not found in Ollama. Available models: {[m.get('name') for m in models]}"
            except Exception as e:
                return f"Error checking Ollama connection: {str(e)}"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.info(f"Sending synchronous request to Ollama API with model {self.model}")
            
            # Increase timeout to 3 minutes
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180.0  
            )
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                # Post-process the response
                processed_response = self._post_process_response(result["response"])
                return processed_response
            else:
                logger.error(f"Unexpected Ollama response format: {result}")
                return "I couldn't process that query. Please try again."
                
        except Exception as e:
            logger.error(f"Error in synchronous generation: {str(e)}")
            return f"I encountered an issue generating a response: {str(e)}. Please ensure Ollama is running with model {self.model}."
    
    async def close(self):
        """Close the client session"""
        await self.client.aclose()

class FAISSVectorService:
    """Service for vector embeddings and similarity search using FAISS"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", index_path="./data/faiss_index"):
        """Initialize the FAISS vector service"""
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.data_path = os.path.join(os.path.dirname(index_path), "document_data.pkl")
        self.embedding_model = None
        self.index = None
        self.document_data = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        logger.info(f"Initialized FAISS Vector Service with model: {embedding_model}")
    
    def load_model(self):
        """Load the embedding model"""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                traceback.print_exc()
                return False
    
    def load_index(self):
        """Load the FAISS index if it exists"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.data_path):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                # Load document data
                import pickle
                with open(self.data_path, 'rb') as f:
                    self.document_data = pickle.load(f)
                logger.info(f"Loaded {len(self.document_data)} documents from {self.data_path}")
                return True
            else:
                logger.warning(f"No existing FAISS index found at {self.index_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            traceback.print_exc()
            return False
    
    def is_available(self):
        """Check if the vector service is properly initialized and available"""
        # Check if model can be loaded
        if self.embedding_model is None:
            model_loaded = self.load_model()
            if not model_loaded:
                logger.error("Vector service is not available: Failed to load embedding model")
                return False
        
        return True
    
    def create_index(self, df, text_columns, metadata_columns=None):
        """Create a new FAISS index from a dataframe"""
        try:
            # First check if everything is available
            if not self.is_available():
                logger.error("Cannot create index: Vector service is not available")
                return False
            
            # Prepare text documents from the dataframe
            documents = []
            for _, row in df.iterrows():
                # Extract text from specified columns
                text_parts = []
                for col in text_columns:
                    if col in row and pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")
                
                if text_parts:
                    document_text = " | ".join(text_parts)
                    
                    # Extract metadata
                    metadata = {}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row and pd.notna(row[col]):
                                # Convert NumPy types to Python native types
                                if isinstance(row[col], (np.int64, np.int32, np.int16, np.int8)):
                                    metadata[col] = int(row[col])
                                elif isinstance(row[col], (np.float64, np.float32, np.float16)):
                                    metadata[col] = float(row[col])
                                else:
                                    metadata[col] = row[col]
                    
                    documents.append({
                        "text": document_text,
                        "metadata": metadata
                    })
            
            if not documents:
                logger.error("No valid documents found in dataframe")
                return False
            
            # Create embeddings
            logger.info(f"Creating embeddings for {len(documents)} documents")
            texts = [doc["text"] for doc in documents]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents data
            self.document_data = documents
            
            # Save index and data
            logger.info(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            
            import pickle
            with open(self.data_path, 'wb') as f:
                pickle.dump(documents, f)
            
            logger.info(f"Created and saved FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            traceback.print_exc()
            return False
    
    def search(self, query, k=5):
        """Search for similar documents"""
        try:
            if self.index is None:
                success = self.load_index()
                if not success:
                    logger.error("Cannot search: Failed to load index")
                    return []
            
            if self.embedding_model is None:
                model_loaded = self.load_model()
                if not model_loaded:
                    logger.error("Cannot search: Failed to load embedding model")
                    return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search for similar documents
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
            
            # Collect results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_data) and idx >= 0:
                    results.append({
                        "text": self.document_data[idx]["text"],
                        "metadata": self.document_data[idx]["metadata"],
                        "score": float(distances[0][i])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            traceback.print_exc()
            return []

def get_status():
    """Get the current status of the assistant"""
    return {
        "status": "online" if ollama_service is not None else "offline",
        "details": {
            "ollama": {
                "status": "online" if ollama_service is not None else "offline",
                "model": ollama_service.model if ollama_service is not None else None
            },
            "vector_db": {
                "status": "online" if vector_service is not None and vector_service.index is not None else "offline",
                "document_count": len(vector_service.document_data) if vector_service is not None and vector_service.document_data else 0
            },
            "logistics_analyzer": {
                "status": "online" if logistics_analyzer is not None else "offline"
            },
            "data_processor": {
                "status": "online" if data_processor is not None else "offline",
                "record_count": len(data_processor.df) if data_processor is not None and hasattr(data_processor, 'df') else 0
            }
        }
    }

# New class for database-wide analytics
class LogisticsAnalyzer:
    """Service for comprehensive logistics data analysis"""
    
    def __init__(self, data_processor=None):
        """Initialize the analyzer with a data processor"""
        self.data_processor = data_processor
        
    def analyze_query(self, query):
        """Analyze the query to determine what type of analysis is needed"""
        query = query.lower()
        
        # Detect analysis type
        if any(x in query for x in ["destination", "where", "location", "country"]):
            return self.analyze_destinations()
        elif any(x in query for x in ["weight", "heavy", "light"]):
            return self.analyze_weights()
        elif any(x in query for x in ["carrier", "airline"]):
            return self.analyze_carriers()
        elif any(x in query for x in ["commodity", "product", "goods"]):
            return self.analyze_commodities()
        elif any(x in query for x in ["time", "trend", "date", "period", "month"]):
            return self.analyze_time_trends()
        elif any(x in query for x in ["risk", "delay", "issue"]):
            return self.analyze_risks()
        elif any(x in query for x in ["status", "delivered", "transit", "processing"]):
            return self.analyze_shipment_status()
        elif any(x in query for x in ["cost", "value", "price", "revenue"]):
            return self.analyze_financials()
        elif any(x in query for x in ["compare", "comparison"]):
            return self.analyze_comparisons(query)
        else:
            # General analysis
            return self.analyze_overall_logistics()
    
    def analyze_destinations(self):
        """Analyze shipment destinations"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            if 'DSTNTN' not in df.columns:
                return {"error": "Destination data not available"}
                
            # Get destination counts
            destinations = df['DSTNTN'].fillna('Unknown').value_counts().head(10)
            top_destinations = [{'name': str(name), 'value': int(count)} for name, count in destinations.items()]
            
            # Get destination by weight
            if 'GRSS_WGHT' in df.columns:
                # Convert weight to numeric
                df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Group by destination and calculate total weight
                dest_weight = df.groupby('DSTNTN')['weight_numeric'].sum().fillna(0).sort_values(ascending=False).head(10)
                top_destinations_by_weight = [{'name': str(name), 'value': float(weight)} for name, weight in dest_weight.items()]
            else:
                top_destinations_by_weight = []
            
            # Get countries
            if 'CONSGN_COUNTRY' in df.columns:
                countries = df['CONSGN_COUNTRY'].fillna('Unknown').value_counts().head(10)
                top_countries = [{'name': str(name), 'value': int(count)} for name, count in countries.items()]
            else:
                top_countries = []
            
            # Get destination regions
            regions = []
            if 'DSTNTN' in df.columns:
                # Extract regions from destinations where possible
                def extract_region(dest):
                    if pd.isna(dest):
                        return 'Unknown'
                    parts = str(dest).split(',')
                    if len(parts) > 1:
                        return parts[-1].strip()
                    return dest
                
                df['region'] = df['DSTNTN'].apply(extract_region)
                region_counts = df['region'].value_counts().head(5)
                regions = [{'name': str(name), 'value': int(count)} for name, count in region_counts.items()]
            
            return {
                "top_destinations": top_destinations,
                "top_destinations_by_weight": top_destinations_by_weight,
                "top_countries": top_countries,
                "top_regions": regions,
                "total_destinations": len(df['DSTNTN'].unique())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing destinations: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze destinations: {str(e)}"}
    
    def analyze_weights(self):
        """Analyze shipment weights"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            if 'GRSS_WGHT' not in df.columns:
                return {"error": "Weight data not available"}
                
            # Convert weight to numeric
            df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
            
            # Calculate weight statistics
            stats = {
                "total_weight": float(df['weight_numeric'].sum()),
                "average_weight": float(df['weight_numeric'].mean()),
                "median_weight": float(df['weight_numeric'].median()),
                "min_weight": float(df['weight_numeric'].min()),
                "max_weight": float(df['weight_numeric'].max())
            }
            
            # Create weight distribution
            weight_bins = [0, 50, 200, 500, float('inf')]
            weight_labels = ['0-50 kg', '51-200 kg', '201-500 kg', '501+ kg']
            
            # Create weight categories
            categories = pd.cut(
                df['weight_numeric'].dropna(), 
                bins=weight_bins, 
                labels=weight_labels, 
                right=False
            )
            
            # Calculate percentages
            counts = categories.value_counts(normalize=True) * 100
            weight_distribution = [
                {'name': str(name), 'value': float(value)}
                for name, value in counts.items()
            ]
            
            # Top heaviest shipments
            heaviest = df.nlargest(5, 'weight_numeric')
            heaviest_shipments = []
            for _, row in heaviest.iterrows():
                item = {
                    "weight": float(row['weight_numeric']),
                    "destination": str(row['DSTNTN']) if 'DSTNTN' in row else "Unknown"
                }
                
                if 'COMM_DESC' in row:
                    item["commodity"] = str(row['COMM_DESC'])
                    
                if 'AWB_NO' in row:
                    item["awb"] = str(row['AWB_NO'])
                    
                heaviest_shipments.append(item)
            
            # Weight by destination
            if 'DSTNTN' in df.columns:
                dest_weight = df.groupby('DSTNTN')['weight_numeric'].mean().fillna(0).sort_values(ascending=False).head(5)
                weight_by_destination = [
                    {'destination': str(dest), 'avg_weight': float(weight)}
                    for dest, weight in dest_weight.items()
                ]
            else:
                weight_by_destination = []
            
            return {
                "weight_stats": stats,
                "weight_distribution": weight_distribution,
                "heaviest_shipments": heaviest_shipments,
                "weight_by_destination": weight_by_destination
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weights: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze weights: {str(e)}"}
    
    def analyze_carriers(self):
        """Analyze carrier/airline information"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            if 'ARLN_DESC' not in df.columns:
                return {"error": "Carrier data not available"}
                
            # Get carrier counts
            carriers = df['ARLN_DESC'].fillna('Unknown').value_counts()
            top_carriers = [{'name': str(name), 'value': int(count)} for name, count in carriers.head(10).items()]
            
            # Calculate carrier market share
            total_shipments = len(df)
            carrier_share = df['ARLN_DESC'].fillna('Unknown').value_counts(normalize=True) * 100
            carrier_market_share = [
                {'name': str(name), 'percentage': float(share)}
                for name, share in carrier_share.head(5).items()
            ]
            
            # Weight by carrier
            if 'GRSS_WGHT' in df.columns:
                # Convert weight to numeric
                df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Calculate average weight per carrier
                carrier_weight = df.groupby('ARLN_DESC')['weight_numeric'].mean().fillna(0).sort_values(ascending=False)
                weight_by_carrier = [
                    {'carrier': str(carrier), 'avg_weight': float(weight)}
                    for carrier, weight in carrier_weight.head(5).items()
                ]
                
                # Calculate total weight per carrier
                carrier_total_weight = df.groupby('ARLN_DESC')['weight_numeric'].sum().fillna(0).sort_values(ascending=False)
                total_weight_by_carrier = [
                    {'carrier': str(carrier), 'total_weight': float(weight)}
                    for carrier, weight in carrier_total_weight.head(5).items()
                ]
            else:
                weight_by_carrier = []
                total_weight_by_carrier = []
            
            # Carrier by destination
            if 'DSTNTN' in df.columns:
                # Count routes by carrier-destination
                carrier_dest = df.groupby(['ARLN_DESC', 'DSTNTN']).size().reset_index(name='count')
                top_carrier_routes = carrier_dest.sort_values('count', ascending=False).head(5)
                carrier_destinations = [
                    {'carrier': str(row['ARLN_DESC']), 'destination': str(row['DSTNTN']), 'count': int(row['count'])}
                    for _, row in top_carrier_routes.iterrows()
                ]
            else:
                carrier_destinations = []
            
            return {
                "top_carriers": top_carriers,
                "carrier_market_share": carrier_market_share,
                "weight_by_carrier": weight_by_carrier,
                "total_weight_by_carrier": total_weight_by_carrier,
                "top_carrier_routes": carrier_destinations,
                "total_carriers": len(df['ARLN_DESC'].unique())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing carriers: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze carriers: {str(e)}"}
    
    def analyze_commodities(self):
        """Analyze commodity information"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            if 'COMM_DESC' not in df.columns:
                return {"error": "Commodity data not available"}
                
            # Get commodity counts
            commodities = df['COMM_DESC'].fillna('Unknown').value_counts()
            top_commodities = [{'name': str(name), 'value': int(count)} for name, count in commodities.head(10).items()]
            
            # Calculate commodity percentages
            commodity_percentage = df['COMM_DESC'].fillna('Unknown').value_counts(normalize=True) * 100
            commodity_breakdown = [
                {'name': str(name), 'value': float(percentage)}
                for name, percentage in commodity_percentage.head(5).items()
            ]
            
            # Add "Others" category if needed
            top5_total = sum(item['value'] for item in commodity_breakdown)
            if top5_total < 100:
                commodity_breakdown.append({
                    'name': 'Others',
                    'value': 100.0 - top5_total
                })
            
            # Weight by commodity
            if 'GRSS_WGHT' in df.columns:
                # Convert weight to numeric
                df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Calculate total weight per commodity
                commodity_weight = df.groupby('COMM_DESC')['weight_numeric'].sum().fillna(0).sort_values(ascending=False)
                weight_by_commodity = [
                    {'commodity': str(comm), 'total_weight': float(weight)}
                    for comm, weight in commodity_weight.head(5).items()
                ]
                
                # Calculate average weight per commodity
                avg_commodity_weight = df.groupby('COMM_DESC')['weight_numeric'].mean().fillna(0).sort_values(ascending=False)
                avg_weight_by_commodity = [
                    {'commodity': str(comm), 'avg_weight': float(weight)}
                    for comm, weight in avg_commodity_weight.head(5).items()
                ]
            else:
                weight_by_commodity = []
                avg_weight_by_commodity = []
            
            # Commodity by destination
            if 'DSTNTN' in df.columns:
                # Find top commodity for each destination
                top_comm_by_dest = df.groupby('DSTNTN')['COMM_DESC'].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown')
                top_commodity_by_destination = [
                    {'destination': str(dest), 'top_commodity': str(comm)}
                    for dest, comm in top_comm_by_dest.head(5).items()
                ]
            else:
                top_commodity_by_destination = []
            
            return {
                "top_commodities": top_commodities,
                "commodity_breakdown": commodity_breakdown,
                "weight_by_commodity": weight_by_commodity,
                "avg_weight_by_commodity": avg_weight_by_commodity,
                "top_commodity_by_destination": top_commodity_by_destination,
                "total_commodity_types": len(df['COMM_DESC'].unique())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commodities: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze commodities: {str(e)}"}
    
    def analyze_risks(self):
        """Analyze shipment risks and issues"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            # First try to use the data processor's risk assessment
            if hasattr(self.data_processor, 'generate_risk_assessment'):
                try:
                    risk_data = self.data_processor.generate_risk_assessment()
                    return risk_data
                except Exception as e:
                    logger.warning(f"Error using data processor's risk assessment: {str(e)}")
            
            # Fallback to basic risk analysis
            df = self.data_processor.df
            
            # Identify potential risk indicators
            risk_indicators = {
                "high_weight": 0,
                "unusual_destinations": 0,
                "uncommon_commodities": 0
            }
            
            # Check for high weight shipments
            if 'GRSS_WGHT' in df.columns:
                # Convert to numeric
                weights = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Define high weight threshold (e.g., top 5%)
                high_threshold = weights.quantile(0.95)
                risk_indicators["high_weight"] = int((weights > high_threshold).sum())
            
            # Check for unusual destinations
            if 'DSTNTN' in df.columns:
                destinations = df['DSTNTN'].value_counts()
                
                # Define unusual destinations as those with only 1-2 shipments
                unusual_dests = destinations[destinations <= 2].index.tolist()
                risk_indicators["unusual_destinations"] = len(unusual_dests)
            
            # Check for uncommon commodities
            if 'COMM_DESC' in df.columns:
                commodities = df['COMM_DESC'].value_counts()
                
                # Define uncommon commodities as those with only 1-2 shipments
                uncommon_comms = commodities[commodities <= 2].index.tolist()
                risk_indicators["uncommon_commodities"] = len(uncommon_comms)
            
            # Generate overall risk score (simple average of normalized indicators)
            total_shipments = len(df)
            risk_score = (
                (risk_indicators["high_weight"] / max(total_shipments, 1) * 100) +
                (risk_indicators["unusual_destinations"] / max(total_shipments, 1) * 100) +
                (risk_indicators["uncommon_commodities"] / max(total_shipments, 1) * 100)
            ) / 3
            
            # Determine risk level
            risk_level = "Low"
            if risk_score > 10:
                risk_level = "Medium"
            if risk_score > 25:
                risk_level = "High"
            
            return {
                "riskScore": float(risk_score),
                "overallRisk": risk_level,
                "riskIndicators": risk_indicators,
                "riskByRegion": self._analyze_risk_by_region()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risks: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze risks: {str(e)}"}
    
    def _analyze_risk_by_region(self):
        """Helper method to analyze risk by geographic region"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return []
                
            df = self.data_processor.df
            
            # Extract regions from destination
            if 'DSTNTN' not in df.columns:
                return []
                
            # Extract region from destination
            def extract_region(dest):
                if pd.isna(dest):
                    return 'Unknown'
                parts = str(dest).split(',')
                if len(parts) > 1:
                    return parts[-1].strip()
                return dest
            
            df['region'] = df['DSTNTN'].apply(extract_region)
            
            # Calculate risk score by region (based on unusual weight distributions)
            region_risks = []
            
            if 'GRSS_WGHT' in df.columns:
                # Convert to numeric
                df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Calculate weight stats by region
                region_stats = df.groupby('region')['weight_numeric'].agg(['mean', 'std', 'count']).reset_index()
                
                # Calculate coefficient of variation as a risk indicator
                region_stats['cv'] = region_stats['std'] / region_stats['mean']
                
                # Fill NaN values
                region_stats['cv'] = region_stats['cv'].fillna(0)
                
                # Normalize to 0-100 scale
                max_cv = max(region_stats['cv'].max(), 1)
                region_stats['risk_score'] = region_stats['cv'] / max_cv * 100
                
                # Sort by risk score and take top 5
                top_regions = region_stats.sort_values('risk_score', ascending=False).head(5)
                
                # Format for response
                for _, row in top_regions.iterrows():
                    region_risks.append({
                        "region": str(row['region']),
                        "riskScore": float(row['risk_score']),
                        "shipmentCount": int(row['count'])
                    })
            
            return region_risks
            
        except Exception as e:
            logger.error(f"Error analyzing risk by region: {str(e)}")
            return []
    
    def analyze_shipment_status(self):
        """Analyze shipment status information"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            # Check if we have status information
            status_columns = []
            for col in df.columns:
                if "STATUS" in col or "STAT" in col:
                    status_columns.append(col)
            
            if not status_columns:
                # Generate synthetic status based on date fields if available
                status_data = self._generate_synthetic_status()
                if status_data:
                    return status_data
                return {"error": "Status data not available"}
            
            # Use the first status column found
            status_col = status_columns[0]
            
            # Get status counts
            status_counts = df[status_col].fillna('Unknown').value_counts()
            status_distribution = [
                {'status': str(status), 'count': int(count)}
                for status, count in status_counts.items()
            ]
            
            # Calculate percentages
            total = status_counts.sum()
            status_percentages = (status_counts / total * 100).round(1)
            status_percentage_dist = [
                {'status': str(status), 'percentage': float(pct)}
                for status, pct in status_percentages.items()
            ]
            
            # Analyze status by destination if available
            status_by_destination = []
            if 'DSTNTN' in df.columns:
                # Get top 5 destinations
                top_dests = df['DSTNTN'].value_counts().head(5).index
                
                for dest in top_dests:
                    dest_df = df[df['DSTNTN'] == dest]
                    dest_status = dest_df[status_col].fillna('Unknown').value_counts()
                    
                    # Find most common status
                    most_common = dest_status.index[0] if len(dest_status) > 0 else 'Unknown'
                    
                    status_by_destination.append({
                        'destination': str(dest),
                        'most_common_status': str(most_common),
                        'percentage': float((dest_status[most_common] / len(dest_df) * 100).round(1))
                    })
            
            return {
                "status_distribution": status_distribution,
                "status_percentages": status_percentage_dist,
                "status_by_destination": status_by_destination
            }
            
        except Exception as e:
            logger.error(f"Error analyzing shipment status: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze shipment status: {str(e)}"}
    
    def _generate_synthetic_status(self):
        """Generate synthetic status data if real status not available"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return None
                
            df = self.data_processor.df
            
            # Look for date columns to infer status
            date_columns = []
            for col in df.columns:
                if "DATE" in col or "DT" in col:
                    date_columns.append(col)
            
            if not date_columns:
                return None
            
            # Try to find usable date columns
            date_data = {}
            for col in date_columns:
                try:
                    # Try to convert to datetime
                    df[f'{col}_dt'] = pd.to_datetime(df[col], errors='coerce')
                    valid_count = df[f'{col}_dt'].notna().sum()
                    
                    if valid_count > len(df) * 0.3:  # If at least 30% valid
                        date_data[col] = {
                            'column': f'{col}_dt',
                            'valid_count': valid_count
                        }
                except:
                    continue
            
            if not date_data:
                return None
            
            # Generate synthetic status based on date patterns
            now = pd.Timestamp.now()
            
            # Try to identify creation/shipping date
            creation_col = None
            delivery_col = None
            
            for col, data in date_data.items():
                # Assume earlier dates are creation dates
                if creation_col is None or df[data['column']].min() < df[date_data[creation_col]['column']].min():
                    creation_col = col
                
                # Assume later dates are delivery dates
                if delivery_col is None or df[data['column']].max() > df[date_data[delivery_col]['column']].max():
                    delivery_col = col
            
            # Assign synthetic status
            df['synthetic_status'] = 'Unknown'
            
            if creation_col and delivery_col and creation_col != delivery_col:
                # If both creation and delivery dates exist
                creation_dt = date_data[creation_col]['column']
                delivery_dt = date_data[delivery_col]['column']
                
                # Delivered: Has both dates
                df.loc[(df[creation_dt].notna()) & (df[delivery_dt].notna()), 'synthetic_status'] = 'Delivered'
                
                # In Transit: Has creation date but not delivery date, and creation within last 30 days
                df.loc[(df[creation_dt].notna()) & (df[delivery_dt].isna()) & 
                       (now - df[creation_dt] < pd.Timedelta(days=30)), 'synthetic_status'] = 'In Transit'
                
                # Processing: Has creation date but not delivery date, and creation date is recent
                df.loc[(df[creation_dt].notna()) & (df[delivery_dt].isna()) & 
                       (now - df[creation_dt] < pd.Timedelta(days=7)), 'synthetic_status'] = 'Processing'
                
                # Delayed: Has creation date but not delivery date, and creation date is old
                df.loc[(df[creation_dt].notna()) & (df[delivery_dt].isna()) & 
                       (now - df[creation_dt] > pd.Timedelta(days=30)), 'synthetic_status'] = 'Delayed'
            
            elif creation_col:
                # If only creation date exists
                creation_dt = date_data[creation_col]['column']
                
                # Assume delivered if creation date is older than 30 days
                df.loc[(df[creation_dt].notna()) & 
                       (now - df[creation_dt] > pd.Timedelta(days=30)), 'synthetic_status'] = 'Delivered'
                
                # Assume in transit if creation date is between 7 and 30 days
                df.loc[(df[creation_dt].notna()) & 
                       (now - df[creation_dt] > pd.Timedelta(days=7)) &
                       (now - df[creation_dt] <= pd.Timedelta(days=30)), 'synthetic_status'] = 'In Transit'
                
                # Assume processing if creation date is within last 7 days
                df.loc[(df[creation_dt].notna()) & 
                       (now - df[creation_dt] <= pd.Timedelta(days=7)), 'synthetic_status'] = 'Processing'
            
            # Get status counts
            status_counts = df['synthetic_status'].value_counts()
            status_distribution = [
                {'status': str(status), 'count': int(count)}
                for status, count in status_counts.items()
            ]
            
            # Calculate percentages
            total = status_counts.sum()
            status_percentages = (status_counts / total * 100).round(1)
            status_percentage_dist = [
                {'status': str(status), 'percentage': float(pct)}
                for status, pct in status_percentages.items()
            ]
            
            return {
                "status_distribution": status_distribution,
                "status_percentages": status_percentage_dist,
                "is_synthetic": True,
                "note": "Status data is synthetic, inferred from date patterns"
            }
            
        except Exception as e:
            logger.error(f"Error generating synthetic status: {str(e)}")
            return None
    
    def analyze_financials(self):
        """Analyze financial aspects of shipments"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            # Look for value/cost columns
            value_columns = []
            for col in df.columns:
                if any(term in col for term in ["VALUE", "COST", "PRICE", "REVENUE"]):
                    value_columns.append(col)
            
            if value_columns:
                # Use the first value column found
                value_col = value_columns[0]
                
                # Convert to numeric
                df['value_numeric'] = pd.to_numeric(df[value_col], errors='coerce')
                
                # Calculate value statistics
                stats = {
                    "total_value": float(df['value_numeric'].sum()),
                    "average_value": float(df['value_numeric'].mean()),
                    "median_value": float(df['value_numeric'].median()),
                    "min_value": float(df['value_numeric'].min()),
                    "max_value": float(df['value_numeric'].max())
                }
                
                # Value by destination
                value_by_destination = []
                if 'DSTNTN' in df.columns:
                    dest_value = df.groupby('DSTNTN')['value_numeric'].sum().fillna(0).sort_values(ascending=False)
                    value_by_destination = [
                        {'destination': str(dest), 'total_value': float(value)}
                        for dest, value in dest_value.head(5).items()
                    ]
                
                # Value by commodity
                value_by_commodity = []
                if 'COMM_DESC' in df.columns:
                    comm_value = df.groupby('COMM_DESC')['value_numeric'].sum().fillna(0).sort_values(ascending=False)
                    value_by_commodity = [
                        {'commodity': str(comm), 'total_value': float(value)}
                        for comm, value in comm_value.head(5).items()
                    ]
                
                return {
                    "value_stats": stats,
                    "value_by_destination": value_by_destination,
                    "value_by_commodity": value_by_commodity,
                    "value_column_used": value_col
                }
            
            # If no explicit value columns, estimate based on weight
            elif 'GRSS_WGHT' in df.columns:
                # Convert weight to numeric
                df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                
                # Estimate value based on weight (very rough approximation)
                base_rate = 1000  # Base value per shipment
                rate_per_kg = 200  # Additional value per kg
                
                df['estimated_value'] = base_rate + (df['weight_numeric'] * rate_per_kg)
                
                # Calculate estimated value statistics
                stats = {
                    "total_value": float(df['estimated_value'].sum()),
                    "average_value": float(df['estimated_value'].mean()),
                    "median_value": float(df['estimated_value'].median()),
                    "min_value": float(df['estimated_value'].min()),
                    "max_value": float(df['estimated_value'].max())
                }
                
                # Value by destination
                value_by_destination = []
                if 'DSTNTN' in df.columns:
                    dest_value = df.groupby('DSTNTN')['estimated_value'].sum().fillna(0).sort_values(ascending=False)
                    value_by_destination = [
                        {'destination': str(dest), 'total_value': float(value)}
                        for dest, value in dest_value.head(5).items()
                    ]
                
                # Value by commodity
                value_by_commodity = []
                if 'COMM_DESC' in df.columns:
                    comm_value = df.groupby('COMM_DESC')['estimated_value'].sum().fillna(0).sort_values(ascending=False)
                    value_by_commodity = [
                        {'commodity': str(comm), 'total_value': float(value)}
                        for comm, value in comm_value.head(5).items()
                    ]
                
                return {
                    "value_stats": stats,
                    "value_by_destination": value_by_destination,
                    "value_by_commodity": value_by_commodity,
                    "is_estimated": True,
                    "note": "Values are estimated based on weight"
                }
            
            return {"error": "No value or weight data available for financial analysis"}
            
        except Exception as e:
            logger.error(f"Error analyzing financials: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze financials: {str(e)}"}
    
    def analyze_comparisons(self, query):
        """Analyze comparisons requested in the query"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            # Detect what entities to compare
            entities_to_compare = []
            
            # Check for destination comparison
            if any(term in query for term in ["destination", "country", "location"]):
                entities_to_compare.append("destination")
            
            # Check for carrier comparison
            if any(term in query for term in ["carrier", "airline"]):
                entities_to_compare.append("carrier")
            
            # Check for commodity comparison
            if any(term in query for term in ["commodity", "product", "goods"]):
                entities_to_compare.append("commodity")
            
            # If no specific entities detected, use common ones
            if not entities_to_compare:
                entities_to_compare = ["destination", "carrier"]
            
            # Perform comparisons for each entity
            comparison_results = {}
            
            for entity in entities_to_compare:
                if entity == "destination" and 'DSTNTN' in df.columns:
                    # Get top destinations
                    top_dests = df['DSTNTN'].value_counts().head(5)
                    
                    # Compare by weight if available
                    weight_comparison = []
                    if 'GRSS_WGHT' in df.columns:
                        df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                        
                        for dest in top_dests.index:
                            dest_df = df[df['DSTNTN'] == dest]
                            
                            weight_comparison.append({
                                'destination': str(dest),
                                'shipment_count': int(top_dests[dest]),
                                'total_weight': float(dest_df['weight_numeric'].sum()),
                                'avg_weight': float(dest_df['weight_numeric'].mean())
                            })
                    
                    comparison_results["destination_comparison"] = weight_comparison
                
                elif entity == "carrier" and 'ARLN_DESC' in df.columns:
                    # Get top carriers
                    top_carriers = df['ARLN_DESC'].value_counts().head(5)
                    
                    # Compare by weight if available
                    carrier_comparison = []
                    if 'GRSS_WGHT' in df.columns:
                        df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                        
                        for carrier in top_carriers.index:
                            carrier_df = df[df['ARLN_DESC'] == carrier]
                            
                            carrier_comparison.append({
                                'carrier': str(carrier),
                                'shipment_count': int(top_carriers[carrier]),
                                'total_weight': float(carrier_df['weight_numeric'].sum()),
                                'avg_weight': float(carrier_df['weight_numeric'].mean())
                            })
                    
                    comparison_results["carrier_comparison"] = carrier_comparison
                
                elif entity == "commodity" and 'COMM_DESC' in df.columns:
                    # Get top commodities
                    top_comms = df['COMM_DESC'].value_counts().head(5)
                    
                    # Compare by weight if available
                    comm_comparison = []
                    if 'GRSS_WGHT' in df.columns:
                        df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                        
                        for comm in top_comms.index:
                            comm_df = df[df['COMM_DESC'] == comm]
                            
                            comm_comparison.append({
                                'commodity': str(comm),
                                'shipment_count': int(top_comms[comm]),
                                'total_weight': float(comm_df['weight_numeric'].sum()),
                                'avg_weight': float(comm_df['weight_numeric'].mean())
                            })
                    
                    comparison_results["commodity_comparison"] = comm_comparison
            
            # If no comparisons generated, return error
            if not comparison_results:
                return {"error": "No data available for comparison"}
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error analyzing comparisons: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to analyze comparisons: {str(e)}"}
    
    def analyze_overall_logistics(self):
        """Generate comprehensive analysis of logistics data"""
        try:
            if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                return {"error": "Data processor not available"}
                
            df = self.data_processor.df
            
            # Get basic summary statistics
            summary = {}
            if hasattr(self.data_processor, 'get_summary_statistics'):
                try:
                    summary = self.data_processor.get_summary_statistics()
                except Exception as e:
                    logger.warning(f"Error getting summary statistics: {str(e)}")
            
            if not summary:
                # Calculate basic summary manually
                summary = {
                    "activeShipments": len(df),
                    "totalWeight": float(pd.to_numeric(df['GRSS_WGHT'], errors='coerce').sum()) if 'GRSS_WGHT' in df.columns else 0,
                    "avgDeliveryTime": 5.2  # Placeholder value
                }
            
            # Get destinations
            destinations = {}
            if hasattr(self.data_processor, 'get_destination_analysis'):
                try:
                    destinations = self.data_processor.get_destination_analysis()
                except Exception as e:
                    logger.warning(f"Error getting destination analysis: {str(e)}")
            
            if not destinations and 'DSTNTN' in df.columns:
                # Calculate destinations manually
                top_dests = df['DSTNTN'].value_counts().head(10)
                destinations = [{'name': str(name), 'value': int(count)} for name, count in top_dests.items()]
            
            # Get weight distribution
            weight_dist = {}
            if hasattr(self.data_processor, 'get_weight_distribution'):
                try:
                    weight_dist = self.data_processor.get_weight_distribution()
                except Exception as e:
                    logger.warning(f"Error getting weight distribution: {str(e)}")
            
            # Get carrier data
            carrier_data = {}
            if 'ARLN_DESC' in df.columns:
                top_carriers = df['ARLN_DESC'].value_counts().head(5)
                carrier_data = [{'name': str(name), 'value': int(count)} for name, count in top_carriers.items()]
            
            # Get commodity data
            commodity_data = {}
            if 'COMM_DESC' in df.columns:
                top_commodities = df['COMM_DESC'].value_counts().head(5)
                commodity_data = [{'name': str(name), 'value': int(count)} for name, count in top_commodities.items()]
            
            # Get monthly trends
            monthly_trends = {}
            if hasattr(self.data_processor, 'get_monthly_trends'):
                try:
                    monthly_trends = self.data_processor.get_monthly_trends()
                except Exception as e:
                    logger.warning(f"Error getting monthly trends: {str(e)}")
            
            # Get risk assessment
            risk_data = {}
            if hasattr(self.data_processor, 'generate_risk_assessment'):
                try:
                    risk_data = self.data_processor.generate_risk_assessment()
                except Exception as e:
                    logger.warning(f"Error getting risk assessment: {str(e)}")
            
            return {
                "summary": summary,
                "destinations": destinations,
                "weight_distribution": weight_dist,
                "carriers": carrier_data,
                "commodities": commodity_data,
                "monthly_trends": monthly_trends,
                "risk_assessment": risk_data
            }
            
        except Exception as e:
            logger.error(f"Error performing overall analysis: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to perform overall analysis: {str(e)}"}
    
    def analyze_time_trends(self):
            """Analyze time-based trends"""
            try:
                if self.data_processor is None or not hasattr(self.data_processor, 'df'):
                    return {"error": "Data processor not available"}
                    
                df = self.data_processor.df
                
                # First try to use data processor's existing method if available
                if hasattr(self.data_processor, 'get_monthly_trends'):
                    try:
                        monthly_trends = self.data_processor.get_monthly_trends()
                        if monthly_trends:
                            return {"monthly_trends": monthly_trends}
                    except Exception as e:
                        logger.warning(f"Error using data processor's get_monthly_trends: {str(e)}")
                
                # Fallback to manual analysis
                date_columns = []
                for col in df.columns:
                    if "DATE" in col or "DT" in col:
                        date_columns.append(col)
                
                if not date_columns:
                    return {"error": "No date columns found for trend analysis"}
                
                # Try to find a usable date column
                usable_date_col = None
                for col in date_columns:
                    try:
                        # Try to convert to datetime
                        df[f'{col}_dt'] = pd.to_datetime(df[col], errors='coerce')
                        if df[f'{col}_dt'].notna().sum() > len(df) * 0.5:  # If at least 50% valid
                            usable_date_col = f'{col}_dt'
                            break
                    except:
                        continue
                
                if usable_date_col is None:
                    return {"error": "No usable date column found for trend analysis"}
                
                # Extract month and year
                df['month_year'] = df[usable_date_col].dt.strftime('%b %Y')
                
                # Group by month and count shipments
                monthly_counts = df.groupby('month_year').size().reset_index(name='shipments')
                
                # Sort by actual date order
                df['month_year_sort'] = df[usable_date_col].dt.strftime('%Y-%m')
                month_order = df.groupby('month_year')['month_year_sort'].first().sort_values().index
                monthly_counts = monthly_counts.set_index('month_year').loc[month_order].reset_index()
                
                # Limit to last 12 months
                monthly_counts = monthly_counts.tail(12)
                
                # Format for response
                monthly_shipments = [
                    {"month": str(row['month_year']), "shipments": int(row['shipments'])}
                    for _, row in monthly_counts.iterrows()
                ]
                
                # Calculate weight trends if weight data is available
                monthly_weights = []
                if 'GRSS_WGHT' in df.columns:
                    # Convert weight to numeric
                    df['weight_numeric'] = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                    
                    # Group by month and sum weights
                    monthly_weight_sums = df.groupby('month_year')['weight_numeric'].sum().reset_index()
                    
                    # Sort by actual date order
                    monthly_weight_sums = monthly_weight_sums.set_index('month_year').loc[month_order].reset_index()
                    
                    # Limit to last 12 months
                    monthly_weight_sums = monthly_weight_sums.tail(12)
                    
                    # Format for response
                    monthly_weights = [
                        {"month": str(row['month_year']), "total_weight": float(row['weight_numeric'])}
                        for _, row in monthly_weight_sums.iterrows()
                    ]
                
                # Calculate commodity trends
                commodity_trends = []
                if 'COMM_DESC' in df.columns:
                    # Get top 3 commodities
                    top_commodities = df['COMM_DESC'].value_counts().head(3).index.tolist()
                    
                    # For each top commodity, get monthly counts
                    for commodity in top_commodities:
                        commodity_df = df[df['COMM_DESC'] == commodity]
                        monthly_commodity = commodity_df.groupby('month_year').size().reset_index(name='count')
                        
                        # Sort by actual date order
                        monthly_commodity = monthly_commodity.set_index('month_year').loc[month_order].reset_index()
                        
                        # Limit to last 6 months
                        monthly_commodity = monthly_commodity.tail(6)
                        
                        # Add to trends list
                        commodity_trends.append({
                            "commodity": str(commodity),
                            "monthly_counts": [
                                {"month": str(row['month_year']), "count": int(row['count'])}
                                for _, row in monthly_commodity.iterrows()
                            ]
                        })
                
                return {
                    "monthly_shipments": monthly_shipments,
                    "monthly_weights": monthly_weights,
                    "commodity_trends": commodity_trends
                }
            except Exception as e:
                logger.error(f"Error analyzing time trends: {str(e)}")
                traceback.print_exc()
                return {"error": f"Failed to analyze time trends: {str(e)}"}
            
# Add these endpoint definitions to your assistant.py file at the module level
# (outside of any class, after the router definition)

@router.get("/status", response_model=StatusResponse)
async def assistant_status():
    """Get the status of the AI Assistant"""
    status_info = get_status()
    return {
        "status": status_info["status"],
        "details": status_info["details"]
    }

@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_query_suggestions():
    """Get suggested queries for the user"""
    # Add visualization-focused suggestions
    suggestions = [
        "What are the top shipping destinations in our logistics data?",
        "Show me a chart of package weights across different destinations",
        "Visualize the shipping trends over the past months",
        "Generate a pie chart of commodity types being shipped",
        "Display a visualization of carrier performance comparison",
        "Show me a graph of the distribution of shipment weights",
        "Create a chart showing the busiest shipping routes",
        "Visualize the average delivery times by destination",
        "Plot the relationship between package weight and shipping cost",
        "Generate a visualization of monthly shipping volumes",
        "Analyze our logistics data by destination",
        "Compare carrier performance across all shipments",
        "What are the risk factors in our entire logistics operation?",
        "Perform a comprehensive analysis of our commodity distribution",
        "Show me trends across all time periods in our dataset",
    ]
    
    return {
        "suggestions": suggestions
    }

@router.post("/init-vector-index")
async def initialize_vector_index():
    """Initialize the vector index with data from the CSV file"""
    global vector_service, data_processor
    
    if data_processor is None:
        raise HTTPException(status_code=503, detail="Data processor not initialized")
    
    try:
        # Check if vector service is available
        if vector_service is None:
            raise HTTPException(status_code=503, detail="Vector service not available")
        
        # Get data from data processor
        df = data_processor.df
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="No data available in data processor")
        
        # Define columns to use for text and metadata
        text_columns = [
            'DSTNTN', 'CONSGN_COUNTRY', 'COMM_DESC', 'ARLN_DESC',
            'EXPRTR_NAME', 'CONSGN_NAME', 'STTN_OF_ORGN'
        ]
        
        metadata_columns = [
            'AWB_NO', 'DSTNTN', 'CONSGN_COUNTRY', 'COMM_DESC', 
            'GRSS_WGHT', 'NO_OF_PKGS', 'ARLN_DESC'
        ]
        
        # Create index
        success = vector_service.create_index(df, text_columns, metadata_columns)
        
        if success:
            # Check if we have document data loaded
            doc_count = len(vector_service.document_data) if vector_service.document_data else 0
            
            return {
                "status": "success",
                "message": f"Vector index created successfully with {doc_count} documents",
                "document_count": doc_count
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create vector index")
        
    except Exception as e:
        logger.error(f"Error initializing vector index: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error initializing vector index: {str(e)}")

@router.post("/query", response_model=AssistantResponse)
async def assistant_query(request: AssistantQuery):
    """Process a query to the AI Assistant using comprehensive database analysis"""
    global logistics_analyzer, ollama_service
    
    if ollama_service is None:
        raise HTTPException(status_code=503, detail="AI Assistant services are not initialized")
    
    try:
        # Get query
        query = request.query
        
        # Detect if query is asking for visualization or might benefit from one
        needs_visualization = any(x in query.lower() for x in [
            "chart", "graph", "plot", "visualize", "visualization", "show me", 
            "display", "draw", "diagram", "trend", "compare", "distribution",
            "breakdown", "analyze", "analysis"  # Added more visualization triggers
        ])
        
        # Get similar documents for semantic context
        similar_docs = []
        if vector_service is not None and vector_service.index is not None:
            similar_docs = vector_service.search(query, k=5)  # Increased from 3 to 5 for better context
        
        # Prepare document context
        context_text = ""
        if similar_docs:
            context_text = "\n\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(similar_docs)])
        
        # Perform comprehensive data analysis based on the query
        analysis_results = {}
        analysis_text = ""
        
        if logistics_analyzer is not None:
            # Run comprehensive analysis on the entire database
            analysis_results = logistics_analyzer.analyze_query(query)
            
            # Convert analysis results to text for the AI
            if isinstance(analysis_results, dict) and not analysis_results.get("error"):
                analysis_text = "Here is comprehensive analysis of the entire database:"
                
                # Format the analysis results as text
                for key, value in analysis_results.items():
                    if isinstance(value, list) and len(value) > 0:
                        analysis_text += f"\n\n{key.replace('_', ' ').title()}:"
                        if len(value) <= 3:
                            # Show full details for small lists
                            for i, item in enumerate(value):
                                analysis_text += f"\n{i+1}. {json.dumps(item)}"
                        else:
                            # Show summary for longer lists
                            analysis_text += f"\n- {len(value)} items found"
                            analysis_text += f"\n- First few items: {json.dumps(value[:3])}"  # Increased from 2 to 3
                    elif isinstance(value, dict):
                        analysis_text += f"\n\n{key.replace('_', ' ').title()}:\n{json.dumps(value, indent=2)}"
                    else:
                        analysis_text += f"\n\n{key.replace('_', ' ').title()}: {value}"
        
        # Get summary statistics for context
        stats_text = "No summary statistics available"
        try:
            if data_processor is not None:
                summary_stats = data_processor.get_summary_statistics()
                stats_text = json.dumps(
                    {k: v for k, v in summary_stats.items() if k != "topDestinations"}, 
                    indent=2
                )
        except Exception as e:
            logger.warning(f"Error getting summary statistics: {str(e)}")
            stats_text = "Error retrieving summary statistics"
        
        # Build an improved prompt with better instructions
        system_prompt, user_prompt = prepare_enhanced_prompts(
            query=query,
            analysis_text=analysis_text,
            stats_text=stats_text,
            context_text=context_text
        )
        
        # Increase response timeout and token limits for more detailed responses
        try:
            response = await ollama_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000  # Increased from 1500
            )
        except Exception as e:
            logger.warning(f"Async generation failed: {str(e)}, falling back to sync method")
            # Fall back to synchronous method with increased timeout
            response = ollama_service.generate_sync(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000  # Increased from 1500
            )
        
        # Try to extract visualization recommendation from the response
        viz_recommendation = None
        recommended_viz_type = None
        
        if needs_visualization:
            viz_recommendation = extract_visualization_recommendation(response)
            if viz_recommendation:
                recommended_viz_type = viz_recommendation["visualization_type"]
                logger.info(f"Extracted visualization recommendation: {viz_recommendation['recommendation']}")
                logger.info(f"Recommended visualization type: {recommended_viz_type}")
            
        # Auto-generate visualization if needed or requested
        visualization_data = None
        if needs_visualization and logistics_analyzer is not None:
            try:
                # Use recommended visualization type if available, otherwise determine based on query content
                viz_type = recommended_viz_type if recommended_viz_type else "bar"  # Default
                
                if not recommended_viz_type:
                    # Determine visualization type based on query content if no recommendation
                    if "trend" in query.lower() or "time" in query.lower() or "period" in query.lower() or "month" in query.lower():
                        viz_type = "line"
                    elif "distribution" in query.lower() or "breakdown" in query.lower() or "proportion" in query.lower() or "percentage" in query.lower():
                        viz_type = "pie"
                
                # Use appropriate analysis result for visualization based on query
                viz_data = None
                
                # Check various data types that can be visualized
                if "destination" in query.lower() or "where" in query.lower() or "location" in query.lower() or "country" in query.lower():
                    dest_analysis = logistics_analyzer.analyze_destinations()
                    if not dest_analysis.get("error"):
                        viz_data = {
                            "visualization_type": viz_type,
                            "data": dest_analysis.get("top_destinations", []),
                            "config": {
                                "xKey": "name",
                                "yKey": "value",
                                "nameKey": "name",
                                "valueKey": "value",
                                "xLabel": "Destination",
                                "yLabel": "Number of Shipments"
                            },
                            "description": "Distribution of shipments by destination"
                        }
                elif "weight" in query.lower() or "heavy" in query.lower() or "kg" in query.lower():
                    weight_analysis = logistics_analyzer.analyze_weights()
                    if not weight_analysis.get("error"):
                        if viz_type == "pie":
                            viz_data = {
                                "visualization_type": "pie",
                                "data": weight_analysis.get("weight_distribution", []),
                                "config": {
                                    "nameKey": "name",
                                    "valueKey": "value"
                                },
                                "description": "Weight distribution of shipments"
                            }
                        else:
                            viz_data = {
                                "visualization_type": "bar",
                                "data": weight_analysis.get("weight_distribution", []),
                                "config": {
                                    "xKey": "name",
                                    "yKey": "value",
                                    "xLabel": "Weight Range",
                                    "yLabel": "Percentage (%)"
                                },
                                "description": "Weight distribution of shipments"
                            }
                elif "carrier" in query.lower() or "airline" in query.lower() or "transport" in query.lower():
                    carrier_analysis = logistics_analyzer.analyze_carriers()
                    if not carrier_analysis.get("error"):
                        viz_data = {
                            "visualization_type": viz_type,
                            "data": carrier_analysis.get("top_carriers", []),
                            "config": {
                                "xKey": "name",
                                "yKey": "value",
                                "nameKey": "name",
                                "valueKey": "value",
                                "xLabel": "Carrier",
                                "yLabel": "Number of Shipments"
                            },
                            "description": "Distribution of shipments by carrier"
                        }
                elif "commodity" in query.lower() or "product" in query.lower() or "goods" in query.lower() or "items" in query.lower():
                    commodity_analysis = logistics_analyzer.analyze_commodities()
                    if not commodity_analysis.get("error"):
                        if viz_type == "pie":
                            viz_data = {
                                "visualization_type": "pie",
                                "data": commodity_analysis.get("commodity_breakdown", []),
                                "config": {
                                    "nameKey": "name",
                                    "valueKey": "value"
                                },
                                "description": "Distribution of shipments by commodity"
                            }
                        else:
                            viz_data = {
                                "visualization_type": "bar",
                                "data": commodity_analysis.get("top_commodities", []),
                                "config": {
                                    "xKey": "name",
                                    "yKey": "value",
                                    "xLabel": "Commodity",
                                    "yLabel": "Number of Shipments"
                                },
                                "description": "Distribution of shipments by commodity"
                            }
                elif "time" in query.lower() or "trend" in query.lower() or "month" in query.lower() or "period" in query.lower():
                    time_analysis = logistics_analyzer.analyze_time_trends()
                    if not time_analysis.get("error") and "monthly_shipments" in time_analysis:
                        viz_data = {
                            "visualization_type": "line",
                            "data": time_analysis.get("monthly_shipments", []),
                            "config": {
                                "xKey": "month",
                                "yKey": "shipments",
                                "xLabel": "Month",
                                "yLabel": "Number of Shipments"
                            },
                            "description": "Monthly shipment trends"
                        }
                elif "risk" in query.lower() or "issue" in query.lower() or "problem" in query.lower() or "delay" in query.lower():
                    risk_analysis = logistics_analyzer.analyze_risks()
                    if not risk_analysis.get("error") and "riskByRegion" in risk_analysis:
                        viz_data = {
                            "visualization_type": "bar",
                            "data": risk_analysis.get("riskByRegion", []),
                            "config": {
                                "xKey": "region",
                                "yKey": "riskScore",
                                "xLabel": "Region",
                                "yLabel": "Risk Score"
                            },
                            "description": "Risk scores by geographic region"
                        }
                elif "compare" in query.lower() or "comparison" in query.lower():
                    comparison_results = logistics_analyzer.analyze_comparisons(query)
                    if not comparison_results.get("error"):
                        # Try to find suitable visualization data in comparison results
                        if "destination_comparison" in comparison_results:
                            viz_data = {
                                "visualization_type": "bar",
                                "data": comparison_results.get("destination_comparison", []),
                                "config": {
                                    "xKey": "destination",
                                    "yKey": "shipment_count",
                                    "xLabel": "Destination",
                                    "yLabel": "Shipment Count"
                                },
                                "description": "Comparison of destinations by shipment count"
                            }
                        elif "carrier_comparison" in comparison_results:
                            viz_data = {
                                "visualization_type": "bar",
                                "data": comparison_results.get("carrier_comparison", []),
                                "config": {
                                    "xKey": "carrier",
                                    "yKey": "shipment_count",
                                    "xLabel": "Carrier",
                                    "yLabel": "Shipment Count"
                                },
                                "description": "Comparison of carriers by shipment count"
                            }
                
                # If no specific visualization found but we want one, fall back to overall analysis
                if viz_data is None and needs_visualization:
                    overall_analysis = logistics_analyzer.analyze_overall_logistics()
                    if not overall_analysis.get("error"):
                        # Try to find any visualizable data
                        if "destinations" in overall_analysis and overall_analysis["destinations"]:
                            viz_data = {
                                "visualization_type": viz_type,
                                "data": overall_analysis["destinations"],
                                "config": {
                                    "xKey": "name",
                                    "yKey": "value",
                                    "nameKey": "name",
                                    "valueKey": "value",
                                    "xLabel": "Destination",
                                    "yLabel": "Number of Shipments"
                                },
                                "description": "Top shipping destinations"
                            }
                        elif "carriers" in overall_analysis and overall_analysis["carriers"]:
                            viz_data = {
                                "visualization_type": viz_type,
                                "data": overall_analysis["carriers"],
                                "config": {
                                    "xKey": "name",
                                    "yKey": "value",
                                    "nameKey": "name",
                                    "valueKey": "value",
                                    "xLabel": "Carrier",
                                    "yLabel": "Number of Shipments"
                                },
                                "description": "Top carriers by shipment count"
                            }
                        elif "commodities" in overall_analysis and overall_analysis["commodities"]:
                            viz_data = {
                                "visualization_type": viz_type,
                                "data": overall_analysis["commodities"],
                                "config": {
                                    "xKey": "name",
                                    "yKey": "value",
                                    "nameKey": "name",
                                    "valueKey": "value",
                                    "xLabel": "Commodity",
                                    "yLabel": "Number of Shipments"
                                },
                                "description": "Top commodities by shipment count"
                            }
                        elif "monthly_trends" in overall_analysis and overall_analysis["monthly_trends"]:
                            viz_data = {
                                "visualization_type": "line",
                                "data": overall_analysis["monthly_trends"],
                                "config": {
                                    "xKey": "month",
                                    "yKey": "shipments",
                                    "xLabel": "Month",
                                    "yLabel": "Number of Shipments"
                                },
                                "description": "Monthly shipment trends"
                            }
                
                # If we have visualization data and a recommendation, include it in the metadata
                if viz_data and viz_recommendation:
                    viz_data["ai_recommendation"] = viz_recommendation["recommendation"]
                    viz_data["is_ai_recommended"] = True
                
                visualization_data = viz_data
                
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}")
                traceback.print_exc()
                visualization_data = None
        
        # Determine if this used comprehensive analysis
        used_comprehensive_analysis = logistics_analyzer is not None and bool(analysis_results)
        
        # Return the response with enhanced context
        return {
            "answer": response,
            "context": {
                "similar_documents_count": len(similar_docs),
                "analysis_used": "comprehensive" if used_comprehensive_analysis else "basic",
                "used_full_database": used_comprehensive_analysis,
                "query_timestamp": datetime.now().isoformat(),
                "ai_recommended_visualization": viz_recommendation["recommendation"] if viz_recommendation else None,
                "response_length": len(response)
            },
            "visualization": visualization_data
        }
        
    except Exception as e:
        logger.error(f"Error processing assistant query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/visualize", response_model=VisualizationResponse)
async def generate_visualization(request: VisualizationRequest):
    """Generate visualization based on a specific query"""
    global logistics_analyzer
    
    if logistics_analyzer is None:
        raise HTTPException(status_code=503, detail="Analytics service is not initialized")
    
    try:
        query = request.query
        viz_type = request.visualization_type or "bar"
        
        # Determine which analysis to run based on the query
        viz_data = None
        
        if "destination" in query.lower() or "where" in query.lower():
            dest_analysis = logistics_analyzer.analyze_destinations()
            if not dest_analysis.get("error"):
                if viz_type == "pie":
                    viz_data = {
                        "visualization_type": "pie",
                        "data": dest_analysis.get("top_destinations", []),
                        "config": {
                            "nameKey": "name",
                            "valueKey": "value"
                        },
                        "description": "Distribution of shipments by destination"
                    }
                else:  # Default to bar
                    viz_data = {
                        "visualization_type": "bar",
                        "data": dest_analysis.get("top_destinations", []),
                        "config": {
                            "xKey": "name",
                            "yKey": "value",
                            "xLabel": "Destination",
                            "yLabel": "Number of Shipments"
                        },
                        "description": "Number of shipments by destination"
                    }
        
        elif "weight" in query.lower():
            weight_analysis = logistics_analyzer.analyze_weights()
            if not weight_analysis.get("error"):
                if viz_type == "pie":
                    viz_data = {
                        "visualization_type": "pie",
                        "data": weight_analysis.get("weight_distribution", []),
                        "config": {
                            "nameKey": "name",
                            "valueKey": "value"
                        },
                        "description": "Weight distribution of shipments"
                    }
                else:  # Default to bar
                    viz_data = {
                        "visualization_type": "bar",
                        "data": weight_analysis.get("weight_distribution", []),
                        "config": {
                            "xKey": "name",
                            "yKey": "value",
                            "xLabel": "Weight Range",
                            "yLabel": "Percentage (%)"
                        },
                        "description": "Weight distribution of shipments"
                    }
        
        elif "carrier" in query.lower() or "airline" in query.lower():
            carrier_analysis = logistics_analyzer.analyze_carriers()
            if not carrier_analysis.get("error"):
                if viz_type == "pie":
                    viz_data = {
                        "visualization_type": "pie",
                        "data": carrier_analysis.get("top_carriers", []),
                        "config": {
                            "nameKey": "name",
                            "valueKey": "value"
                        },
                        "description": "Distribution of shipments by carrier"
                    }
                else:  # Default to bar
                    viz_data = {
                        "visualization_type": "bar",
                        "data": carrier_analysis.get("top_carriers", []),
                        "config": {
                            "xKey": "name",
                            "yKey": "value",
                            "xLabel": "Carrier",
                            "yLabel": "Number of Shipments"
                        },
                        "description": "Number of shipments by carrier"
                    }
        
        elif "commodity" in query.lower() or "product" in query.lower():
            commodity_analysis = logistics_analyzer.analyze_commodities()
            if not commodity_analysis.get("error"):
                if viz_type == "pie":
                    viz_data = {
                        "visualization_type": "pie",
                        "data": commodity_analysis.get("commodity_breakdown", []),
                        "config": {
                            "nameKey": "name",
                            "valueKey": "value"
                        },
                        "description": "Distribution of shipments by commodity"
                    }
                else:  # Default to bar
                    viz_data = {
                        "visualization_type": "bar",
                        "data": commodity_analysis.get("top_commodities", []),
                        "config": {
                            "xKey": "name",
                            "yKey": "value",
                            "xLabel": "Commodity",
                            "yLabel": "Number of Shipments"
                        },
                        "description": "Number of shipments by commodity"
                    }
        
        elif "time" in query.lower() or "trend" in query.lower() or "month" in query.lower():
            time_analysis = logistics_analyzer.analyze_time_trends()
            if not time_analysis.get("error") and "monthly_shipments" in time_analysis:
                viz_data = {
                    "visualization_type": "line",
                    "data": time_analysis.get("monthly_shipments", []),
                    "config": {
                        "xKey": "month",
                        "yKey": "shipments",
                        "xLabel": "Month",
                        "yLabel": "Number of Shipments"
                    },
                    "description": "Monthly shipment trends"
                }
        
        # Fallback to general analysis if no specific visualization could be generated
        if viz_data is None:
            overall_analysis = logistics_analyzer.analyze_overall_logistics()
            if not overall_analysis.get("error"):
                if "destinations" in overall_analysis and overall_analysis["destinations"]:
                    viz_data = {
                        "visualization_type": viz_type,  # Use requested type
                        "data": overall_analysis["destinations"],
                        "config": {
                            "xKey": "name",
                            "yKey": "value",
                            "nameKey": "name",
                            "valueKey": "value",
                            "xLabel": "Category",
                            "yLabel": "Value"
                        },
                        "description": "General logistics data visualization"
                    }
        
        if viz_data is None:
            raise HTTPException(status_code=404, detail="Could not generate appropriate visualization for the query")
        
        return viz_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")