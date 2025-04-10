"""
AI Assistant module for LogixSense platform.

This module implements the NLP-powered intelligent assistant that provides
natural language interfaces for logistics data analysis, query processing,
and insights generation.

Based on Ollama with Mistral 7B model and FAISS as the vector database.
"""

import os
import json
import httpx
import numpy as np
import pandas as pd
import asyncio
import logging
import traceback
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

# Request models
class AssistantQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

# Response models
class AssistantResponse(BaseModel):
    answer: str
    context: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class SuggestionsResponse(BaseModel):
    suggestions: List[str]

# Services
class OllamaService:
    """Service for interacting with Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434", model="mistral:7b"):
        """Initialize the Ollama service with base URL and model"""
        self.base_url = base_url
        self.model = model
        # Increase timeout to 2 minutes and set reasonable limits
        self.client = httpx.AsyncClient(
            timeout=120.0,  
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        logger.info(f"Initialized Ollama service with model: {model}")
    
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1000):
        """Generate text using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False  # Ensure this is False, not false (Python boolean)
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.info(f"Sending request to Ollama API at {self.base_url}/api/generate")
            
            # Add a retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=120.0  # Explicit timeout for this request
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "response" in result:
                        return result["response"]
                    else:
                        logger.error(f"Unexpected Ollama response format: {result}")
                        return "I couldn't process that query. Please try again."
                
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except httpx.RequestError as e:
            logger.error(f"Request error when connecting to Ollama: {str(e)}")
            return f"I'm having trouble connecting to my AI backend. Please ensure Ollama is running."
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama API: {e.response.status_code} - {e.response.text}")
            return f"The AI service returned an error. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I encountered an issue processing your query. Please try again later."
    
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
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
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
            return False
    
    def create_index(self, df, text_columns, metadata_columns=None):
        """Create a new FAISS index from a dataframe"""
        try:
            self.load_model()
            
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
            embeddings = self.embedding_model.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents data
            self.document_data = documents
            
            # Save index and data
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
                    return []
            
            if self.embedding_model is None:
                self.load_model()
            
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
            return []

# Global service instances
ollama_service = None
vector_service = None
data_processor = None

# Initialize services
def initialize_services(processor):
    """Initialize the assistant services"""
    global ollama_service, vector_service, data_processor
    
    try:
        # Store data processor
        data_processor = processor
        
        # Initialize Ollama service
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")
        ollama_service = OllamaService(base_url=ollama_url, model=ollama_model)
        
        # Initialize vector service
        vector_service = FAISSVectorService()
        
        # Try to load existing index
        success = vector_service.load_index()
        
        logger.info(f"AI Assistant services initialized successfully. FAISS index loaded: {success}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing AI Assistant services: {str(e)}")
        traceback.print_exc()
        return False

def get_status():
    """Get the current status of the assistant"""
    return {
        "ollama": {
            "status": "online" if ollama_service is not None else "offline",
            "model": ollama_service.model if ollama_service is not None else None,
            "url": ollama_service.base_url if ollama_service is not None else None
        },
        "vector_db": {
            "status": "online" if vector_service is not None and vector_service.index is not None else "offline",
            "document_count": len(vector_service.document_data) if vector_service is not None and vector_service.document_data else 0
        },
        "data_processor": {
            "status": "online" if data_processor is not None else "offline",
            "record_count": len(data_processor.df) if data_processor is not None and data_processor.df is not None else 0
        }
    }

def generate_sync(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1000):
    """Generate text using Ollama API synchronously (as fallback)"""
    try:
        import requests
        
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
        
        logger.info(f"Sending synchronous request to Ollama API")
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120.0
        )
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            return result["response"]
        else:
            logger.error(f"Unexpected Ollama response format: {result}")
            return "I couldn't process that query. Please try again."
            
    except Exception as e:
        logger.error(f"Error in synchronous generation: {str(e)}")
        return "I encountered an issue generating a response. Please try again later."

# Router endpoints
@router.get("/status", response_model=StatusResponse)
async def assistant_status():
    """Get the status of the AI Assistant"""
    if ollama_service is None or vector_service is None:
        return {
            "status": "offline",
            "message": "AI Assistant services are not initialized"
        }
    
    return {
        "status": "online",
        "details": get_status()
    }

@router.post("/query", response_model=AssistantResponse)
async def assistant_query(request: AssistantQuery):
    """Process a query to the AI Assistant"""
    if ollama_service is None or vector_service is None:
        raise HTTPException(status_code=503, detail="AI Assistant services are not initialized")
    
    try:
        # Get query
        query = request.query
        
        # Get similar documents for context
        similar_docs = vector_service.search(query, k=5)
        
        # Prepare prompt with context
        context_text = "\n\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(similar_docs)])
        
        # Get summary statistics for context
        try:
            if data_processor is not None:
                summary_stats = data_processor.get_summary_statistics()
                stats_text = json.dumps(
                    {k: v for k, v in summary_stats.items() if k != "topDestinations"}, 
                    indent=2
                )
            else:
                stats_text = "No summary statistics available"
        except Exception as e:
            logger.warning(f"Error getting summary statistics: {str(e)}")
            stats_text = "Error retrieving summary statistics"
        
        # Build the prompt
        system_prompt = """You are LogixSense AI Assistant, a specialist in logistics and supply chain analytics.
Your goal is to provide detailed, accurate information about logistics operations based on the shipping data.
When analyzing data, provide specific metrics, trends, and actionable insights.
Always maintain a professional, data-focused approach while being conversational and helpful."""
        
        user_prompt = f"""I need information about our logistics operations.

Query: {query}

Here is some context from our logistics database:

Summary Statistics:
{stats_text}

Relevant Documents:
{context_text}

Please provide a detailed, data-driven response focusing on specific metrics and actionable insights.
Include exact numbers where possible, and make comparisons between different aspects of the logistics operations."""
        
        # Try async generation first
        try:
            response = await ollama_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1500
            )
        except Exception as e:
            logger.warning(f"Async generation failed: {str(e)}, falling back to sync method")
            # Fall back to synchronous method
            response = ollama_service.generate_sync(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1500
            )
        
        # Return the response
        return {
            "answer": response,
            "context": {
                "similar_documents_count": len(similar_docs),
                "query_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing assistant query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/init-vector-index")
async def initialize_vector_index():
    """Initialize the vector index with data from the CSV file"""
    if data_processor is None:
        raise HTTPException(status_code=503, detail="Data processor not initialized")
    
    try:
        # Initialize vector service if not already done
        global vector_service
        if vector_service is None:
            vector_service = FAISSVectorService()
        
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
            return {
                "status": "success",
                "message": f"Vector index created successfully with {len(df)} documents",
                "index_location": vector_service.index_path
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create vector index")
        
    except Exception as e:
        logger.error(f"Error initializing vector index: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error initializing vector index: {str(e)}")

@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_query_suggestions():
    """Get suggested queries for the user"""
    suggestions = [
        "What are the top shipping destinations in our logistics data?",
        "Show me the distribution of package weights across different destinations",
        "Analyze the shipping times for different carriers",
        "What are the trends in commodity types being shipped?",
        "Which exporters have the highest volume of shipments?",
        "Identify potential delivery issues in our shipping data",
        "Compare shipping performance between different countries",
        "What's the average weight of shipments to European destinations?",
        "Analyze customs clearance times for different destinations",
        "What shipping routes have the longest transit times?"
    ]
    
    return {
        "suggestions": suggestions
    }

# In your assistant.py
print("Assistant module imported, debugging module structure:")
print(f"Module: {__name__}")
print(f"Router defined: {'router' in locals() or 'router' in globals()}")
print(f"OllamaService defined: {'OllamaService' in locals() or 'OllamaService' in globals()}")
print(f"FAISSVectorService defined: {'FAISSVectorService' in locals() or 'FAISSVectorService' in globals()}")