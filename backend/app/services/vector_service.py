"""
Qdrant Vector Service for LogixSense AI Assistant.

This module provides enhanced vector embedding and similarity search functionality
using Qdrant as the vector database with full-dataset analysis capabilities.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantService:
    """Vector storage and search service using Qdrant."""
    
    def __init__(self, 
                 url: str = "http://localhost:6333", 
                 collection_name: str = "logistics_data",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Qdrant service.
        
        Args:
            url: URL of the Qdrant server
            collection_name: Name of the collection to use
            embedding_model: Name of the sentence transformer model
        """
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available. Install with 'pip install qdrant-client sentence-transformers'")
            self.available = False
            return
            
        self.url = url
        self.collection_name = collection_name
        self.model_name = embedding_model
        self.embedding_model = None
        self.client = None
        self.available = True
        
        # Try to initialize the client
        try:
            self.client = QdrantClient(url=url)
            logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            self.available = False
    
    def _load_model(self) -> bool:
        """Load the embedding model"""
        if not self.available:
            return False
            
        if self.embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                traceback.print_exc()
                self.available = False
                return False
        return True
    
    def create_collection(self) -> bool:
        """Create the collection if it doesn't exist"""
        if not self.available or self.client is None:
            return False
            
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create the collection
                self._load_model()  # Load model to get vector size
                vector_size = self.embedding_model.get_sentence_embedding_dimension()
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {self.collection_name} with vector size {vector_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.available or self.client is None:
            return {"error": "Qdrant service not available"}
            
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "status": "available",
                "name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.vectors_count,
                "distance": str(collection_info.config.params.vectors.distance),
                "server_url": self.url
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "name": self.collection_name,
                "server_url": self.url
            }
    
    def _extract_text_and_metadata(self, row: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from a row"""
        # Extract text from all columns
        text_parts = []
        for col, value in row.items():
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        
        text = " | ".join(text_parts)
        
        # Extract metadata - convert numpy types to Python types
        metadata = {}
        for col, value in row.items():
            if pd.notna(value):
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    metadata[col] = value.item()
                else:
                    metadata[col] = str(value)
        
        return text, metadata
    
    def ingest_logistics_data(self, df: pd.DataFrame) -> bool:
        """
        Ingest logistics data into Qdrant.
        
        Args:
            df: DataFrame containing logistics data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available or self.client is None:
            logger.error("Qdrant service not available")
            return False
            
        if df is None or df.empty:
            logger.error("No data to ingest")
            return False
            
        try:
            # Create collection if it doesn't exist
            if not self.create_collection():
                return False
                
            # Load embedding model
            if not self._load_model():
                return False
                
            # Process and upload in batches
            batch_size = 100
            start_idx = 0
            total_points = 0
            
            while start_idx < len(df):
                batch_df = df.iloc[start_idx:start_idx + batch_size]
                
                # Prepare batch data
                batch_texts = []
                batch_metadatas = []
                batch_ids = []
                
                for idx, row in batch_df.iterrows():
                    text, metadata = self._extract_text_and_metadata(row)
                    batch_texts.append(text)
                    batch_metadatas.append(metadata)
                    batch_ids.append(str(idx))  # Use row index as ID
                
                # Create embeddings
                batch_embeddings = self.embedding_model.encode(batch_texts)
                
                # Upload points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=models.Batch(
                        ids=batch_ids,
                        vectors=batch_embeddings.tolist(),
                        payloads=batch_metadatas
                    )
                )
                
                total_points += len(batch_df)
                logger.info(f"Ingested batch: {start_idx} to {start_idx + len(batch_df) - 1}")
                start_idx += batch_size
            
            logger.info(f"Successfully ingested {total_points} points into {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data: {str(e)}")
            traceback.print_exc()
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        if not self.available or self.client is None:
            logger.error("Qdrant service not available")
            return []
            
        try:
            # Load embedding model
            if not self._load_model():
                return []
                
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search collection
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            traceback.print_exc()
            return []
    
    def analyze_collection(self) -> Dict[str, Any]:
        """Analyze the collection to get statistics"""
        if not self.available or self.client is None:
            return {"error": "Qdrant service not available"}
            
        try:
            # Get collection info
            collection_info = self.get_collection_info()
            
            # Get sample points for analysis
            sample_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Extract field names from payloads
            fields = set()
            for point in sample_points:
                if point.payload:
                    fields.update(point.payload.keys())
            
            return {
                "collection_info": collection_info,
                "fields": list(fields),
                "sample_count": len(sample_points)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing collection: {str(e)}")
            return {"error": str(e)}