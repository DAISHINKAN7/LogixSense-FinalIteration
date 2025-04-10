"""
Vector Service for LogixSense AI Assistant.

This module provides vector embedding and similarity search functionality
using FAISS as the vector database.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorService:
    """Vector storage and search service using FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 index_path: str = "./data/faiss_index",
                 metadata_path: str = "./data/document_metadata.pkl"):
        """
        Initialize the FAISS vector service.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to save/load the FAISS index
            metadata_path: Path to save/load document metadata
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = None
        self.index = None
        self.document_metadata = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        logger.info(f"Initialized FAISS Vector Service with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                traceback.print_exc()
                raise
    
    def _save_index(self) -> None:
        """Save FAISS index and document metadata to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"FAISS index saved to {self.index_path}")
            
            if self.document_metadata:
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.document_metadata, f)
                logger.info(f"Document metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            traceback.print_exc()
    
    def load_index(self) -> bool:
        """
        Load FAISS index and document metadata from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                logger.warning("Index or metadata file not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            logger.info(f"FAISS index loaded from {self.index_path}")
            
            # Load document metadata
            with open(self.metadata_path, 'rb') as f:
                self.document_metadata = pickle.load(f)
            logger.info(f"Document metadata loaded: {len(self.document_metadata)} documents")
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            traceback.print_exc()
            return False
    
    def _extract_text_from_row(self, row: pd.Series, 
                              text_columns: List[str], 
                              metadata_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract text and metadata from a dataframe row.
        
        Args:
            row: DataFrame row
            text_columns: Columns to use for text content
            metadata_columns: Columns to include as metadata
        
        Returns:
            Dictionary with text and metadata
        """
        # Extract text from specified columns
        text_parts = []
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")
                
        # Skip if no text was extracted
        if not text_parts:
            return None
            
        # Create the document text
        document_text = " | ".join(text_parts)
        
        # Extract metadata
        metadata = {
            "id": str(row.name),  # Use row index as ID
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata_columns:
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    # Convert to appropriate type for serialization
                    if isinstance(row[col], (np.integer, np.floating, np.bool_)):
                        metadata[col] = row[col].item()
                    else:
                        metadata[col] = row[col]
        
        return {
            "text": document_text,
            "metadata": metadata
        }
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            NumPy array of embeddings
        """
        self._load_model()
        
        # Process in batches if there are many texts
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    def create_index_from_dataframe(self, df: pd.DataFrame, 
                                   text_columns: List[str],
                                   metadata_columns: Optional[List[str]] = None) -> bool:
        """
        Create a FAISS index from a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_columns: Columns to use for document text
            metadata_columns: Columns to include as metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the embedding model
            self._load_model()
            
            # Extract documents from the dataframe
            logger.info(f"Extracting text from DataFrame with {len(df)} rows")
            documents = []
            for idx, row in df.iterrows():
                doc = self._extract_text_from_row(row, text_columns, metadata_columns)
                if doc:
                    documents.append(doc)
            
            if not documents:
                logger.error("No valid documents extracted from DataFrame")
                return False
                
            logger.info(f"Extracted {len(documents)} valid documents")
            
            # Create embeddings
            texts = [doc["text"] for doc in documents]
            logger.info(f"Creating embeddings for {len(texts)} documents")
            embeddings = self._create_embeddings(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store document metadata
            self.document_metadata = documents
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Successfully created FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
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
        try:
            # Load model and index if not already loaded
            self._load_model()
            
            if self.index is None:
                success = self.load_index()
                if not success:
                    logger.error("Failed to load index for search")
                    return []
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(k, len(self.document_metadata))
            )
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_metadata) and idx >= 0:
                    # Calculate similarity score (convert distance to similarity)
                    similarity = 1.0 / (1.0 + distances[0][i])
                    
                    results.append({
                        "text": self.document_metadata[idx]["text"],
                        "metadata": self.document_metadata[idx]["metadata"],
                        "similarity": float(similarity)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            traceback.print_exc()
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add new documents to the existing index.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
                
            # Load model and index if not already loaded
            self._load_model()
            
            if self.index is None:
                success = self.load_index()
                if not success:
                    logger.error("Failed to load index for adding documents")
                    return False
            
            # Create embeddings for new documents
            texts = [doc["text"] for doc in documents]
            embeddings = self._create_embeddings(texts)
            
            # Add to index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Add to metadata
            self.document_metadata.extend(documents)
            
            # Save updated index and metadata
            self._save_index()
            
            logger.info(f"Added {len(documents)} new documents to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = {
                "document_count": len(self.document_metadata) if self.document_metadata else 0,
                "dimension": self.index.d if self.index else None,
                "index_type": type(self.index).__name__ if self.index else None,
                "embedding_model": self.model_name,
                "index_path": self.index_path,
                "metadata_path": self.metadata_path
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "error": str(e),
                "document_count": len(self.document_metadata) if self.document_metadata else 0
            }