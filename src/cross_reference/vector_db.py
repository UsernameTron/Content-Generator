"""
Vector database for C. Pete Connor's content using embeddings.

This module handles the creation, storage, and querying of content embeddings
to enable cross-referencing during generation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContentVectorDB:
    """Vector database for content embeddings and retrieval."""
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        model_name: str = "all-MiniLM-L6-v2", 
        device: str = "auto"
    ):
        """
        Initialize the content vector database.
        
        Args:
            embeddings_dir: Directory to store embeddings and index
            model_name: Name of the sentence-transformer model to use
            device: Device to use for embeddings (auto, cpu, cuda, mps)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device} for content embeddings")
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index_path = self.embeddings_dir / "content_index.faiss"
        self.metadata_path = self.embeddings_dir / "content_metadata.json"
        
        # Content metadata
        self.content_metadata = []
        
        # Load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.content_metadata = json.load(f)
            logger.info(f"Loaded existing index with {len(self.content_metadata)} entries")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new index with dimension {self.embedding_dim}")
    
    def add_content(
        self, 
        content: str, 
        metadata: Dict, 
        chunk_size: int = 512, 
        chunk_overlap: int = 128
    ) -> int:
        """
        Add content to the vector database.
        
        Args:
            content: The text content to add
            metadata: Additional information about the content
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Number of chunks indexed
        """
        # Split content into chunks
        chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
        
        # Create embeddings for chunks
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        
        # Convert to numpy and add to index
        embeddings_np = embeddings.cpu().numpy()
        
        # Get current index size
        current_size = len(self.content_metadata)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "id": current_size + i,
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            self.content_metadata.append(chunk_metadata)
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"Added {len(chunks)} chunks to index")
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar content.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of content items with similarity scores
        """
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        
        # Search the index
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.content_metadata):
                result = self.content_metadata[idx].copy()
                # Ensure distances is properly processed as a float
                distance_value = float(distances[0][i]) if isinstance(distances[0], np.ndarray) else float(distances[0])
                result["similarity"] = float(1.0 - distance_value / 100.0)  # Normalize distance to similarity
                results.append(result)
        
        return results
    
    def _split_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
            
        tokens = text.split()
        chunks = []
        
        if len(tokens) <= chunk_size:
            return [text]
            
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk = ' '.join(tokens[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(tokens):
                break
                
        return chunks
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.content_metadata, f)
        logger.info(f"Saved index with {len(self.content_metadata)} entries")
    
    def index_from_training_data(self, data_path: str = "data/training_data.jsonl") -> int:
        """
        Index content from training data.
        
        Args:
            data_path: Path to training data file
            
        Returns:
            Number of items indexed
        """
        if not os.path.exists(data_path):
            logger.error(f"Training data file not found: {data_path}")
            return 0
            
        logger.info(f"Indexing content from {data_path}")
        count = 0
        
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages = data.get("messages", [])
                    
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if content:
                                metadata = {
                                    "source": "training_data",
                                    "file": data_path,
                                    "role": "assistant"
                                }
                                self.add_content(content, metadata)
                                count += 1
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    continue
        
        logger.info(f"Indexed {count} content items from training data")
        return count
