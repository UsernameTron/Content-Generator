"""
Tests for the cross-referencing functionality.

This module contains tests for the vector database, content retrieval,
and reference integration components of the cross-referencing system.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.cross_reference.vector_db import ContentVectorDB
from src.cross_reference.retrieval import ContentRetriever
from src.cross_reference.integration import ReferenceIntegrator
from src.cross_reference.cross_referencing_generator import CrossReferencingGenerator

class TestVectorDB(unittest.TestCase):
    """Tests for the ContentVectorDB class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.test_embeddings_dir = "tests/test_data/embeddings"
        os.makedirs(self.test_embeddings_dir, exist_ok=True)
        
        # Initialize with mocked sentence transformer
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = MagicMock()
            mock_model.encode.return_value.cpu.return_value.numpy.return_value = [[0.1] * 384]
            mock_st.return_value = mock_model
            
            self.vector_db = ContentVectorDB(embeddings_dir=self.test_embeddings_dir, device="cpu")
    
    def test_add_content(self):
        """Test adding content to the vector database."""
        with patch.object(self.vector_db, "_save_index") as mock_save:
            with patch.object(self.vector_db, "_split_into_chunks") as mock_split:
                # Mock split_into_chunks to return a single chunk
                mock_split.return_value = ["Test content"]
                
                # Add content
                count = self.vector_db.add_content(
                    content="Test content",
                    metadata={"source": "test"}
                )
                
                # Verify
                self.assertEqual(count, 1)
                self.assertEqual(len(self.vector_db.content_metadata), 1)
                self.assertEqual(self.vector_db.content_metadata[0]["text"], "Test content")
                mock_save.assert_called_once()
    
    def test_search(self):
        """Test searching for content."""
        # Mock the index search method
        self.vector_db.index.search = MagicMock(return_value=([1.0], [[0]]))
        
        # Add test metadata
        self.vector_db.content_metadata = [{"id": 0, "text": "Test content"}]
        
        # Search
        results = self.vector_db.search("query", k=1)
        
        # Verify
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Test content")
        self.assertIn("similarity", results[0])
    
    def test_split_into_chunks(self):
        """Test splitting text into chunks."""
        # Test with short text
        short_text = "This is a short text."
        chunks = self.vector_db._split_into_chunks(short_text, 10, 2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
        
        # Test with longer text
        long_text = "This is a longer text that should be split into multiple chunks based on size."
        chunks = self.vector_db._split_into_chunks(long_text, 5, 2)
        self.assertTrue(len(chunks) > 1)

class TestContentRetriever(unittest.TestCase):
    """Tests for the ContentRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the vector database
        self.mock_vector_db = MagicMock()
        self.retriever = ContentRetriever(self.mock_vector_db)
    
    def test_retrieve_relevant_content(self):
        """Test retrieving relevant content."""
        # Mock search results
        mock_results = [
            {"text": "Result 1", "similarity": 0.9},
            {"text": "Result 2", "similarity": 0.8},
            {"text": "Result 3", "similarity": 0.6}
        ]
        self.mock_vector_db.search.return_value = mock_results
        
        # Test with threshold filtering
        results = self.retriever.retrieve_relevant_content("query", min_similarity=0.7)
        
        # Verify
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Result 1")
        self.assertEqual(results[1]["text"], "Result 2")
    
    def test_retrieve_with_context(self):
        """Test retrieving with context."""
        # Mock search results
        mock_results = [{"text": "Result 1", "similarity": 0.9}]
        self.mock_vector_db.search.return_value = mock_results
        
        # Test with context
        context = {"audience": "technical", "platform": "blog"}
        results = self.retriever.retrieve_with_context("query", context, k=1)
        
        # Verify
        self.assertEqual(len(results), 1)
        self.mock_vector_db.search.assert_called_once()
        # Verify enhanced query contains context
        call_args = self.mock_vector_db.search.call_args[0][0]
        self.assertTrue("technical" in call_args)
        self.assertTrue("blog" in call_args)

class TestReferenceIntegrator(unittest.TestCase):
    """Tests for the ReferenceIntegrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the retriever
        self.mock_retriever = MagicMock()
        self.integrator = ReferenceIntegrator(self.mock_retriever)
    
    def test_enhance_prompt_with_references(self):
        """Test enhancing a prompt with references."""
        # Mock retrieval results
        mock_results = [
            {"text": "Reference text 1", "similarity": 0.9},
            {"text": "Reference text 2", "similarity": 0.8}
        ]
        self.mock_retriever.retrieve_with_context.return_value = mock_results
        
        # Test prompt enhancement
        enhanced_prompt = self.integrator.enhance_prompt_with_references(
            prompt="Original prompt",
            context={"audience": "general"}
        )
        
        # Verify
        self.assertTrue("Original prompt" in enhanced_prompt)
        self.assertTrue("Reference 1" in enhanced_prompt)
        self.assertTrue("Reference 2" in enhanced_prompt)
    
    def test_add_references_to_generated_content(self):
        """Test adding references to generated content."""
        # Test adding explicit citations
        content = "Generated content"
        references = [
            {"text": "Reference text 1", "similarity": 0.9},
            {"text": "Reference text 2", "similarity": 0.8}
        ]
        
        enhanced_content = self.integrator.add_references_to_generated_content(
            content=content,
            relevant_references=references,
            explicit_citation=True
        )
        
        # Verify
        self.assertTrue("Generated content" in enhanced_content)
        self.assertTrue("References" in enhanced_content)
        self.assertTrue("[1]" in enhanced_content)
        self.assertTrue("[2]" in enhanced_content)

if __name__ == "__main__":
    unittest.main()
