"""
Cross-referencing functionality for C. Pete Connor's content generator.

This package provides tools for creating a vector database of content,
retrieving relevant past content during generation, and integrating
references into newly generated content.
"""

from .vector_db import ContentVectorDB
from .retrieval import ContentRetriever
from .integration import ReferenceIntegrator
