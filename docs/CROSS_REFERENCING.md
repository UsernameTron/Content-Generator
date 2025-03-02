# Cross-Referencing Functionality

This document describes the cross-referencing functionality implemented for the C. Pete Connor content generator. This feature allows the generator to reference and incorporate relevant past content during new content generation.

## Overview

The cross-referencing system consists of three core components:

1. **Vector Database (vector_db.py)**: Creates and manages embeddings for content, enabling semantic search.
2. **Content Retrieval (retrieval.py)**: Provides mechanisms for retrieving relevant past content.
3. **Reference Integration (integration.py)**: Integrates references to past content into newly generated content.

## Components

### 1. Vector Database

The `ContentVectorDB` class provides functionality to:
- Create a vector database of content using sentence-transformers
- Generate and store embeddings for content chunks
- Search for similar content based on semantic meaning
- Index content from existing training data

```python
from src.cross_reference.vector_db import ContentVectorDB

# Initialize the vector database
vector_db = ContentVectorDB()

# Add content to the database
vector_db.add_content(
    content="Your content text here",
    metadata={"source": "example", "platform": "blog"}
)

# Search for similar content
results = vector_db.search("Query text here", k=5)
```

### 2. Content Retrieval

The `ContentRetriever` class provides functionality to:
- Retrieve relevant content based on a query
- Filter results by similarity threshold
- Enhance retrieval with context (audience, platform, etc.)
- Prepare content specifically for generation

```python
from src.cross_reference.retrieval import ContentRetriever

# Initialize the retriever
retriever = ContentRetriever()

# Retrieve relevant content
results = retriever.retrieve_relevant_content(
    query="Query text here",
    k=3,
    min_similarity=0.7
)
```

### 3. Reference Integration

The `ReferenceIntegrator` class provides functionality to:
- Enhance prompts with relevant references
- Add references to generated content
- Detect opportunities for references
- Generate content with integrated references

```python
from src.cross_reference.integration import ReferenceIntegrator

# Initialize the integrator
integrator = ReferenceIntegrator()

# Enhance a prompt with references
enhanced_prompt = integrator.enhance_prompt_with_references(
    prompt="Original prompt text",
    context={"audience": "technical", "platform": "blog"}
)
```

## Cross-Referencing Generator

The `CrossReferencingGenerator` extends the base `ModelContentGenerator` and provides the following enhancements:

- Generates content with references to relevant past content
- Indexes existing content for cross-referencing
- Allows toggling cross-referencing on/off

```python
from src.cross_reference.cross_referencing_generator import CrossReferencingGenerator

# Initialize the generator
generator = CrossReferencingGenerator()

# Index existing content
generator.index_existing_content()

# Generate content with cross-references
content = generator.generate_content_with_references(
    prompt="Generate a blog post about artificial intelligence",
    audience="technical",
    platform="blog"
)
```

## Desktop Launcher

A desktop launcher script (`Pete_Connor_Cross_Reference_Indexing.command`) is provided to easily index existing content for cross-referencing. This script:

1. Indexes content from training and validation data
2. Provides logging information about the indexing process
3. Creates a vector database for future content generation

To use:
1. Double-click the launcher on your desktop
2. Wait for the indexing process to complete
3. Press Enter to close the window

## Implementation Details

- **Embeddings**: The system uses sentence-transformers to create embeddings for content chunks.
- **Vector Search**: FAISS is used for efficient vector search to find semantically similar content.
- **Chunk Management**: Content is divided into overlapping chunks for better retrieval.
- **Apple Silicon Optimization**: Environment variables are set for optimal performance on Apple Silicon.

## Dependencies

The following dependencies are required for the cross-referencing functionality:
- faiss-cpu (vector search)
- sentence-transformers (text embeddings)
- torch (tensor operations)
- pandas (data management)

These are automatically added to the requirements.txt file.

## Integration with Existing System

The cross-referencing functionality is designed to integrate seamlessly with the existing content generation system, providing enhanced content that leverages past work without requiring changes to the core generation process.
