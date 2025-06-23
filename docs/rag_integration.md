# RAG (Retrieval Augmented Generation) Integration

## Overview

The RAG integration enables assistants to access and reference uploaded documents during conversations. This allows for more accurate, contextual responses based on your specific knowledge base.

## Architecture

### Components

1. **Document Storage**: Documents are stored in AWS S3 for durability and scalability
2. **Text Processing**: LangChain processes documents, extracting text and splitting into chunks
3. **Embeddings**: OpenAI embeddings convert text chunks into vector representations
4. **Vector Database**: pgvector stores embeddings for fast similarity search
5. **Retrieval**: Relevant document chunks are retrieved based on user queries
6. **Generation**: LLMs generate responses enhanced with retrieved context

### Database Models

- **Document**: Stores document metadata and S3 references
- **DocumentChunk**: Text chunks with embeddings
- **VectorStore**: pgvector embeddings for semantic search

## Features

✅ **Document Upload & Processing**
- Supports PDF, DOCX, TXT, and Markdown files
- Automatic text extraction and chunking
- Configurable chunking strategies (recursive, character, semantic)
- Deduplication based on file hash

✅ **Intelligent Retrieval**
- Semantic similarity search using embeddings
- Configurable similarity thresholds
- Per-assistant document isolation
- Context-aware document selection

✅ **LangChain Integration**
- Works with all LangChain-supported providers
- Automatic context injection into system prompts
- Seamless integration with existing conversations

✅ **Production Ready**
- Asynchronous processing for better performance
- Error handling and fallback mechanisms
- Progress tracking for document processing
- S3 integration for scalable storage

## Configuration

### Assistant RAG Settings

Each assistant can be configured with RAG-specific settings:

```python
assistant.rag_settings = {
    "enabled": True,                           # Enable/disable RAG
    "search_limit": 3,                         # Number of chunks to retrieve
    "similarity_threshold": 0.7,               # Minimum similarity score
    "embedding_model": "text-embedding-3-small", # OpenAI embedding model
    "chunking_strategy": "recursive",          # Text splitting strategy
    "chunk_size": 1000,                       # Size of text chunks
    "chunk_overlap": 200,                     # Overlap between chunks
    "auto_process": True,                     # Auto-process uploads
    "include_metadata": True,                 # Include doc metadata
    "context_window_tokens": 4000            # Max context tokens
}
```

### Environment Variables

```bash
# Required for embeddings
OPENAI_API_KEY=your_openai_api_key

# Required for document storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1

# Required for vector storage
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## Usage

### 1. Upload Documents

```python
from app.services.rag_service import RAGService
from app.db.models import Assistant

# Initialize RAG service
rag_service = RAGService(db_session=db)

# Upload and process document
with open("knowledge_base.pdf", "rb") as f:
    document = await rag_service.upload_and_process_document(
        file_data=f.read(),
        filename="knowledge_base.pdf",
        content_type="application/pdf",
        assistant_id=assistant.id,
        organization_id=assistant.organization_id,
        name="Company Knowledge Base",
        category="policies",
        tags=["hr", "policies", "procedures"]
    )
```

### 2. Check Processing Status

```python
# Get document processing status
status = rag_service.get_document_status(document.id)
print(f"Processing: {status['progress_percentage']}%")
print(f"Status: {status['processing_status']}")
```

### 3. Search Documents

```python
# Search for relevant content
results = await rag_service.search_documents(
    query="What is the vacation policy?",
    assistant_id=assistant.id,
    limit=5,
    similarity_threshold=0.7
)

for result in results:
    print(f"Document: {result['document']['name']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity_score']}")
```

### 4. Automatic RAG Integration

Once documents are uploaded and processed, RAG automatically enhances conversations:

```python
# RAG is automatically used in LLM conversations
llm_service = LLMService(assistant=assistant)

# This will automatically retrieve relevant documents
await llm_service.process_transcript(
    transcript="What are the company holidays?",
    is_final=True,
    metadata={},
    response_callback=your_callback
)
```

## Document Processing

### Supported File Types

| Type | Extension | MIME Type |
|------|-----------|-----------|
| PDF | `.pdf` | `application/pdf` |
| Word | `.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| Text | `.txt` | `text/plain` |
| Markdown | `.md` | `text/markdown` |

### Chunking Strategies

#### Recursive Character Splitter (Default)
- Splits text hierarchically: paragraphs → sentences → words
- Best for most document types
- Preserves semantic boundaries

```python
chunking_strategy = "recursive"
chunk_size = 1000
chunk_overlap = 200
```

#### Character Splitter
- Simple splitting based on character count
- Faster but less semantic awareness

```python
chunking_strategy = "character"
chunk_size = 1000
chunk_overlap = 200
```

#### Semantic Splitter
- Uses embeddings to determine semantic boundaries
- More intelligent but slower processing

```python
chunking_strategy = "semantic"
# chunk_size and chunk_overlap are less relevant for semantic splitting
```

## How RAG Enhances Conversations

### Before RAG
```
User: "What is our refund policy?"
Assistant: "I don't have specific information about your refund policy. Generally, refund policies vary by company..."
```

### With RAG
```
User: "What is our refund policy?"
Assistant: "Based on our company policy document, our refund policy states that customers can request a full refund within 30 days of purchase. The refund process takes 5-7 business days to complete..."
```

### Context Injection

RAG automatically enhances the system prompt with relevant context:

```
Original System Prompt:
"You are a helpful customer service assistant."

Enhanced with RAG:
"You are a helpful customer service assistant.

## Available Document Context:
From document 'Customer Service Manual': Refunds are processed within 30 days...
From document 'Company Policies': Our return policy allows...

Use this context when relevant to provide accurate responses."
```

## API Examples

### Document Management

```python
# List assistant documents
documents = rag_service.get_assistant_documents(
    assistant_id=assistant.id,
    include_processing=False,
    limit=10
)

# Delete document
success = await rag_service.delete_document(
    document_id=doc.id,
    assistant_id=assistant.id
)
```

### Advanced Search

```python
# Search with metadata filtering
results = await rag_service.search_documents(
    query="employee benefits",
    assistant_id=assistant.id,
    limit=5,
    similarity_threshold=0.8,
    include_metadata=True
)

# Filter by document category or tags (if implemented)
hr_docs = [doc for doc in documents if doc.category == "hr"]
```

## Performance Considerations

### Optimal Chunk Sizes
- **Small chunks** (500-1000 chars): Better precision, more results
- **Large chunks** (1500-2000 chars): Better context, fewer results
- **Overlap**: 10-20% of chunk size for continuity

### Embedding Models
- **text-embedding-3-small**: Fast, cost-effective, good quality
- **text-embedding-3-large**: Higher quality, more expensive
- **text-embedding-ada-002**: Legacy, still reliable

### Search Optimization
- **Similarity threshold**: 0.7-0.8 for most use cases
- **Search limit**: 3-5 chunks for balanced context
- **Context window**: Consider LLM's token limits

## Troubleshooting

### Common Issues

#### Documents Not Processing
```python
# Check document status
status = rag_service.get_document_status(document_id)
if status['processing_status'] == 'failed':
    print(f"Error: {status['processing_error']}")
```

#### No Relevant Results
- Lower similarity threshold
- Check document content quality
- Verify embeddings are generated
- Review search query specificity

#### Performance Issues
- Reduce chunk overlap
- Optimize chunk sizes
- Use smaller embedding models
- Implement caching

### Monitoring

```python
# Check RAG availability
if not RAG_AVAILABLE:
    print("RAG dependencies not installed")

# Monitor document processing
progress = document.get_processing_progress()
print(f"Processing progress: {progress}%")

# Check embedding status
chunks = document.chunks
failed_chunks = [c for c in chunks if c.embedding_status == 'failed']
```

## Migration and Scaling

### Database Migration

When deploying RAG, ensure your database has pgvector extension:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for performance
CREATE INDEX ON vector_store USING ivfflat (embedding_vector vector_cosine_ops);
```

### S3 Setup

Organize documents with proper prefixes:
```
documents/
├── assistant_1/
│   ├── 20240101_120000_manual.pdf
│   └── 20240101_120100_policies.docx
└── assistant_2/
    └── 20240101_120200_faq.txt
```

## Best Practices

### Document Preparation
1. **Clean text**: Remove headers, footers, page numbers
2. **Consistent formatting**: Use standard headings and structure
3. **Logical sections**: Organize content hierarchically
4. **Update regularly**: Keep documents current

### RAG Configuration
1. **Tune similarity threshold**: Based on your domain
2. **Optimize chunk size**: Balance context vs. precision
3. **Monitor performance**: Track retrieval quality
4. **Test thoroughly**: Verify responses with different queries

### Security
1. **Access control**: Documents are isolated per assistant
2. **Data encryption**: S3 server-side encryption enabled
3. **API keys**: Secure storage of embedding service keys
4. **Audit logs**: Track document access and modifications

## Future Enhancements

- **Hybrid search**: Combine semantic and keyword search
- **Document summarization**: Automatic document summaries
- **Advanced filtering**: Search by metadata, date ranges
- **Real-time updates**: Live document synchronization
- **Multi-modal**: Support for images and tables
- **Custom embeddings**: Domain-specific embedding models

This RAG integration provides a solid foundation for knowledge-enhanced conversations while maintaining scalability and performance. 