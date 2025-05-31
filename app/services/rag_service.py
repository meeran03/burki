"""
RAG (Retrieval Augmented Generation) service for handling document processing,
embedding, and retrieval using LangChain and pgvector.
"""

import os
import logging
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
from datetime import datetime

# LangChain imports
try:
    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredFileLoader,
    )
    from langchain_openai import OpenAIEmbeddings
    from langchain_postgres import PGVector
    from langchain_postgres.vectorstores import PGVector
    import tempfile
    import unstructured
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain dependencies not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Database imports
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select, update

# pgvector imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Local imports
from app.db.models import Document, DocumentChunk, Assistant
from app.db.database import get_async_db_session
from app.services.s3_service import S3Service
from app.rag.postgres_searcher import PostgresSearcher

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for handling RAG functionality including document processing,
    embedding, and retrieval.
    """

    def __init__(
        self,
        s3_service: Optional[S3Service] = None,
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        """
        Initialize RAG service.

        Args:
            s3_service: S3 service instance
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key for embeddings
            database_url: PostgreSQL database URL for pgvector
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain dependencies not installed")

        self.s3_service = s3_service or S3Service.create_default_instance()
        self.embedding_model = embedding_model
        self.database_url = database_url or os.getenv("DATABASE_URL")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=embedding_model,
        )

        # Supported document types
        self.supported_types = {
            "application/pdf": "pdf",
            "text/plain": "txt",
            "text/markdown": "md",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/msword": "doc",
        }

        logger.info(f"RAG service initialized with embedding model: {embedding_model}")

    async def upload_and_process_document(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        assistant_id: int,
        organization_id: int,
        name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive",
    ) -> Document:
        """
        Upload document to S3 and process for RAG.

        Args:
            file_data: Document file data
            filename: Original filename
            content_type: MIME type
            assistant_id: Assistant ID
            organization_id: Organization ID
            name: Display name for the document
            category: Document category
            tags: List of tags
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking (recursive, character, semantic)

        Returns:
            Document: Created document record
        """
        try:
            # Validate content type
            if content_type not in self.supported_types:
                raise ValueError(f"Unsupported content type: {content_type}")

            # Calculate file hash for deduplication
            file_hash = hashlib.sha256(file_data).hexdigest()

            async with await get_async_db_session() as db:
                # Check if document already exists
                existing_doc = await db.execute(
                    select(Document).filter(
                        and_(
                            Document.assistant_id == assistant_id,
                            Document.file_hash == file_hash,
                            Document.is_active == True
                        )
                    )
                )
                existing_doc = existing_doc.scalar_one_or_none()

                if existing_doc:
                    logger.info(f"Document with hash {file_hash} already exists")
                    return existing_doc

                # Upload to S3
                s3_key, s3_url = await self.s3_service.upload_document(
                    file_data, assistant_id, filename, content_type
                )

                # Create document record
                document = Document(
                    assistant_id=assistant_id,
                    organization_id=organization_id,
                    name=name or filename,
                    original_filename=filename,
                    content_type=content_type,
                    file_size=len(file_data),
                    file_hash=file_hash,
                    s3_key=s3_key,
                    s3_url=s3_url,
                    s3_bucket=self.s3_service.bucket_name,
                    document_type=self.supported_types[content_type],
                    processing_status="pending",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy,
                    category=category,
                    tags=tags or [],
                )

                db.add(document)
                await db.commit()
                await db.refresh(document)

                # Process document asynchronously
                asyncio.create_task(self._process_document_async(document.id, file_data))

                logger.info(f"Document uploaded and queued for processing: {document.id}")
                return document

        except Exception as e:
            logger.error(f"Error uploading and processing document: {e}")
            raise

    async def _process_document_async(self, document_id: int, file_data: bytes) -> None:
        """
        Process document asynchronously - extract text, chunk, and embed.

        Args:
            document_id: Document ID to process
            file_data: Raw file data
        """
        try:
            async with await get_async_db_session() as db:
                # Get document from database
                result = await db.execute(
                    select(Document).filter(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                if not document:
                    logger.error(f"Document {document_id} not found")
                    return

                # Store assistant_id to avoid accessing detached instance later
                assistant_id = document.assistant_id
                content_type = document.content_type
                chunking_strategy = document.chunking_strategy
                chunk_size = document.chunk_size
                chunk_overlap = document.chunk_overlap

                # Update status to processing
                await db.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(processing_status="processing")
                )
                await db.commit()

                # Extract text from document
                text_content = await self._extract_text(file_data, content_type)
                
                if not text_content:
                    await db.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(processing_status="failed", processing_error="Failed to extract text from document")
                    )
                    await db.commit()
                    return

                # Create text splitter based on strategy
                text_splitter = self._create_text_splitter(
                    chunking_strategy,
                    chunk_size,
                    chunk_overlap
                )

                # Split text into chunks
                chunks = text_splitter.split_text(text_content)
                
                if not chunks:
                    await db.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(processing_status="failed", processing_error="No chunks generated from document")
                    )
                    await db.commit()
                    return

                # Update document with total chunks
                await db.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(total_chunks=len(chunks))
                )
                await db.commit()

                # Process each chunk
                processed_chunks = 0
                for i, chunk_content in enumerate(chunks):
                    try:
                        await self._process_chunk(
                            document_id=document_id,
                            chunk_index=i,
                            content=chunk_content,
                            assistant_id=assistant_id,  # Use stored value instead of document.assistant_id
                        )
                        processed_chunks += 1
                        
                        # Update progress using direct query to avoid session issues
                        await db.execute(
                            update(Document)
                            .where(Document.id == document_id)
                            .values(processed_chunks=processed_chunks)
                        )
                        await db.commit()

                    except Exception as e:
                        logger.error(f"Error processing chunk {i} for document {document_id}: {e}")
                        continue

                # Mark as completed
                await db.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(processing_status="completed", processed_at=datetime.utcnow())
                )
                await db.commit()

                logger.info(f"Document {document_id} processing completed: {processed_chunks}/{len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error in async document processing {document_id}: {e}")
            # Update document status to failed using direct query
            try:
                async with await get_async_db_session() as db:
                    await db.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(processing_status="failed", processing_error=str(e))
                    )
                    await db.commit()
            except Exception as db_error:
                logger.error(f"Error updating document status: {db_error}")

    async def _extract_text(self, file_data: bytes, content_type: str) -> str:
        """
        Extract text from file data based on content type.

        Args:
            file_data: Raw file data
            content_type: MIME type

        Returns:
            str: Extracted text content
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file_path = temp_file.name

            try:
                if content_type == "application/pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif content_type == "text/plain":
                    loader = TextLoader(temp_file_path, encoding="utf-8")
                else:
                    # Use unstructured for other formats
                    loader = UnstructuredFileLoader(temp_file_path)

                documents = loader.load()
                text_content = "\n\n".join([doc.page_content for doc in documents])
                
                return text_content

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def _create_text_splitter(
        self, strategy: str, chunk_size: int, chunk_overlap: int
    ) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter]:
        """
        Create text splitter based on strategy.

        Args:
            strategy: Chunking strategy
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Text splitter instance
        """
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
        elif strategy == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n",
            )
        elif strategy == "semantic":
            # Note: Semantic chunker requires embeddings
            try:
                return SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                )
            except Exception as e:
                logger.warning(f"Semantic chunker failed, falling back to recursive: {e}")
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
        else:
            # Default to recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    async def _process_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        assistant_id: int,
    ) -> None:
        """
        Process a single chunk - create embedding and store in database.

        Args:
            document_id: Document ID
            chunk_index: Index of chunk within document
            content: Chunk content
            assistant_id: Assistant ID
        """
        try:
            async with await get_async_db_session() as db:
                # Check if chunk already exists
                result = await db.execute(
                    select(DocumentChunk).filter(
                        and_(
                            DocumentChunk.document_id == document_id,
                            DocumentChunk.chunk_index == chunk_index
                        )
                    )
                )
                existing_chunk = result.scalar_one_or_none()

                if existing_chunk:
                    logger.debug(f"Chunk {chunk_index} for document {document_id} already exists")
                    return

                # Generate embedding
                try:
                    # Run embedding in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    embedding = await loop.run_in_executor(
                        None, self.embeddings.embed_query, content
                    )

                    # Create chunk record with embedding
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=content,
                        embedding=embedding,
                    )

                    db.add(chunk)
                    await db.commit()
                    await db.refresh(chunk)

                    logger.debug(f"Chunk {chunk_index} for document {document_id} processed successfully")

                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk_index}: {e}")
                    # Create chunk without embedding if embedding fails
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=content,
                        embedding=None,
                    )

                    db.add(chunk)
                    await db.commit()
                    raise

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} for document {document_id}: {e}")
            raise

    async def search_documents(
        self,
        query: str,
        assistant_id: int,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search documents using semantic similarity.

        Args:
            query: Search query
            assistant_id: Assistant ID to filter documents
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            include_metadata: Whether to include document metadata

        Returns:
            List[Dict[str, Any]]: List of relevant chunks with metadata
        """
        try:
            # Initialize PostgresSearcher for DocumentChunk with join to Document table
            join_tables = {
                "document": {
                    "table": "documents",
                    "on": "documentchunks.document_id = document.id"
                }
            }
            
            searcher = PostgresSearcher(
                db_model=DocumentChunk,
                embed_dimensions=1536,  # OpenAI text-embedding-3-small dimensions
                join_tables=join_tables
            )

            # Define filters to only search chunks for this assistant with completed processing
            filters = [
                {
                    "column": "document.assistant_id",
                    "comparison_operator": "=", 
                    "value": assistant_id,
                    "type": "integer"
                },
                {
                    "column": "document.processing_status",
                    "comparison_operator": "=",
                    "value": "completed",
                    "type": "string"
                },
                {
                    "column": "document.is_active",
                    "comparison_operator": "=",
                    "value": True,
                    "type": "boolean"
                }
            ]

            # Use hybrid search (vector + full-text)
            chunks = searcher.search_and_embed(
                query_text=query,
                top=limit,
                enable_vector_search=True,
                enable_text_search=True,
                filters=filters
            )

            results = []
            for chunk in chunks:
                result = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                }

                if include_metadata:
                    # Get document metadata using async session
                    async with await get_async_db_session() as db:
                        doc_result = await db.execute(
                            select(Document).filter(Document.id == chunk.document_id)
                        )
                        document = doc_result.scalar_one_or_none()
                        
                        if document:
                            result["document"] = {
                                "id": document.id,
                                "name": document.name,
                                "original_filename": document.original_filename,
                                "category": document.category,
                                "tags": document.tags,
                                "created_at": document.created_at.isoformat(),
                            }

                results.append(result)

            logger.info(f"Found {len(results)} relevant chunks for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            # Fallback to basic text search if vector search fails
            try:
                async with await get_async_db_session() as db:
                    result = await db.execute(
                        select(DocumentChunk)
                        .join(Document)
                        .filter(
                            and_(
                                Document.assistant_id == assistant_id,
                                Document.processing_status == "completed",
                                Document.is_active == True,
                                DocumentChunk.content.ilike(f"%{query}%")
                            )
                        )
                        .limit(limit)
                    )
                    chunks = result.scalars().all()

                    results = []
                    for chunk in chunks:
                        result = {
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "similarity_score": 0.5,  # Lower score for text-only match
                        }
                        
                        if include_metadata:
                            doc_result = await db.execute(
                                select(Document).filter(Document.id == chunk.document_id)
                            )
                            document = doc_result.scalar_one_or_none()
                            
                            if document:
                                result["document"] = {
                                    "id": document.id,
                                    "name": document.name,
                                    "original_filename": document.original_filename,
                                    "category": document.category,
                                    "tags": document.tags,
                                    "created_at": document.created_at.isoformat(),
                                }
                        
                        results.append(result)
                    
                    logger.warning(f"Fallback to text search returned {len(results)} results")
                    return results
                    
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []

    async def get_assistant_documents(
        self,
        assistant_id: int,
        include_processing: bool = False,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Get all documents for an assistant.

        Args:
            assistant_id: Assistant ID
            include_processing: Whether to include documents being processed
            limit: Maximum number of documents to return

        Returns:
            List[Document]: List of documents
        """
        try:
            async with await get_async_db_session() as db:
                query = select(Document).filter(
                    and_(
                        Document.assistant_id == assistant_id,
                        Document.is_active == True
                    )
                )

                if not include_processing:
                    query = query.filter(Document.processing_status == "completed")

                if limit:
                    query = query.limit(limit)

                result = await db.execute(query.order_by(Document.created_at.desc()))
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Error getting assistant documents: {e}")
            return []

    async def delete_document(self, document_id: int, assistant_id: int) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document ID
            assistant_id: Assistant ID (for authorization)

        Returns:
            bool: True if successful
        """
        try:
            async with await get_async_db_session() as db:
                # Get document
                result = await db.execute(
                    select(Document).filter(
                        and_(
                            Document.id == document_id,
                            Document.assistant_id == assistant_id
                        )
                    )
                )
                document = result.scalar_one_or_none()

                if not document:
                    logger.warning(f"Document {document_id} not found for assistant {assistant_id}")
                    return False

                # Delete from S3
                if document.s3_key:
                    await self.s3_service.delete_document(document.s3_key)

                # Soft delete in database
                await db.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(is_active=False)
                )
                await db.commit()

                logger.info(f"Document {document_id} deleted successfully")
                return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def get_document_status(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get processing status of a document.

        Args:
            document_id: Document ID

        Returns:
            Optional[Dict[str, Any]]: Document status information
        """
        try:
            async with await get_async_db_session() as db:
                result = await db.execute(
                    select(Document).filter(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                if not document:
                    return None

                return {
                    "id": document.id,
                    "name": document.name,
                    "processing_status": document.processing_status,
                    "processing_error": document.processing_error,
                    "total_chunks": document.total_chunks,
                    "processed_chunks": document.processed_chunks,
                    "progress_percentage": document.get_processing_progress(),
                    "is_complete": document.is_processing_complete(),
                    "created_at": document.created_at.isoformat(),
                    "processed_at": document.processed_at.isoformat() if document.processed_at else None,
                }

        except Exception as e:
            logger.error(f"Error getting document status: {e}")
            return None

    @classmethod
    def create_default_instance(cls) -> "RAGService":
        """
        Create RAG service instance with default configuration.

        Returns:
            RAGService: Configured RAG service instance
        """
        return cls() 