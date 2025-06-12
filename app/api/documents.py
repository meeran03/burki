from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status, Response
from sqlalchemy.orm import Session

from app.core.auth import get_current_user_flexible
from app.db.database import get_db
from app.db.models import Assistant, User

# RAG service is optional (development mode might not have the dependencies)
try:
    from app.services.rag_service import RAGService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

router = APIRouter(prefix="/api/v1/assistants/{assistant_id}/documents", tags=["documents"])


def _require_rag_available():
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Document functionality not available – RAG dependencies missing")


def _verify_assistant(db: Session, assistant_id: int, organization_id: int) -> Assistant:
    assistant = db.query(Assistant).filter(Assistant.id == assistant_id, Assistant.organization_id == organization_id).first()
    if not assistant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant not found in your organization")
    return assistant


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    assistant_id: int,
    file: UploadFile = File(...),
    name: Optional[str] = Query(None, description="Display name for the document"),
    category: Optional[str] = Query(None, description="Document category"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """Upload a document to an assistant knowledge-base and begin processing."""
    _require_rag_available()
    _verify_assistant(db, assistant_id, current_user.organization_id)

    # Read bytes
    file_bytes = await file.read()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    rag_service = RAGService.create_default_instance()
    document = await rag_service.upload_and_process_document(
        file_data=file_bytes,
        filename=file.filename,
        content_type=file.content_type,
        assistant_id=assistant_id,
        organization_id=current_user.organization_id,
        name=name,
        category=category,
        tags=tag_list,
    )

    return {
        "id": document.id,
        "name": document.name,
        "status": document.processing_status,
        "created_at": document.created_at.isoformat(),
    }


@router.get("/", response_model=List[dict])
async def list_documents(
    assistant_id: int,
    include_processing: bool = Query(True, description="Include documents still processing"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """Return documents belonging to an assistant."""
    _require_rag_available()
    _verify_assistant(db, assistant_id, current_user.organization_id)

    rag_service = RAGService.create_default_instance()
    documents = await rag_service.get_assistant_documents(assistant_id, include_processing=include_processing)

    return [
        {
            "id": d.id,
            "name": d.name,
            "filename": d.original_filename,
            "content_type": d.content_type,
            "file_size": d.file_size,
            "status": d.processing_status,
            "processed_chunks": d.processed_chunks,
            "total_chunks": d.total_chunks,
            "created_at": d.created_at.isoformat(),
            "processed_at": d.processed_at.isoformat() if d.processed_at else None,
        }
        for d in documents
    ]


@router.get("/{document_id}/status")
async def get_document_status(
    assistant_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    _require_rag_available()
    _verify_assistant(db, assistant_id, current_user.organization_id)

    rag_service = RAGService.create_default_instance()
    status_info = await rag_service.get_document_status(document_id)
    if not status_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return status_info


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    assistant_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    _require_rag_available()
    _verify_assistant(db, assistant_id, current_user.organization_id)

    rag_service = RAGService.create_default_instance()
    success = await rag_service.delete_document(document_id, assistant_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT) 