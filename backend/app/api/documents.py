"""
Document API endpoints

Upload and manage research papers.
"""
import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.models import Document, KnowledgeBase, DocumentStatus
from app.schemas import DocumentResponse
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    pmid: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document for processing."""
    # Verify workspace exists
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # Save file
    upload_dir = settings.BASE_DIR / "data" / "papers" / f"workspace_{workspace_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create document record
    document = Document(
        workspace_id=workspace_id,
        original_filename=file.filename,
        file_type=file.content_type,
        file_size=len(content),
        file_path=str(file_path),
        status=DocumentStatus.PENDING,
        pmid=pmid if pmid else None,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Process document asynchronously
    try:
        rag_service = get_rag_service(db, workspace_id)
        await rag_service.process_document(
            document_id=document.id,
            file_path=str(file_path),
            pmid=pmid,
        )
        await db.refresh(document)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}",
        )

    return DocumentResponse.model_validate(document)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a workspace."""
    result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    return [DocumentResponse.model_validate(doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    workspace_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a document by ID."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    workspace_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Delete from vector store
    rag_service = get_rag_service(db, workspace_id)
    await rag_service.delete_document(document_id)

    # Delete file
    if document.file_path and Path(document.file_path).exists():
        os.remove(document.file_path)

    # Delete from database
    await db.delete(document)
    await db.commit()
