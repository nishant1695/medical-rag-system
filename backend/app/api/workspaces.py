"""
Workspace API endpoints

Manage knowledge base workspaces (subspecialties).
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import KnowledgeBase, Document
from app.schemas import WorkspaceCreate, WorkspaceResponse

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace: WorkspaceCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new workspace."""
    # Create workspace
    db_workspace = KnowledgeBase(
        name=workspace.name,
        description=workspace.description,
        subspecialty=workspace.subspecialty,
        system_prompt=workspace.system_prompt,
    )
    db.add(db_workspace)
    await db.commit()
    await db.refresh(db_workspace)

    return WorkspaceResponse(
        id=db_workspace.id,
        name=db_workspace.name,
        description=db_workspace.description,
        subspecialty=db_workspace.subspecialty,
        created_at=db_workspace.created_at,
        document_count=0,
    )


@router.get("", response_model=List[WorkspaceResponse])
async def list_workspaces(
    db: AsyncSession = Depends(get_db),
):
    """List all workspaces."""
    result = await db.execute(select(KnowledgeBase))
    workspaces = result.scalars().all()

    # Count documents for each workspace
    workspace_responses = []
    for ws in workspaces:
        count_result = await db.execute(
            select(func.count(Document.id)).where(Document.workspace_id == ws.id)
        )
        doc_count = count_result.scalar() or 0

        workspace_responses.append(
            WorkspaceResponse(
                id=ws.id,
                name=ws.name,
                description=ws.description,
                subspecialty=ws.subspecialty,
                created_at=ws.created_at,
                document_count=doc_count,
            )
        )

    return workspace_responses


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a workspace by ID."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # Count documents
    count_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    doc_count = count_result.scalar() or 0

    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        description=workspace.description,
        subspecialty=workspace.subspecialty,
        created_at=workspace.created_at,
        document_count=doc_count,
    )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a workspace."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    await db.delete(workspace)
    await db.commit()
