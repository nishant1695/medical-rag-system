"""
Chat and search API endpoints

Query the knowledge base with safety checks and provide an SSE streaming
chat endpoint that uses the medical agent.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import KnowledgeBase, ChatMessage as ChatMessageModel
from app.schemas import SearchRequest, SearchResponse, SourceChunk, ChatRequest, HistoryMessage
from app.services.rag_service import get_rag_service
from app.services.medical_safety_classifier import safety_classifier
from app.services.chat_agent import chat_stream_endpoint

router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["chat"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    workspace_id: int,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents in the knowledge base.

    Returns relevant chunks with medical metadata and citations.
    """
    # Verify workspace
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # Safety check
    safety_class = safety_classifier.classify(request.query)
    if safety_classifier.should_block_query(safety_class):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "safety_classification": safety_class,
                "message": safety_classifier.get_warning_message(safety_class),
            },
        )

    # Retrieve documents
    rag_service = get_rag_service(db, workspace_id)
    retrieval_result = await rag_service.query(
        question=request.query,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    # Convert to response format
    source_chunks = []
    for i, chunk in enumerate(retrieval_result.chunks):
        source_chunks.append(
            SourceChunk(
                index=f"{i+1}",  # Simple numeric index for search
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                document_id=chunk.document_id,
                page_no=chunk.page_no,
                heading_path=chunk.heading_path,
                score=chunk.score,
                pmid=chunk.pmid,
                evidence_level=chunk.evidence_level,
            )
        )

    return SearchResponse(
        query=request.query,
        chunks=source_chunks,
        evidence_summary=retrieval_result.evidence_summary,
    )


@router.post("/chat")
async def chat_stream(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming chat endpoint. Returns a StreamingResponse with SSE events.

    Example curl (raw SSE stream):
    curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \
      -X POST "http://localhost:8000/api/v1/workspaces/1/chat" \
      -d '{"message":"What are the outcomes of DIEP flap surgery?","history":[],"enable_thinking":false,"force_search":true}'
    """
    # Delegate to chat agent's stream endpoint which returns a StreamingResponse
    return await chat_stream_endpoint(workspace_id=workspace_id, request=request, db=db)


@router.get("/history", response_model=list[HistoryMessage])
async def get_chat_history(
    workspace_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """
    Return the most recent chat messages for a workspace (newest-last order).
    Used by the frontend to restore conversation on page load.
    """
    result = await db.execute(
        select(ChatMessageModel)
        .where(ChatMessageModel.workspace_id == workspace_id)
        .order_by(ChatMessageModel.created_at.asc())
        .limit(limit)
    )
    return result.scalars().all()


@router.delete("/history", status_code=204)
async def clear_chat_history(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete all chat messages for a workspace."""
    from sqlalchemy import delete
    await db.execute(
        delete(ChatMessageModel).where(ChatMessageModel.workspace_id == workspace_id)
    )
    await db.commit()


@router.get("/stats")
async def get_workspace_stats(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get workspace statistics."""
    # Verify workspace
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # Get chunk count from vector store
    rag_service = get_rag_service(db, workspace_id)
    chunk_count = rag_service.get_chunk_count()

    return {
        "workspace_id": workspace_id,
        "chunk_count": chunk_count,
        "name": workspace.name,
        "subspecialty": workspace.subspecialty,
    }


@router.get("/graph")
async def get_knowledge_graph_data(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Return the knowledge graph for a workspace as {nodes, edges}.

    Nodes are medical entities extracted from the literature (procedures,
    conditions, outcomes, anatomy, techniques, populations, drugs).
    Edges are typed relationships between them (treats, complicates,
    compared_to, requires, associated_with, contraindicates, part_of).

    Useful for visualising what concepts the system has learned and how
    they are connected across the indexed papers.
    """
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    from app.services.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph(workspace_id)
    return await kg.get_graph_data(db)

