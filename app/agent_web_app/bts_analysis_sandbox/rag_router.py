import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from opencode_router import get_orchestrator
from rag_service import RAGService

logger = logging.getLogger("rag-router")
router = APIRouter(prefix="/rag")


class RagRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


@router.on_event("startup")
async def init_rag_service():
    try:
        router.rag_service = RAGService(get_orchestrator())
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.exception("RAG init failed")
        router.rag_service = None


@router.post("/stream")
async def rag_stream(request: RagRequest):
    if router.rag_service is None:
        async def err():
            yield f"data: {json.dumps({'type': 'error', 'error': 'RAG service not initialized'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    service: RAGService = router.rag_service

    async def event_stream():
        part_types: dict[str, str] = {}
        async for event in service.ask_stream(request.message, request.thread_id):
            payload = event.get("payload", {})
            evt_type = payload.get("type", "")
            props = payload.get("properties", {})

            if evt_type == "message.part.updated":
                part = props.get("part", {})
                pid = part.get("id")
                ptype = part.get("type", "")
                if pid and ptype:
                    part_types[pid] = ptype
            elif evt_type == "message.part.delta":
                pid = props.get("partID")
                delta = props.get("delta", "")
                if not delta:
                    continue
                ptype = part_types.get(pid, "text")
                if ptype == "reasoning":
                    yield f"data: {json.dumps({'type': 'reason', 'content': delta})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'message', 'content': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
