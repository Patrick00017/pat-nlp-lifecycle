import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils import load_config

from opencode_orchestrator import OpencodeOrchestrator

logger = logging.getLogger("opencode-router")

_script_dir = Path(__file__).parent
config = load_config(str(_script_dir / "config.yaml"))
oc_config = config.get("opencode", {})

_orchestrator: OpencodeOrchestrator | None = None


def get_orchestrator() -> OpencodeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OpencodeOrchestrator(
            host=oc_config.get("host", "127.0.0.1"),
            port=oc_config.get("port", 4096),
            password=oc_config.get("password", "my-opencode-password"),
            project_directory=oc_config.get("project_directory"),
        )
    return _orchestrator


router = APIRouter(prefix="/opencode")


class OpenCodeChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str
    agent: Optional[str] = "general"


class CreateSessionRequest(BaseModel):
    agent: Optional[str] = "general"


class CreateSessionResponse(BaseModel):
    session_id: str
    session: dict


@router.post("/chat/stream")
async def opencode_chat_stream(request: OpenCodeChatRequest):
    orch = get_orchestrator()
    try:
        await orch.ensure_server_running()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async def event_stream():
        try:
            if request.thread_id:
                session_id = request.thread_id
            else:
                session_id = await orch.create_session(agent=request.agent)

            yield f"data: {json.dumps({'type': 'thread_id', 'value': session_id})}\n\n"

            messages = await orch.prompt(session_id, request.message)
            for msg in messages:
                role = msg.get("role", "")
                parts = msg.get("parts", [])
                for part in parts if isinstance(parts, list) else []:
                    if isinstance(part, dict) and part.get("type") == "text":
                        yield f"data: {json.dumps({'type': 'message', 'content': part.get('text', '')})}\n\n"
                    elif isinstance(part, dict) and part.get("type") == "tool":
                        yield f"data: {json.dumps({'type': 'tool', 'name': part.get('name')})}\n\n"
                    elif isinstance(part, dict) and part.get("type") == "reasoning":
                        yield f"data: {json.dumps({'type': 'reason', 'content': part.get('text', '')})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("stream error")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/session", response_model=CreateSessionResponse)
async def opencode_create_session(request: CreateSessionRequest):
    orch = get_orchestrator()
    try:
        await orch.ensure_server_running()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        session_id = await orch.create_session(agent=request.agent)
        session_info = await orch.get_session_info(session_id)
        return CreateSessionResponse(session_id=session_id, session=session_info)
    except Exception as e:
        logger.exception("opencode create session failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def opencode_list_sessions(limit: int = 20):
    orch = get_orchestrator()
    try:
        await orch.ensure_server_running()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sessions = await orch.list_sessions(limit=limit)
    return {"sessions": sessions}


@router.get("/session/{session_id}")
async def opencode_get_session(session_id: str):
    orch = get_orchestrator()
    try:
        await orch.ensure_server_running()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    info = await orch.get_session_info(session_id)
    messages = await orch.get_messages(session_id)
    return {"session": info, "messages": messages}


@router.get("/agents")
async def opencode_list_agents():
    orch = get_orchestrator()
    await orch.ensure_server_running()
    agents = await orch.list_agents()
    return {"agents": agents}


@router.get("/providers")
async def opencode_list_providers():
    orch = get_orchestrator()
    await orch.ensure_server_running()
    providers = await orch.list_providers()
    return {"providers": providers}
