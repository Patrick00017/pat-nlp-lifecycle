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
    agent: Optional[str] = "timeline-analyst"


class CreateSessionRequest(BaseModel):
    agent: Optional[str] = "timeline-analyst"


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
            session_id = (
                request.thread_id
                or await orch.create_session(agent=request.agent)
            )

            yield f"data: {json.dumps({'type': 'thread_id', 'value': session_id})}\n\n"

            part_types: dict[str, str] = {}

            async for event in orch.stream_chat(session_id, request.message):
                payload = event.get("payload", {})
                evt_type = payload.get("type", "")
                props = payload.get("properties", {})

                if evt_type == "message.part.updated":
                    part = props.get("part", {})
                    pid = part.get("id")
                    part_type = part.get("type", "")
                    if pid and part_type:
                        part_types[pid] = part_type

                elif evt_type == "message.part.delta":
                    pid = props.get("partID")
                    delta = props.get("delta", "")
                    if not delta:
                        continue
                    part_type = part_types.get(pid, "text")
                    if part_type == "reasoning":
                        yield f"data: {json.dumps({'type': 'reason', 'content': delta})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'message', 'content': delta})}\n\n"

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


@router.get("/diagnose")
async def opencode_diagnose():
    import time
    import httpx
    orch = get_orchestrator()
    result = {
        "config": {
            "host": orch.host,
            "port": orch.port,
            "base_url": orch.base_url,
        }
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(base_url=orch.base_url, auth=orch._auth, timeout=5) as c:
            resp = await c.get("/global/health")
            result["health_check"] = {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "elapsed_ms": round((time.time() - t0) * 1000),
            }
    except Exception as e:
        result["health_check"] = {
            "ok": False,
            "error": repr(e),
            "elapsed_ms": round((time.time() - t0) * 1000),
        }

    if result["health_check"].get("ok"):
        try:
            t0 = time.time()
            sid = await orch.create_session(agent="timeline-analyst")
            result["create_session"] = {
                "ok": True,
                "session_id": sid,
                "elapsed_ms": round((time.time() - t0) * 1000),
            }
            info = await orch.get_session_info(sid)
            result["session_info"] = info
        except Exception as e:
            result["create_session"] = {"ok": False, "error": repr(e)}

    return result


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
