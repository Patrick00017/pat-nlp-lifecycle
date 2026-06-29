import os
import json
import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger("opencode-orchestrator")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096
DEFAULT_PASSWORD = "my-opencode-password"


def _extract_text(messages: list[dict]) -> str:
    text = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            parts = msg.get("parts", [])
            for part in parts if isinstance(parts, list) else []:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
    return text


def _extract_tool_calls(messages: list[dict]) -> list[dict]:
    calls = []
    for msg in messages:
        parts = msg.get("parts", [])
        for part in parts if isinstance(parts, list) else []:
            if isinstance(part, dict) and part.get("type") == "tool":
                calls.append(
                    {
                        "name": part.get("name"),
                        "input": part.get("input"),
                        "result": part.get("result"),
                    }
                )
    return calls


class OpencodeOrchestrator:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: str = DEFAULT_PASSWORD,
        project_directory: str | None = None,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.project_directory = project_directory or os.getcwd()
        self.base_url = f"http://{host}:{port}"
        self._auth = ("opencode", password)
        self._client: httpx.AsyncClient | None = None
        self._auto_approve_task: asyncio.Task | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, auth=self._auth, timeout=300
            )
        return self._client

    async def ensure_server_running(self, wait_seconds: int = 10) -> bool:
        for i in range(wait_seconds * 2):
            if await self._health_check():
                logger.info("Connected to opencode server at %s", self.base_url)
                self._start_auto_approve()
                return True
            await asyncio.sleep(0.5)

        raise RuntimeError(
            f"opencode server not reachable at {self.base_url}. "
            f"Make sure to run: opencode serve --port {self.port} --hostname {self.host}"
        )

    async def _health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, auth=self._auth, timeout=5
            ) as c:
                resp = await c.get("/global/health")
                return resp.status_code == 200
        except Exception:
            return False

    def _start_auto_approve(self):
        if self._auto_approve_task is None or self._auto_approve_task.done():
            self._auto_approve_task = asyncio.create_task(self._auto_approve_loop())

    async def _auto_approve_loop(self):
        try:
            while True:
                try:
                    resp = await self.client.get("/permission")
                    if resp.status_code == 200:
                        body = resp.json()
                        perms = body if isinstance(body, list) else body.get("data", [])
                        for req in perms:
                            request_id = req.get("id")
                            logger.info("Auto-approving permission %s", request_id)
                            await self.client.post(
                                f"/permission/{request_id}/reply",
                                json={"response": "allow_once"},
                            )
                except Exception as e:
                    logger.debug("Auto-approve check: %s", e)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def create_session(
        self, agent: str = "general", directory: str | None = None
    ) -> str:
        dir_path = directory or self.project_directory
        logger.info("Creating session (agent=%s, dir=%s)", agent, dir_path)
        payload: dict[str, Any] = {}
        if agent:
            payload["agent"] = agent

        resp = await self.client.post("/session", json=payload)
        resp.raise_for_status()
        session = resp.json()
        session_id = session.get("id") or session.get("data", {}).get("id")
        if not session_id:
            raise RuntimeError(f"Unexpected create session response: {session}")
        logger.info("Session created: %s", session_id)
        return session_id

    async def prompt(self, session_id: str, text: str) -> list[dict]:
        logger.info("Submitting prompt to session %s", session_id)
        payload = {
            "parts": [
                {"type": "text", "text": text},
            ],
        }

        resp = await self.client.post(f"/session/{session_id}/message", json=payload)
        resp.raise_for_status()

        logger.info("Prompt submitted, fetching all messages...")
        return await self.get_messages(session_id, order="asc")

    async def stream_chat(self, session_id: str, text: str):
        """Async generator yielding raw event dicts from /global/event SSE during message processing."""
        event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _read_events():
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url, auth=self._auth, timeout=None
                ) as c:
                    async with c.stream("GET", "/global/event") as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                await event_queue.put(json.loads(line[6:]))
            except Exception as e:
                logger.exception("event reader error")
            finally:
                await event_queue.put(None)

        async def _send_message():
            try:
                payload = {"model": {
                    "modelID": "deepseek-v4-flash",
                    "providerID": "opencode-go",
                    "api": {
                        "id": "deepseek-v4-flash",
                        "url": "https://opencode.ai/zen/go/v1",
                        "npm": "@ai-sdk/openai-compatible"
                    },
                    "name": "DeepSeek V4 Flash",
                    "family": "deepseek-flash",
                    "capabilities": {
                        "temperature": True,
                        "reasoning": True,
                        "attachment": False,
                        "toolcall": True,
                        "input": {
                            "text": True,
                            "audio": False,
                            "image": False,
                            "video": False,
                            "pdf": False
                        },
                        "output": {
                            "text": True,
                            "audio": False,
                            "image": False,
                            "video": False,
                            "pdf": False
                        },
                        "interleaved": {
                            "field": "reasoning_content"
                        }
                    },
                    "cost": {
                        "input": 0.14,
                        "output": 0.28,
                        "cache": {
                            "read": 0.0028,
                            "write": 0
                        }
                    },
                    "limit": {
                        "context": 1000000,
                        "output": 384000
                    },
                    "status": "active",
                    "options": {},
                    "headers": {},
                    "release_date": "2026-04-24",
                    "variants": {
                        "low": {
                            "reasoningEffort": "low"
                        },
                        "medium": {
                            "reasoningEffort": "medium"
                        },
                        "high": {
                            "reasoningEffort": "high"
                        },
                        "max": {
                            "reasoningEffort": "max"
                        }
                    }
                }, "parts": [{"type": "text", "text": text}]}
                await self.client.post(
                    f"/session/{session_id}/message", json=payload
                )
            except Exception as e:
                logger.exception("send message error")

        reader = asyncio.create_task(_read_events())
        sender = asyncio.create_task(_send_message())

        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=120)
            except asyncio.TimeoutError:
                logger.warning("stream_chat timed out waiting for events")
                break
            if event is None:
                break

            payload = event.get("payload", {})
            props = payload.get("properties", {})
            evt_session = (
                props.get("sessionID")
                or props.get("session_id")
                or event.get("sessionID")
            )
            if evt_session and evt_session != session_id:
                continue

            yield event

            if payload.get("type") in ("session.idle",):
                break
            if (
                payload.get("type") == "session.status"
                and isinstance(props.get("status"), dict)
                and props.get("status", {}).get("type") == "idle"
            ):
                break
            if (
                payload.get("type") == "message.part.updated"
                and isinstance(props.get("part"), dict)
                and props.get("part", {}).get("type") == "step-finish"
                and props.get("part", {}).get("reason") == "stop"
            ):
                break

        reader.cancel()
        try:
            await reader
        except (asyncio.CancelledError, Exception):
            pass
        if not sender.done():
            await sender

    async def get_messages(self, session_id: str, order="asc", limit=50) -> list[dict]:
        resp = await self.client.get(
            f"/session/{session_id}/message",
            params={"order": order, "limit": limit},
        )
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            return body.get("data", body.get("messages", []))
        return []

    async def list_sessions(self, limit=20) -> list[dict]:
        resp = await self.client.get("/session", params={"limit": limit})
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return body
        return body.get("data", [])

    async def get_session_info(self, session_id: str) -> dict:
        resp = await self.client.get(f"/session/{session_id}")
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

    async def list_agents(self) -> list[dict]:
        resp = await self.client.get("/agent")
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return body
        return body.get("data", [])

    async def list_providers(self) -> list[dict]:
        resp = await self.client.get("/provider")
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return body
        return body.get("all", [])

    async def read_file(self, path: str, directory: str | None = None) -> bytes:
        params: dict[str, str] = {}
        if directory:
            params["directory"] = directory
        resp = await self.client.get(f"/file/content", params={"path": path} | params)
        resp.raise_for_status()
        return resp.content

    async def list_files(
        self, path: str = "", directory: str | None = None
    ) -> list[dict]:
        params: dict[str, str] = {"path": path}
        if directory:
            params["directory"] = directory
        resp = await self.client.get("/file", params=params)
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return body
        return body.get("data", [])

    async def shutdown(self):
        if self._auto_approve_task:
            self._auto_approve_task.cancel()
            try:
                await self._auto_approve_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
