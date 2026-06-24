import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BTS FSM Results API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "environments" / "fsm_results.json"
_cache = None

def _load():
    global _cache
    if _cache is None:
        _cache = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return _cache

@app.get("/api/fsm-results")
def get_results():
    return _load()

@app.get("/api/health")
def health():
    return {"status": "ok"}
