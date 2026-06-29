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

@app.get("/api/fsm-results")
def get_results():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {"status": "ok"}

from opencode_router import router as opencode_router
app.include_router(opencode_router)
