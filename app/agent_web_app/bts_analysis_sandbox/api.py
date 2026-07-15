import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rag_service import UPLOAD_DIR, build_chroma

app = FastAPI(title="BTS FSM Results API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "environments" / "fsm_results.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def clear_fsm_results():
    DATA_PATH.write_text("{}", encoding="utf-8")


@app.get("/api/fsm-results")
def get_results():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/kb/files")
def kb_files():
    files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return {"status": "ok", "files": files}


@app.post("/api/kb/upload")
async def kb_upload(files: list[UploadFile] = File(...)):
    saved = []
    for uf in files:
        safe_name = Path(uf.filename or "unnamed").name
        dest = UPLOAD_DIR / safe_name
        content = await uf.read()
        dest.write_bytes(content)
        saved.append({
            "name": safe_name,
            "size": len(content),
        })
    return {"status": "ok", "files": saved}


@app.post("/api/kb/build")
def kb_build():
    if not UPLOAD_DIR.exists():
        return {"status": "error", "detail": "上传目录不存在"}
    file_paths = [str(p) for p in UPLOAD_DIR.iterdir() if p.is_file()]
    if not file_paths:
        return {"status": "error", "detail": "没有可构建的文件，请先上传"}
    try:
        unused_vs, chunk_count = build_chroma(file_paths=file_paths, force_rebuild=True)
        return {"status": "ok", "chunk_count": chunk_count}
    except Exception as e:
        import traceback
        return {"status": "error", "detail": str(e), "traceback": traceback.format_exc()}


@app.delete("/api/kb/file/{name}")
def kb_delete_file(name: str):
    target = UPLOAD_DIR / name
    if target.exists():
        target.unlink()
        return {"status": "ok"}
    return {"status": "error", "detail": "文件不存在"}


from opencode_router import router as opencode_router
app.include_router(opencode_router)
