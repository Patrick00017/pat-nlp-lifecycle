import logging
from pathlib import Path

import fitz
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opencode_orchestrator import OpencodeOrchestrator

logger = logging.getLogger("rag-service")

PDF_PATH = Path(__file__).parent / "assets" / "usermanual.pdf"
MODEL_PATH = Path(__file__).parent / "model" / "bge-small-zh-v1.5"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


def _load_pdf_fitz(pdf_path: str) -> list[Document]:
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": page_num + 1}))
    return docs


class RAGService:
    def __init__(self, orchestrator: OpencodeOrchestrator):
        self.orchestrator = orchestrator
        self.embeddings = HuggingFaceEmbeddings(
            model_name=str(MODEL_PATH),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self):
        if CHROMA_DIR.exists():
            logger.info(f"Loading Chroma from {CHROMA_DIR}")
            return Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
            )
        logger.info(f"Loading PDF from {PDF_PATH}")
        docs = _load_pdf_fitz(str(PDF_PATH))
        logger.info(f"Loaded {len(docs)} pages, chunking...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks, indexing into Chroma...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    async def ask_stream(self, query: str, session_id: str | None = None):
        retrieved = self.vector_store.similarity_search(query, k=5)
        context = "\n\n".join(d.page_content for d in retrieved)
        prompt = (
            "你是一个 BTS 产线系统使用助手。请严格基于以下文档内容回答问题，"
            "如果文档中没有相关信息，请明确说明不知道，不要编造。"
            f"\n\n文档内容：\n{context}\n\n用户问题：{query}"
        )
        if session_id is None:
            session_id = await self.orchestrator.create_session(agent="doc-query-responder")
        async for event in self.orchestrator.stream_chat(session_id, prompt):
            yield event
