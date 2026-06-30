import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opencode_orchestrator import OpencodeOrchestrator

logger = logging.getLogger("rag-service")

PDF_PATH = Path(__file__).parent / "assets" / "usermanual.pdf"
MODEL_PATH = Path(__file__).parent / "model" / "bge-small-zh-v1.5"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


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
        loader = PyPDFLoader(str(PDF_PATH))
        docs = loader.load()
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
            session_id = await self.orchestrator.create_session(agent="general")
        async for event in self.orchestrator.stream_chat(session_id, prompt):
            yield event
