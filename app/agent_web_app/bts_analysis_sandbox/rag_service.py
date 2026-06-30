import logging
from pathlib import Path

import fitz
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def build_chroma():
    embeddings = HuggingFaceEmbeddings(
        model_name=str(MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if CHROMA_DIR.exists():
        logger.info(f"Loading Chroma from {CHROMA_DIR}")
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    logger.info(f"Loading PDF from {PDF_PATH}")
    docs = _load_pdf_fitz(str(PDF_PATH))
    logger.info(f"Loaded {len(docs)} pages, chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks, indexing into Chroma...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
