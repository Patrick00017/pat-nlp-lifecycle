"""
Hybrid Q&A Demo
- BM25 for traditional search (threshold 0.5)
- Sentence embeddings + LLM for fallback
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

# BM25 dependencies
from rank_bm25 import BM25Okapi

# Embedding dependencies (for fallback) - using BGE small Chinese model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# LLM dependencies
import requests


class HybridQASystem:
    def __init__(self, docs_folder: str, bm25_threshold: float = 0.5):
        self.docs_folder = docs_folder
        self.bm25_threshold = bm25_threshold

        self.chunks = []
        self.chunk_sources = []
        self.bm25 = None

        self.embedding_model = None
        self.chunk_embeddings = None

    def load_documents(self) -> List[Tuple[str, str]]:
        """Load all markdown files from docs folder"""
        docs = []
        folder = Path(self.docs_folder)

        for md_file in sorted(folder.glob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append((md_file.name, content))

        return docs

    def chunk_documents(
        self, docs: List[Tuple[str, str]], chunk_size: int = 500
    ) -> None:
        """Split documents into chunks"""
        for filename, content in docs:
            lines = content.split("\n")
            current_chunk = []
            current_size = 0

            for line in lines:
                line_size = len(line)
                if current_size + line_size > chunk_size and current_chunk:
                    self.chunks.append("\n".join(current_chunk))
                    self.chunk_sources.append(filename)
                    current_chunk = []
                    current_size = 0
                current_chunk.append(line)
                current_size += line_size

            if current_chunk:
                self.chunks.append("\n".join(current_chunk))
                self.chunk_sources.append(filename)

        print(f"Created {len(self.chunks)} chunks from {len(docs)} documents")

    def build_bm25_index(self) -> None:
        """Build BM25 index"""
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("BM25 index built")

    def build_embedding_index(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        """Build embedding index for fallback using BGE small Chinese model"""
        print(f"Loading {model_name}...")
        self.embedding_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.embedding_model.eval()

        print("Encoding chunks... (this may take a while)")
        self.chunk_embeddings = self._encode_texts(self.chunks)
        print(f"Embedding index built with {model_name}")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings using BGE model"""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        return np.array(embeddings)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using BGE model"""
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
            outputs = self.embedding_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].numpy()[0]

    def bm25_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """BM25 search returns (chunk_index, score)"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def embedding_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Embedding similarity search returns (chunk_index, score)"""
        query_embedding = self._encode_query(query)
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, float(similarities[idx])) for idx in top_indices]

    def normalize_bm25_score(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores to 0-1 range using min-max scaling"""
        if scores.max() == 0:
            return scores
        return (scores - scores.min()) / (scores.max() - scores.min())

    def call_llm(self, prompt: str, max_tokens: int = 100, endpoint: str = "http://127.0.0.1:8080/completion") -> str:
        """Call llama.cpp server"""
        try:
            response = requests.post(endpoint, json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stop": ["</s>", "\n\n"]
            }, timeout=30)
            response.raise_for_status()
            return response.json().get("content", "")
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer question using hybrid approach"""
        bm25_results = self.bm25_search(question, top_k=3)
        best_idx, raw_bm25_score = bm25_results[0]

        # Normalize BM25 score
        all_scores = self.bm25.get_scores(question.lower().split())
        normalized_bm25_score = self.normalize_bm25_score(all_scores)[best_idx]

        result = {
            "question": question,
            "method": (
                "bm25" if normalized_bm25_score >= self.bm25_threshold else "embedding"
            ),
            "bm25_score": round(normalized_bm25_score, 3),
            "source": self.chunk_sources[best_idx],
            "answer": self.chunks[best_idx][:1000],  # Limit output length
        }

        if normalized_bm25_score >= self.bm25_threshold:
            # Use BM25 result (traditional method - no LLM)
            result["method"] = "bm25"
            result["answer"] = self._extract_relevant_section(
                question, self.chunks[best_idx]
            )
        else:
            # Fallback to embedding + LLM
            embedding_results = self.embedding_search(question, top_k=3)
            emb_best_idx, emb_score = embedding_results[0]

            # Build context from top-k docs
            context_doc = self.chunks[emb_best_idx][:1500]  # Limit context

            # Build prompt for LLM
            prompt = f"""Based on the following context from the documentation, answer the user's question accurately and concisely.

Context:
{context_doc}

Question: {question}

Provide a clear, direct answer based on the context above. If the context doesn't contain enough information, state that clearly.

Answer:"""

            llm_answer = self.call_llm(prompt, max_tokens=100)

            result["method"] = "llm"
            result["embedding_score"] = round(emb_score, 3)
            result["source"] = self.chunk_sources[emb_best_idx]
            result["answer"] = llm_answer

        return result

    def _extract_relevant_section(self, query: str, content: str) -> str:
        """Extract the most relevant section from content"""
        query_lower = query.lower()
        lines = content.split("\n")

        best_section = []
        best_score = 0

        current_section = []
        for line in lines:
            if line.strip().startswith("#"):
                if current_section and best_score > 0:
                    best_section = current_section
                current_section = [line]
            else:
                current_section.append(line)
                # Simple keyword matching
                query_words = set(query_lower.split())
                line_words = set(line.lower().split())
                score = len(query_words & line_words)
                if score > best_score:
                    best_score = score
                    best_section = current_section[:]

        if not best_section:
            best_section = lines[:20]  # Return first 20 lines as fallback

        return "\n".join(best_section[:50])  # Limit to 50 lines


def main():
    # Initialize system
    docs_folder = r"D:\code\pat-nlp-lifecycle\docs"
    qa_system = HybridQASystem(docs_folder, bm25_threshold=0.5)

    # Build indexes
    print("Loading documents...")
    docs = qa_system.load_documents()

    print("Chunking documents...")
    qa_system.chunk_documents(docs)

    print("Building BM25 index...")
    qa_system.build_bm25_index()

    print("Building embedding index...")
    qa_system.build_embedding_index()

    print("\n" + "=" * 60)
    print("Hybrid Q&A System Ready!")
    print("=" * 60)

    # Test questions
    test_questions = [
        "如何使用强换功能?",
        "什么是IPS系统?",
        "如何添加新订单?",
        "服务器配置需要哪些?",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)

        result = qa_system.answer(question)

        print(f"方法: {result['method']}")
        print(f"BM25分数: {result.get('bm25_score', 'N/A')}")
        if result.get("embedding_score"):
            print(f"Embedding分数: {result['embedding_score']}")
        print(f"来源: {result['source']}")
        print(f"答案:\n{result['answer'][:500]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
