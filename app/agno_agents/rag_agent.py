from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.chroma import ChromaDb, SearchType
from agno.models.llama_cpp import LlamaCpp

# Create Knowledge Instance with ChromaDB
knowledge = Knowledge(
    name="Basic SDK Knowledge Base",
    description="Agno 2.0 Knowledge Implementation with ChromaDB",
    vector_db=ChromaDb(
        collection="vectors",
        path="tmp/chromadb",
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="llama3:8b"),
    ),
)

knowledge.insert(
    name="UserManual",
    path="D:/code/pat-nlp-lifecycle/app/usermanual.pdf",
    metadata={"doc_type": "user_manual"},
)


# Create and use the agent
rag_agent = Agent(
    model=LlamaCpp(
        id="qwen3-4b",
        base_url="http://localhost:8080/v1",  # Custom server URL
    ),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)

rag_agent.print_response("什么是强换操作")

# Delete operations examples
vector_db = knowledge.vector_db
vector_db.delete_by_name("UserManual")
