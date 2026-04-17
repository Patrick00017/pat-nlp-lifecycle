import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.documents import Document
import pymupdf
import os
import bm25s
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

model = OllamaLLM(model="qwen3:8b")

embeddings = OllamaEmbeddings(model='llama3:8b')
# embeddings.
# vector_store = InMemoryVectorStore(embeddings)
vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embeddings,
    persist_directory="./manual_db"
)
retriever = bm25s.BM25() # bm25
corpus = []


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    # add the most similar doc using bm25s
    query_tokens = bm25s.tokenize("强换")
    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
    result_doc = Document(page_content=results[0][0])
    retrieved_docs.append(result_doc)
    # use both vec search doc and bm25s doc
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print(f"输入的文档信息: {docs_content}")
    system_message = (
        "你是一个文档查询助手，按照用户的问题完全按照文档内容进行回答:"
        f"\n\n{docs_content}"
    )
    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

def add_data_to_vector_db():
    # Only keep post title, headers, and content from the full HTML.
    pdf = pymupdf.open("usermanual.pdf")
    content = ''
    for page in pdf:
        # Perform OCR on the page
        text = page.get_text() # Specify the language (e.g., English)
        # text_page = page.get_textpage_ocr()
        # Extract text from the OCR result
        # text = text_page.extract_text()
        content += f"\n{text}"
        corpus.append(text)
    docs = [Document(page_content=content)]

    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")

    print(docs[0].page_content[:500])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    document_ids = vector_store.add_documents(documents=all_splits)

    print(document_ids[:3])

    # tokenize the bm25s
    corpus_tokens = bm25s.tokenize(corpus)
    retriever.index(corpus_tokens)

def init_corpus_data():
    pdf = pymupdf.open("usermanual.pdf")
    for page in pdf:
        text = page.get_text()
        corpus.append(text)
    # tokenize the bm25s
    corpus_tokens = bm25s.tokenize(corpus)
    retriever.index(corpus_tokens)

def main():
    # add_data_to_vector_db()
    init_corpus_data()

    query = "IPS的参数优化路径"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
