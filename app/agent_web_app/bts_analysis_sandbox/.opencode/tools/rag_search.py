import sys, json, os

script_dir = os.path.dirname(os.path.abspath(__file__))
sandbox_root = os.path.dirname(os.path.dirname(script_dir))
os.chdir(sandbox_root)
sys.path.insert(0, sandbox_root)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    CHROMA_DIR = os.path.join(sandbox_root, "chroma_db")
    MODEL_PATH = os.path.join(sandbox_root, "model", "bge-small-zh-v1.5")

    def search(query):
        emb = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
        docs = vs.similarity_search(query, k=5)
        return [
            {"content": d.page_content, "page": d.metadata.get("page", "")}
            for d in docs
        ]

    query = sys.argv[1] if len(sys.argv) > 1 else ""
    if not query:
        print(json.dumps({"status": "error", "error": "query parameter is required"}, ensure_ascii=True))
    else:
        results = search(query)
        print(json.dumps({"status": "ok", "results": results}, ensure_ascii=True))

except Exception as e:
    import traceback
    print(json.dumps({
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc(),
    }, ensure_ascii=True))
