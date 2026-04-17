import bm25s
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_all_splits():
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

    # print(all_splits)
    all_raw_splits = [cont.page_content for cont in all_splits]
    return all_raw_splits

corpus = get_all_splits()
corpus_tokens = bm25s.tokenize(corpus)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

query = "强换"
query_tokens = bm25s.tokenize(query)

results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
print(len(results))
print(results)
print(scores)