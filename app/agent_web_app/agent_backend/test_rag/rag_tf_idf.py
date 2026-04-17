import os
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def load_documents(directory):
    """
    Loads all text files from a specified directory into a list of documents.
    Each file's content is treated as a single document.
    """
    documents = []
    filepaths = []
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist. Please run the download_data.py script first.")
        return [], []

    print(f"Loading documents from '{directory}'...")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    filepaths.append(filename)
            except Exception as e:
                print(f"Could not read file {filename}: {e}")
    print(f"Successfully loaded {len(documents)} documents.")
    return documents, filepaths

def preprocess_text(text):
    """
    Cleans and tokenizes the text.
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

def retrieve_context(query, documents, vectorizer, doc_vectors):
    """
    Retrieves the most relevant document for a given query using TF-IDF and cosine similarity.
    """
    # Preprocess the query
    processed_query = preprocess_text(query)
    # Vectorize the query using the same vectorizer as the documents
    query_vector = vectorizer.transform([processed_query])
    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_vector, doc_vectors)
    # Get the index of the most similar document
    most_similar_doc_index = similarities.argmax()
    return documents[most_similar_doc_index]

def generate_response(query, context):
    """
    Generates a final response by combining the query and the retrieved context.
    This is a simple augmentation step.
    """
    # For this simple pipeline, we just provide the retrieved context and the original query.
    # A more advanced RAG would use a large language model to synthesize a final answer.
    response = f"Based on the provided documents, here is the most relevant information for your query:\n\n---\nContext:\n{context[:500]}...\n---\n\nUsing this context, the answer to your question might be found."
    return response

def main():
    """
    Main function to run the RAG pipeline.
    """
    # Load documents from the 'data' directory
    documents, filepaths = load_documents('data')
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Preprocess documents for vectorization
    processed_documents = [preprocess_text(doc) for doc in documents]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer to our documents and transform the documents into a matrix of TF-IDF features.
    doc_vectors = vectorizer.fit_transform(processed_documents)
    
    print("\nStarting RAG pipeline. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        # Retrieve relevant context
        retrieved_context = retrieve_context(query, documents, vectorizer, doc_vectors)

        # Generate and print the response
        response = generate_response(query, retrieved_context)
        print("\n" + "="*50)
        print("Generated Response:")
        print(response)
        print("="*50)

if __name__ == "__main__":
    main()