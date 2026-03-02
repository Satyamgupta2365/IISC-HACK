import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import snapshot_download

def build_vector_store():
    print("Loading PDF...")
    pdf_path = "data/ccpa_statute.pdf"
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found. Skipping vector store build.")
        return
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    print("Embedding text...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    
    print("Saving FAISS index...")
    db.save_local("faiss_index")
    print("FAISS index built successfully.")

def download_model():
    """Download LLM weights at build time so they are baked into the Docker image."""
    model_id = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    hf_token = os.environ.get("HF_TOKEN")
    
    print(f"Downloading model: {model_id}...")
    try:
        snapshot_download(
            repo_id=model_id,
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf"]
        )
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to download model: {e}")
        print("Model will need to be downloaded at runtime (slow first request).")

def download_embeddings():
    """Pre-download the sentence-transformers model so it's cached in the image."""
    print("Pre-downloading sentence-transformers/all-MiniLM-L6-v2...")
    try:
        snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model downloaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to download embedding model: {e}")

if __name__ == "__main__":
    download_embeddings()
    build_vector_store()
    download_model()
