import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATA_PATH = "data/"

def clear_data_directory():
    """Deletes all files in the 'data/' directory."""
    if os.path.exists(DATA_PATH):
        print(f"Clearing data directory: {DATA_PATH}")
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)

def load_documents():
    """
    Loads all PDF documents from the 'data/' folder dynamically.
    Returns a combined list of all loaded pages from every PDF.
    """
    documents = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
        print(f"Created data directory: {DATA_PATH}")
        return documents

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in 'data/' folder.")
        return documents

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            file_docs = loader.load()
            documents.extend(file_docs)
            print(f"Loaded {len(file_docs)} page(s) from {pdf_path}")
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")

    print(f"Total loaded documents: {len(documents)}")
    return documents


def split_documents(documents):
    """Splits documents into smaller text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits


def get_embedding_function(model_name="nomic-embed-text"):
    """Initializes the Ollama embedding function (ensure 'ollama serve' is running)."""
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings