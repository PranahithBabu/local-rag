import os
import shutil
from langchain_community.vectorstores import Chroma

# Directory to store ChromaDB data
CHROMA_PATH = "chroma_db"
DB_FILE = os.path.join(CHROMA_PATH, "chroma.sqlite3")

def db_exists():
    """Checks if the ChromaDB file already exists."""
    return os.path.exists(DB_FILE)

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """
    Clears the old vector store and indexes new document chunks.
    """
    # 1. Clear the existing directory
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at: {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # 2. Create a new vector store from documents
    print(f"Indexing {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    # 3. Persist the new data
    vectorstore.persist()
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore
