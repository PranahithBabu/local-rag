from utils.pdf_loader_terminal import load_documents, split_documents, get_embedding_function
from utils.vector_store import get_vector_store, index_documents
from utils.llm_chain import create_rag_chain, query_rag

if __name__ == "__main__":
    # Load Documents
    docs = load_documents()

    # Split Documents
    chunks = split_documents(docs)

    # Get Embedding Function
    embedding_function = get_embedding_function()

    # Index Documents
    print("Attempting to index documents...")
    vector_store = index_documents(chunks, embedding_function)
    # To load existing DB
    # vector_store = get_vector_store(embedding_function)

    # RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")

    # Query
    query_question = "What is the main topic of the document?"
    query_rag(rag_chain, query_question)

    query_question_2 = "What is aven?"
    query_rag(rag_chain, query_question_2)
