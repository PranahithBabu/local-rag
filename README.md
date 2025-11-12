# 🧠 Local RAG-Based PDF Chat Application

This is a Retrieval-Augmented Generation (RAG) application that runs completely locally.
It uses ChromaDB as the vector database and Ollama’s Qwen3 (8B) model for language generation.

Users can upload PDFs and interact with them through natural language chat.
The system extracts text from uploaded documents, generates embeddings, stores them in the local vector database, and retrieves the most relevant text chunks to answer user queries.

## 🔭 Usage

- Avoid reading long FAQ's.
- Get summaries of long books.
- Understand terms and conditions of businesses with simple questions.

## 🧩 Tech Stack
Layer | Technology
--- | ---
Backend | Python
Vector DB |	ChromaDB
Embeddings | SentenceTransformers
LLM | Ollama QWEN3

## 🛠️ Local Setup Instructions

Follow these steps to run the app on your local machine:

1. Clone the repository

    ```
    git clone https://github.com/PranahithBabu/local-rag.git
    cd local-rag
    ```

2. Create and activate a virtual environment

    ```
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install dependencies

    ```
    pip install -r requirements.txt
    ```

    <i>Remove content inside <b>chromadb and data</b> folder to avoid default PDF's.  </i>

4. Download Ollama from https://ollama.com/download

5. Run below commands to get the model for this project;
    ```
    ollama pull qwen3:8b
    ollama run qwen3:8b
    ```

6. Enter `ollama serve` in a new terminal to run the model.

    <i>Note: If port 11434 issue occurs, then it means, the ollama is already running in the background on the same port, and you can avoid this command to start the server.</i>

7. Run the application ```python app.py```

8. Follow the terminal for step-by-step logs and response to given query.
