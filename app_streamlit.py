import os
import streamlit as st
from utils.pdf_loader_ui import load_documents, split_documents, get_embedding_function, clear_data_directory
from utils.vector_store import get_vector_store, index_documents, db_exists
from utils.llm_chain import create_rag_chain, query_rag
import base64

CHROMA_PATH = "chroma_db"
DATA_PATH = "data/"
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets/logo.png")

# Encode image
def get_image_base64(path):
    """Encodes a local image file into Base64."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode().replace("\n", "")
    
# Page config
st.set_page_config(page_title="Local RAG Chat", page_icon="💬", layout="wide")
st.title("💬 Local RAG Chat with Ollama")

logo_base64 = get_image_base64(LOGO_PATH)

APP_TITLE = "ChatSTP"

logo_html = f"<img src='data:image/png;base64,{logo_base64}' alt='App Logo'>" if logo_base64 else ""

HEADER_HTML = f"""
<style>
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        padding: 0.5rem 1rem;
        margin-bottom: 0;
        background-color: var(--default-backgroundColor, #ffffff);
        border-bottom: 1px solid var(--default-borderColor, #e6e6e6);
        
        /* Flexbox for layout */
        display: flex;
        align-items: center;
    }}

    .fixed-header img {{
        width: 50px; /* Adjust logo size */
        margin-right: 1rem;
        vertical-align: middle;
    }}

    .fixed-header h1 {{
        font-size: 1.75rem; /* Match Streamlit's title size */
        font-weight: 600;
        margin: 0;
        padding: 0;
        line-height: 1.2;
        color: var(--default-textColor, #31333F);
    }}
    
    /* Push main content down to avoid overlap with fixed header */
    div[data-testid="stAppViewContainer"] {{
        padding-top: 3rem; /* Adjust this value based on header height */
    }}
</style>

<div class="fixed-header">
    {logo_html}
    <h1>{APP_TITLE}</h1>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = db_exists()
if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar for PDF Upload and Processing
with st.sidebar:
    st.header("Document Processing")
    uploaded_files = st.file_uploader(
        "📄 Upload one or more PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        disabled=st.session_state.processing # Disabling upload while processing
    )

    if st.button("Process Documents", disabled=st.session_state.processing or not uploaded_files):
        if uploaded_files:
            st.session_state.processing = True
            with st.spinner("Processing documents... This may take a moment."):
                try:
                    # 1. Clear old data
                    st.info("Clearing old data...")
                    clear_data_directory()
                    
                    # 2. Save new files
                    st.info(f"Saving {len(uploaded_files)} new file(s)...")
                    os.makedirs(DATA_PATH, exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join(DATA_PATH, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.read())
                    
                    # 3. Load, split, and index
                    st.info("Loading documents...")
                    docs = load_documents()
                    st.info("Splitting documents...")
                    chunks = split_documents(docs)
                    st.info("Getting embedding function...")
                    embedding_function = get_embedding_function()
                    st.info("Indexing documents (this will clear the old index)...")
                    index_documents(chunks, embedding_function)
                    
                    st.session_state.db_ready = True
                    st.success("✅ Documents processed successfully!")
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                finally:
                    st.session_state.processing = False
                    st.rerun()
        else:
            st.warning("Please upload at least one PDF file.")

# Main Chat Interface

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if the database is ready
if not st.session_state.db_ready:
    st.info("Please upload PDF(s) and click 'Process Documents' in the sidebar.")
else:
    # Chat input
    question = st.chat_input("Ask a question about your document(s):", disabled=st.session_state.processing)
    
    if question:
        st.session_state.processing = True # Lock UI
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    embedding_function = get_embedding_function()
                    vector_store = get_vector_store(embedding_function)
                    rag_chain = create_rag_chain(vector_store)
                    
                    # Get response
                    response = query_rag(rag_chain, question)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred while getting response: {e}")
            
        st.session_state.processing = False # Unlock UI
        st.rerun() # Refresh to clear input box lock