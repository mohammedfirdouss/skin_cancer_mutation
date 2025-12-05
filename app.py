"""
Streamlit Web Application for the Skin Cancer Mutation AI Assistant.

This application provides a user-friendly chat interface to interact with the RAG
pipeline defined in `rag_engine.py`. It allows users to ask questions about
skin cancer mutations and receive answers grounded in scientific data.
"""

import streamlit as st
from rag_engine import create_rag_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Skin Cancer AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a clean, professional look ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f4f6f8;
    }
    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Sidebar styling */
    .st-emotion-cache-18ni7ap {
        background-color: #ffffff;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #007bff;
        color: #007bff;
    }
    .stButton>button:hover {
        border-color: #0056b3;
        color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialization and Caching ---
# Use st.cache_resource to initialize the RAG engine once and cache it across sessions.
# This is the most important step for performance, ensuring models are not reloaded.
@st.cache_resource
def get_rag_engine():
    """Initializes and returns the RAG query engine."""
    with st.spinner("Initializing AI System... This may take a moment on first launch."):
        try:
            engine = create_rag_engine()
            return engine
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize the RAG engine. Details: {e}")
            st.stop()

query_engine = get_rag_engine()


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=70)
    st.title("AI Assistant")
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about skin cancer mutations.

    **Technology Stack:**
    - **LLM:** `Llama-3.2-1B (Quantized)`
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **Vector Store:** `ChromaDB`
    - **Framework:** `LlamaIndex`
    """)
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
        st.rerun()

# --- Main Application ---
st.title("🧬 Skin Cancer Mutation AI Assistant")
st.markdown("Ask a question about a skin cancer mutation, its effects, or related therapies.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("e.g., 'What is the clinical significance of BRAF V600E?'"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching scientific literature and databases..."):
            try:
                # Query the RAG engine
                response_data = query_engine.query(prompt)
                
                # Extract the answer and display it
                answer = response_data['response']
                st.markdown(answer)

                # Display source documents in an expander for traceability
                with st.expander("📚 View Sources"):
                    if response_data['source_nodes']:
                        for node in response_data['source_nodes']:
                            st.markdown(f"**Score:** {node.score:.3f}")
                            st.caption(f"**Source:** `{node.metadata.get('source', 'N/A')}`")
                            st.info(node.get_text()[:400] + "...")
                            st.markdown("---")
                    else:
                        st.warning("No specific source documents were retrieved for this query.")
                
                # Add the full response (with sources) to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"Sorry, an error occurred while processing your request. Please try again. Details: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})