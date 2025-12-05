import streamlit as st
import time
from rag_engine import initialize_rag_pipeline

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a medical/clean look
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stMarkdown {
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=80)
    st.title("Configuration")
    st.markdown("---")
    st.info("**Model:** Llama-3.2-1B (Quantized)")
    st.info("**Data Source:** Mol-Instructions (Filtered)")
    
    max_samples = st.slider("Max Documents to Load", min_value=100, max_value=2000, value=500, step=100)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Content
st.title("🧬 Skin Cancer Mutation AI Assistant")
st.markdown("""
This AI assistant helps clinicians and researchers explore **skin cancer mutations** (e.g., BRAF, NRAS).
It uses a **RAG (Retrieval-Augmented Generation)** pipeline to ground answers in scientific data.
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    @st.cache_resource
    def _initialize_rag_pipeline(max_samples_val):
        return initialize_rag_pipeline(max_samples=max_samples_val)

    with st.spinner("Initializing AI System (Loading Models & Index)... This may take a minute."):
        try:
            st.session_state.query_engine = _initialize_rag_pipeline(max_samples)
            st.success("System Initialized Successfully!")
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about a mutation (e.g., 'What is the effect of BRAF V600E?'):"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing scientific data..."):
            try:
                response = st.session_state.query_engine.query(prompt)
                full_response = str(response)
                
                # Simulate typing effect
                message_placeholder.markdown(full_response)
                
                # Show sources in an expander
                with st.expander("📚 View Source Documents"):
                    for node in response.source_nodes:
                        st.markdown(f"**Score:** {node.score:.3f}")
                        st.caption(node.text[:300] + "...")
                        st.markdown("---")
                        
            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.error(full_response)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
