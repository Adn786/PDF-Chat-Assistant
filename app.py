"""
Streamlit Frontend for RAG Pipeline
Chat interface with PDF upload and MMR-based retrieval
"""

import streamlit as st
import os
import traceback
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --success: #10b981;
        --danger: #ef4444;
        --dark: #1f2937;
        --light: #f9fafb;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .chat-message {
        padding: 15px 20px;
        border-radius: 12px;
        margin: 10px 0;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: rgba(99, 102, 241, 0.9);
        color: white;
        border-left: 4px solid #6366f1;
        margin-left: 40px;
    }
    .assistant-message {
        background: rgba(255, 255, 255, 0.95);
        color: #1f2937;
        border-left: 4px solid #10b981;
        margin-right: 40px;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    .stTextInput > div > div > input:focus {
        border: 2px solid #6366f1;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    .stFileUploader {
        border-radius: 8px;
        border: 2px dashed #6366f1;
        padding: 20px;
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Header
st.markdown("""
<div style='text-align: center; color: white; padding: 30px 0;'>
    <h1>📄 PDF Chat Assistant</h1>
    <p style='font-size: 18px; opacity: 0.9;'>Ask questions about your PDF documents using AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Document Management")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None and not st.session_state.pdf_loaded:
        st.session_state.pdf_name = uploaded_file.name

        status_box = st.empty()

        def update_status(msg):
            status_box.info(msg)

        try:
            if st.session_state.rag_pipeline is None:
                if not GROQ_API_KEY:
                    st.error("❌ GROQ_API_KEY not found in .env file!")
                    st.stop()
                update_status("⚙️ Initializing pipeline...")
                st.session_state.rag_pipeline = RAGPipeline(GROQ_API_KEY)

            num_chunks = st.session_state.rag_pipeline.process_pdf(
                uploaded_file,
                status_callback=update_status
            )
            st.session_state.pdf_loaded = True
            st.session_state.chat_history = []

            status_box.success(f"✅ Ready! {num_chunks} chunks embedded and stored.")
            st.info(f"📄 Loaded: {uploaded_file.name}")

        except Exception as e:
            status_box.error(f"❌ Error:\n\n{traceback.format_exc()}")
            st.session_state.pdf_loaded = False

    if st.session_state.pdf_loaded:
        if st.button("🔄 Clear & Start Over", use_container_width=True):
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.pdf_loaded = False
            st.session_state.pdf_name = None
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 💡 How to use:
    1. **Upload** a PDF file
    2. **Ask** any question about it
    3. **Get** instant answers powered by AI

    ### ⚙️ Features:
    - MMR-based retrieval
    - Real-time processing
    - Chat-like interface
    """)

# Main chat interface
if st.session_state.pdf_loaded:
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
        <h3 style='color: #6366f1; margin: 0;'>💬 Chat with your PDF</h3>
        <p style='color: #6b7280; margin: 5px 0 0 0;'>Ask questions about: <strong>""" +
        st.session_state.pdf_name + """</strong></p>
    </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Ask a question...",
            placeholder="What is this document about?",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("Send", use_container_width=True)

    if send_button and user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("🤖 Thinking..."):
            try:
                answer = st.session_state.rag_pipeline.chat(user_input)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ Full Error:\n\n{traceback.format_exc()}")
                st.stop()

else:
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style='text-align: center; color: white; padding: 60px 20px;'>
            <div style='font-size: 80px; margin: 20px 0;'>📤</div>
            <h2>Upload a PDF to Start</h2>
            <p style='font-size: 18px; opacity: 0.9;'>
                Use the sidebar to upload your PDF document and start asking questions
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: white; padding: 30px; margin-top: 40px; opacity: 0.8;'>
    <p>Powered by Groq + ChromaDB + MMR Retrieval</p>
    <p style='font-size: 12px;'>Memory is cleared when you close the window</p>
</div>
""", unsafe_allow_html=True)