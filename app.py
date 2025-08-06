import os
import gc
import shutil
import streamlit as st
import time
import tempfile
from dotenv import load_dotenv
from vectorstore import Vectorstore
from data import Data
from chain import Chain
from htmlTemplates import css, bot_template, user_template
from userinput import UserInput
from langchain.schema import Document

def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.header("Enhanced PDF Chat 📚")

    # Initialize components
    vs = Vectorstore()
    ui = UserInput()
    data = Data()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load existing vectorstore if available
    if 'conversation' not in st.session_state and vs.chroma_db_exists():
        with st.spinner("Loading documents..."):
            vectorstore = vs.fetch_vectorstore()
            chain = Chain()
            st.session_state.conversation = chain.get_conversation_chain(vectorstore)
        st.success("Documents loaded!")

    # Chat input
    user_question = st.chat_input("Ask about your PDFs")
    if user_question and 'conversation' in st.session_state:
        ui.handle_userinput(user_question)
    elif user_question:
        st.warning("Upload and process PDFs first.")

    # Sidebar for document management
    with st.sidebar:
        st.subheader("📁 Documents")
        pdf_docs = st.file_uploader("Upload PDFs (max 3, 10MB each)", type="pdf", accept_multiple_files=True)
        if pdf_docs:
            if len(pdf_docs) > 3:
                st.error("Max 3 files allowed.")
                pdf_docs = None
            else:
                if st.button("Process"):
                    process_documents(pdf_docs, vs, data)

        if vs.chroma_db_exists():
            if st.button("Clear Docs"):
                # Enhanced clearing mechanism
                vs.clear_vectorstore()
                # Clear all relevant session state variables
                if 'conversation' in st.session_state:
                    del st.session_state.conversation
                st.session_state.chat_history = []
                # Force garbage collection to release memory
                gc.collect()
                # Inform user and rerun app to refresh state
                st.success("Documents and memory cleared successfully!")
                time.sleep(1)  # Brief delay to ensure clearing
                st.rerun()

    # Advanced options in expander
    with st.expander("Advanced Options"):
        st.checkbox("Extract Tables", value=True)
        st.checkbox("Extract Images", value=True)

@st.cache_resource
def process_documents(pdf_docs, _vs, _data):
    vs = _vs
    data = _data
    with st.spinner("Processing..."):
        all_documents = []
        for pdf in pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf.getvalue())
                temp_path = temp_file.name
            try:
                docs, _ = data.extract_all_content(temp_path)
                all_documents.extend(docs)
            finally:
                os.unlink(temp_path)
        if not all_documents:
            st.error("No content extracted from PDFs. Check if they are valid or contain extractable data.")
            return
        vectorstore = vs.create_vectorstore(all_documents)
        chain = Chain()
        st.session_state.conversation = chain.get_conversation_chain(vectorstore)
        st.success("Processed!")

if __name__ == "__main__":
    main()
