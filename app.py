import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from chunks import Chunks
from vectorstore import Vectorstore
from data import Data
from chain import Chain
from htmlTemplates import css, bot_template, user_template
from userinput import UserInput
from langchain.schema import Document

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs 📚")

    vs = Vectorstore()
    ui = UserInput()
    data = Data()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask a question related to PDF")
    if user_question and 'conversation' in st.session_state:
        ui.handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)
        if pdf_docs:
            # Save all uploaded PDFs temporarily
            temp_files = []
            for i, pdf in enumerate(pdf_docs):
                temp_path = f"temp_{i}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(pdf.read())
                temp_files.append(temp_path)

            if st.button("Process"):
                with st.spinner("Processing PDFs..."):
                    # Step 2: Create document objects with metadata
                    documents = []
                    all_image_paths = []

                    if os.path.exists("extracted_images"):
                        shutil.rmtree("extracted_images")

                    all_documents = []
                    all_image_paths = []

                    for file in temp_files:
                        docs, images = data.extract_all_content(file)  
                        all_documents.extend(docs)                     
                        all_image_paths.extend(images)                 

                    # Step 3: Pass to vectorstore (it will chunk internally)
                    vectorstore = vs.create_vectorstore(documents)

                    # Step 4: Build the chain
                    chain = Chain()
                    st.session_state.conversation = chain.get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.success("PDFs processed successfully!")

                    if st.button("Clear History"):
                        st.session_state.chat_history = []
                        st.session_state.conversation = None
                        st.success("Chat history cleared.")

                    if st.button("🧹 Clear all previous documents"):
                        vs.clear_vectorstore()


if __name__ == "__main__":
    main()
