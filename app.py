import os
import streamlit as st
from dotenv import load_dotenv
from chunks import Chunks
from vectorstore import Vectorstore
from text import Text
from chain import Chain
from htmlTemplates import css, bot_template, user_template
from userinput import UserInput

def main():
    load_dotenv()
    ui = UserInput()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask a question related to PDF")
    if user_question and 'conversation' in st.session_state:
        ui.handle_userinput(user_question)

   
    with st.sidebar:
        st.subheader("Provided Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)

        if st.button("Clear History"):
             st.session_state.chat_history = []

        if pdf_docs:
            temp_files = []
            for i, pdf in enumerate(pdf_docs):
                temp_path = f"temp_{i}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(pdf.read())
                temp_files.append(temp_path)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text from PDFs
                text = Text()
                raw_text = ""
                for path in temp_files:
                  raw_text += text.get_pdf_text(path) + "\n"

                # Split text into chunks
                chunk = Chunks()
                text_chunks = chunk.get_text_chunks(raw_text)

                # Vectorstore
                def chroma_db_exists(path: str = "./chroma_db") -> bool:
                    return os.path.exists(os.path.join(path, "chroma-collections.parquet"))
                
                vs = Vectorstore()

                if chroma_db_exists():
                    vectorstore = vs.fetch_vectorstore()
                else:
                    vectorstore = vs.create_vectorstore(text_chunks)

                # Create conversation chain
                chain = Chain()
                st.session_state.conversation = chain.get_conversation_chain(vectorstore)

                # Reset chat history when new docs are uploaded
                st.session_state.chat_history = []

        #ui.get_conversation_history()

if __name__ == '__main__':
    main()
