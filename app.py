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
    
    # CRITICAL: Check for existing vectorstore on startup
    if 'conversation' not in st.session_state and vs.chroma_db_exists():
        try:
            st.info("🔄 Loading existing documents from persistent storage...")
            vectorstore = vs.fetch_vectorstore()
            
            # Test if vectorstore has content
            try:
                test_docs = vectorstore.similarity_search("test", k=1)
                if test_docs:
                    chain = Chain()
                    st.session_state.conversation = chain.get_conversation_chain(vectorstore)
                    st.success(f"✅ Loaded existing documents from ChromaDB!")
                    
                    # Get document count and types
                    collection = vectorstore._collection
                    total_docs = collection.count()
                    st.info(f"📋 Found {total_docs} document chunks in persistent storage")
                else:
                    st.warning("⚠️ ChromaDB exists but appears empty")
            except Exception as test_e:
                st.warning(f"⚠️ Error testing vectorstore content: {test_e}")
                
        except Exception as e:
            st.error(f"Error loading existing vectorstore: {e}")
    
    user_question = st.chat_input("Ask a question related to PDF")
    
    if user_question and 'conversation' in st.session_state:
        ui.handle_userinput(user_question)
    elif user_question:
        st.warning("⚠️ Please upload and process PDFs first before asking questions.")
    
    with st.sidebar:
        st.subheader("Upload PDFs")
        
        # Show current status
        if vs.chroma_db_exists():
            try:
                vectorstore = vs.fetch_vectorstore()
                collection = vectorstore._collection
                doc_count = collection.count()
                st.success(f"💾 {doc_count} chunks in persistent memory")
            except:
                st.info("💾 Persistent storage exists")
        else:
            st.info("📝 No documents in persistent memory")
        
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here (Max 10MB per file)",
            type="pdf",
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared.")
                st.rerun()
        
        with col2:
            # Only show clear button if vectorstore exists
            if vs.chroma_db_exists():
                if st.button("🧹 Clear Documents"):
                    try:
                        vs.clear_vectorstore()
                        if 'conversation' in st.session_state:
                            del st.session_state.conversation
                        st.success("Documents cleared from persistent storage.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing documents: {e}")
        
        if pdf_docs:
            oversized_files = [pdf.name for pdf in pdf_docs if pdf.size > MAX_FILE_SIZE]
            if oversized_files:
                st.error(f"Files too large (>10MB): {', '.join(oversized_files)}")
                return
            
            # Show options for handling existing data
            if vs.chroma_db_exists():
                st.warning("📋 Existing documents found in persistent storage")
                append_mode = st.radio(
                    "Choose action:",
                    ["Add to existing documents", "Replace all documents"],
                    index=0
                )
            else:
                append_mode = "Add to existing documents"
            
            if st.button("Process"):
                try:
                    with st.spinner("Processing PDFs..."):
                        # Clear existing if replacing
                        if append_mode == "Replace all documents" and vs.chroma_db_exists():
                            vs.clear_vectorstore()
                            if 'conversation' in st.session_state:
                                del st.session_state.conversation
                        
                        temp_files = []
                        try:
                            for i, pdf in enumerate(pdf_docs):
                                temp_path = f"temp_{i}.pdf"
                                with open(temp_path, "wb") as f:
                                    f.write(pdf.read())
                                temp_files.append(temp_path)
                            
                            if os.path.exists("extracted_images"):
                                shutil.rmtree("extracted_images")
                            
                            all_documents = []
                            all_image_paths = []
                            
                            processing_status = st.empty()
                            for i, file in enumerate(temp_files):
                                try:
                                    processing_status.info(f"Processing PDF {i+1}/{len(temp_files)}: {os.path.basename(file)}")
                                    docs, images = data.extract_all_content(file)
                                    all_documents.extend(docs)
                                    all_image_paths.extend(images)
                                    
                                    # Count document types with new structure
                                    page_docs = len([d for d in docs if d.metadata.get("type") == "page"])
                                    overview_docs = len([d for d in docs if d.metadata.get("type") == "overview"])
                                    
                                    # Count content within pages
                                    pages_with_images = len([d for d in docs if d.metadata.get("has_images", False)])
                                    pages_with_tables = len([d for d in docs if d.metadata.get("has_tables", False)])
                                    
                                    st.write(f"📄 {os.path.basename(file)}: {page_docs} pages, {overview_docs} overview")
                                    if pages_with_images > 0:
                                        st.write(f"   🖼️ {pages_with_images} pages contain images")
                                    if pages_with_tables > 0:
                                        st.write(f"   📊 {pages_with_tables} pages contain tables")
                                    
                                except Exception as e:
                                    st.error(f"Error processing {os.path.basename(file)}: {e}")
                                    continue
                            
                            processing_status.empty()
                            
                            if not all_documents:
                                st.error("No content could be extracted from the PDFs.")
                                return
                            
                            content_summary = data.get_content_summary(all_documents)
                            with st.expander("📊 Content Extraction Summary"):
                                st.text(content_summary)
                            
                            # Create or append to vectorstore
                            st.info("Creating/updating vector embeddings in persistent storage...")
                            vectorstore = vs.create_vectorstore(all_documents)
                            
                            chain = Chain()
                            st.session_state.conversation = chain.get_conversation_chain(vectorstore)
                            st.session_state.chat_history = []
                            
                            # Updated success message for new document structure
                            page_docs = len([d for d in all_documents if d.metadata.get("type") == "page"])
                            overview_docs = len([d for d in all_documents if d.metadata.get("type") == "overview"])
                            
                            st.success(f"✅ Successfully processed {len(pdf_docs)} PDFs!")
                            st.info(f"📋 Created {len(all_documents)} contextual documents:")
                            st.info(f"   • {page_docs} page-based documents")
                            st.info(f"   • {overview_docs} document overviews")
                            
                            # Count content types
                            pages_with_text = len([d for d in all_documents if d.metadata.get("has_text", False)])
                            pages_with_tables = len([d for d in all_documents if d.metadata.get("has_tables", False)])
                            pages_with_images = len([d for d in all_documents if d.metadata.get("has_images", False)])
                            
                            if pages_with_text > 0:
                                st.info(f"📝 {pages_with_text} pages with text content")
                            if pages_with_tables > 0:
                                st.info(f"📊 {pages_with_tables} pages with tables")
                            if pages_with_images > 0:
                                st.info(f"🖼️ {pages_with_images} pages with images processed via OCR")
                            
                            st.success("💾 Documents saved to persistent ChromaDB storage!")
                            
                            # Show final storage stats
                            try:
                                collection = vectorstore._collection
                                total_chunks = collection.count()
                                st.info(f"📊 Total chunks in persistent storage: {total_chunks}")
                            except:
                                pass
                            
                        finally:
                            for temp_file in temp_files:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except Exception as cleanup_e:
                                    st.warning(f"Could not clean up {temp_file}: {cleanup_e}")
                                    
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

if __name__ == "__main__":
    main()
