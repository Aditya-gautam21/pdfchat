# app.py - UPDATED FOR BETTER CLEARING AND TABLE FOCUS

import os
import shutil
import streamlit as st
import time
from dotenv import load_dotenv
from vectorstore import Vectorstore
from data import Data
from chain import Chain
from htmlTemplates import css, bot_template, user_template
from userinput import UserInput
from langchain.schema import Document

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Enhanced PDF Chat with Tables & Images",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    st.header("Enhanced PDF Chat with Tables & Images 📚")

    # Initialize components
    vs = Vectorstore()
    ui = UserInput()
    data = Data()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Enhanced startup with better vectorstore handling
    if 'conversation' not in st.session_state and vs.chroma_db_exists():
        with st.spinner("🔄 Loading existing documents from persistent storage..."):
            try:
                vectorstore = vs.fetch_vectorstore()
                test_docs = vectorstore.similarity_search("test", k=1)
                if test_docs:
                    chain = Chain()
                    st.session_state.conversation = chain.get_conversation_chain(vectorstore)
                    stats = vs.get_vectorstore_stats()
                    if stats.get("exists", False) and "error" not in stats:
                        st.success("✅ Loaded existing documents from ChromaDB!")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Chunks", stats.get("total_chunks", 0))
                        with col2:
                            st.metric("Table Chunks", stats.get("table_chunks", 0))
                        with col3:
                            st.metric("Image Chunks", stats.get("image_chunks", 0))
                        st.info(f"📋 Content breakdown:")
                        st.info(f" • Pages: {stats.get('page_chunks', 0)}")
                        st.info(f" • Tables: {stats.get('table_chunks', 0)} ({stats.get('complete_tables', 0)} complete)")
                        st.info(f" • Images: {stats.get('image_chunks', 0)}")
                        st.info(f" • Overviews: {stats.get('overview_chunks', 0)}")
                    else:
                        st.warning("⚠️ ChromaDB exists but has issues")
                else:
                    st.warning("⚠️ ChromaDB exists but appears empty")
            except Exception as e:
                st.error(f"Error loading existing vectorstore: {e}")
                if st.button("🔧 Clear Corrupted Database"):
                    vs.clear_vectorstore()
                    st.rerun()

    # Enhanced chat input with query type detection
    user_question = st.chat_input("Ask about your PDFs (tables, images, data, or general questions)")
    if user_question and 'conversation' in st.session_state:
        query_lower = user_question.lower()
        query_type = "General"
        if any(kw in query_lower for kw in ['table', 'data', 'number', 'amount', 'show', 'list']):
            query_type = "📊 Table/Data Query"
        elif any(kw in query_lower for kw in ['image', 'picture', 'chart', 'figure']):
            query_type = "🖼️ Image Query"
        elif any(kw in query_lower for kw in ['overview', 'summary', 'about', 'general']):
            query_type = "📋 Overview Query"
        st.info(f"Query Type Detected: {query_type}")
        ui.handle_userinput(user_question)
    elif user_question:
        st.warning("⚠️ Please upload and process PDFs first before asking questions.")

    # Enhanced sidebar
    with st.sidebar:
        st.subheader("📁 Document Management")
        # Current status
        if vs.chroma_db_exists():
            try:
                stats = vs.get_vectorstore_stats()
                if stats.get("exists", False) and "error" not in stats:
                    st.success("💾 Persistent Storage Active")
                    with st.expander("📊 Storage Details", expanded=False):
                        st.write(f"**Total Chunks:** {stats.get('total_chunks', 0)}")
                        st.write(f"**Table Chunks:** {stats.get('table_chunks', 0)}")
                        st.write(f"**Complete Tables:** {stats.get('complete_tables', 0)}")
                        st.write(f"**Image Chunks:** {stats.get('image_chunks', 0)}")
                        st.write(f"**Page Chunks:** {stats.get('page_chunks', 0)}")
                else:
                    st.error("❌ Storage has errors")
                    if st.button("🔧 Fix Storage Issues"):
                        vs.clear_vectorstore()
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Storage error: {e}")
        else:
            st.info("📝 No persistent storage found")
        st.divider()

        # Enhanced file upload
        st.subheader("📤 Upload PDFs")
        MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB
        MAX_FILES = 5
        pdf_docs = st.file_uploader(
            f"Upload PDFs (Max {MAX_FILES} files, {MAX_FILE_SIZE//1024//1024}MB each)",
            type="pdf",
            accept_multiple_files=True
        )

        # File validation
        if pdf_docs:
            if len(pdf_docs) > MAX_FILES:
                st.error(f"❌ Too many files! Maximum {MAX_FILES} allowed.")
                pdf_docs = None
            else:
                oversized_files = [pdf.name for pdf in pdf_docs if pdf.size > MAX_FILE_SIZE]
                if oversized_files:
                    st.error(f"❌ Files too large: {', '.join(oversized_files)}")
                    pdf_docs = None
                else:
                    total_size = sum(pdf.size for pdf in pdf_docs)
                    st.success(f"✅ {len(pdf_docs)} files ready ({total_size//1024//1024:.1f}MB total)")

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat history cleared.")
                st.rerun()
        with col2:
            if vs.chroma_db_exists():
                if st.button("🧹 Clear Docs", use_container_width=True):
                    try:
                        with st.spinner("🧹 Clearing..."):
                            vs.clear_vectorstore()
                            if vs.verify_cleared():
                                st.success("✅ Completely cleared!")
                            else:
                                st.warning("⚠️ Partial clear; restart app if issues persist")
                            # Remove conversation from session state
                            if 'conversation' in st.session_state:
                                del st.session_state.conversation
                            # Clear chat history
                            if 'chat_history' in st.session_state:
                                st.session_state.chat_history = []
                            import gc
                            gc.collect()
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.info("🔧 Try restarting the application if the issue persists")

        # Processing options
        if pdf_docs:
            st.divider()
            st.subheader("⚙️ Processing Options")
            if vs.chroma_db_exists():
                st.warning("📋 Existing documents found")
                append_mode = st.radio(
                    "Choose action:",
                    ["Add to existing documents", "Replace all documents"],
                    index=0,
                    help="Add: Merge with existing docs. Replace: Start fresh."
                )
            else:
                append_mode = "Add to existing documents"
            # Advanced options
            with st.expander("🔧 Advanced Options", expanded=False):
                extract_tables = st.checkbox("📊 Enhanced Table Extraction", value=True)
                extract_images = st.checkbox("🖼️ Enhanced Image OCR", value=True)
                high_quality_ocr = st.checkbox("🔍 High Quality OCR", value=True)
            # Process button
            if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
                process_documents(pdf_docs, vs, data, append_mode, {
                    'extract_tables': extract_tables,
                    'extract_images': extract_images,
                    'high_quality_ocr': high_quality_ocr
                })

        # Debug tools
        if 'conversation' in st.session_state:
            st.divider()
            st.subheader("🔧 Debug Tools")
            if st.button("🧪 Test Table Queries"):
                test_table_functionality()
            if st.button("🔍 Debug Retrieval"):
                debug_retrieval_system()
            with st.expander("⚡ Quick Test Queries", expanded=False):
                test_queries = [
                    "What tables are in this document?",
                    "Show me numerical data",
                    "What images are in this document?",
                    "Give me an overview"
                ]
                for query in test_queries:
                    if st.button(f"Test: {query}", key=f"test_{hash(query)}"):
                        ui.handle_userinput(query)

def process_documents(pdf_docs, vs, data, append_mode, options):
    """Enhanced document processing with better error handling and feedback"""
    try:
        with st.spinner("🔄 Processing PDFs with enhanced extraction..."):
            start_time = time.time()
            # Clear existing if replacing
            if append_mode == "Replace all documents" and vs.chroma_db_exists():
                vs.clear_vectorstore()
                if 'conversation' in st.session_state:
                    del st.session_state.conversation
                st.info("🧹 Cleared existing documents")
            
            # Create temporary files with original names
            temp_files = []
            original_names = []  # NEW: Track original names
            try:
                for i, pdf in enumerate(pdf_docs):
                    original_name = pdf.name  # Use original uploaded name
                    original_names.append(original_name)
                    temp_path = os.path.join("temp_pdfs", original_name)  # NEW: Use original name for temp file
                    os.makedirs("temp_pdfs", exist_ok=True)  # Ensure temp dir exists
                    with open(temp_path, "wb") as f:
                        f.write(pdf.read())
                    temp_files.append(temp_path)
        
                # Clean up existing extracted images
                if os.path.exists("extracted_images"):
                    shutil.rmtree("extracted_images")
                all_documents = []
                all_image_paths = []
                # Process each file with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, file_path in enumerate(temp_files):
                    filename = original_names[i]  # NEW: Use original name
                    progress = (i + 1) / len(temp_files)
                    status_text.info(f"📄 Processing {filename} ({i+1}/{len(temp_files)})")
                    progress_bar.progress(progress)
                    try:
                        docs, images = data.extract_all_content(file_path)
                        for doc in docs:
                            doc.metadata["source"] = filename
                        all_documents.extend(docs)
                        all_image_paths.extend(images)

                        # Show extraction results
                        page_docs = len([d for d in docs if d.metadata.get("type") == "page"])
                        table_docs = len([d for d in docs if d.metadata.get("type") == "table"])
                        image_docs = len([d for d in docs if d.metadata.get("type") == "image"])
                        overview_docs = len([d for d in docs if d.metadata.get("type") == "overview"])
                        st.write(f"✅ {os.path.basename(file_path)}:")
                        st.write(f" 📄 Pages: {page_docs} | 📊 Tables: {table_docs} | 🖼️ Images: {image_docs} | 📋 Overview: {overview_docs}")
                    except Exception as e:
                        st.error(f"❌ Error processing {filename}: {e}")
                        continue
                progress_bar.empty()
                status_text.empty()
                if not all_documents:
                    st.error("❌ No content could be extracted from the PDFs.")
                    return
                # Show comprehensive extraction summary
                with st.expander("📊 Extraction Summary", expanded=True):
                    doc_types = {}
                    for doc in all_documents:
                        doc_type = doc.metadata.get("type", "unknown")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 Pages", doc_types.get("page", 0))
                    with col2:
                        st.metric("📊 Tables", doc_types.get("table", 0))
                    with col3:
                        st.metric("🖼️ Images", doc_types.get("image", 0))
                    with col4:
                        st.metric("📋 Overviews", doc_types.get("overview", 0))
                    st.write(f"**Total Documents Created:** {len(all_documents)}")
                    # Content quality metrics
                    table_chunks = len([d for d in all_documents if d.metadata.get("table_chunk", False)])
                    complete_tables = len([d for d in all_documents if d.metadata.get("is_complete_table", False)])
                    image_chunks = len([d for d in all_documents if d.metadata.get("image_chunk", False)])
                    st.write(f"**Quality Metrics:**")
                    st.write(f" • Table chunks: {table_chunks} ({complete_tables} complete)")
                    st.write(f" • Image chunks: {image_chunks}")
                    st.write(f" • Average content per doc: {sum(len(d.page_content) for d in all_documents) // len(all_documents):,} chars")
                # Create/update vectorstore
                with st.spinner("🔗 Creating vector embeddings..."):
                    vectorstore = vs.create_vectorstore(all_documents)

                # Initialize conversation chain
                chain = Chain()
                st.session_state.conversation = chain.get_conversation_chain(vectorstore)

                # NEW: Force reload to ensure persistent memory is updated
                del st.session_state.conversation  # Clear old reference
                gc.collect()
                vectorstore = vs.fetch_vectorstore()  # Reload fresh
                st.session_state.conversation = chain.get_conversation_chain(vectorstore)
                
                st.session_state.chat_history = []
                # Final success message
                processing_time = time.time() - start_time
                st.success(f"🎉 Successfully processed {len(pdf_docs)} PDFs in {processing_time:.1f}s!")
                # Show final statistics
                final_stats = vs.get_vectorstore_stats()
                if final_stats.get("exists", False):
                    st.info(f"💾 Persistent storage updated:")
                    st.info(f" 📊 Total chunks: {final_stats.get('total_chunks', 0)}")
                    st.info(f" 📋 Table chunks: {final_stats.get('table_chunks', 0)} (high priority)")
                    st.info(f" 🖼️ Image chunks: {final_stats.get('image_chunks', 0)}")
                # Quick functionality test
                if st.checkbox("🧪 Run quick functionality test", value=True):
                    with st.spinner("Testing system functionality..."):
                        try:
                            test_results = chain.test_chain_functionality(st.session_state.conversation)
                            if test_results is None:
                                st.warning("⚠️ Test function returned None - functionality test skipped")
                            elif not test_results:
                                st.warning("⚠️ No test results returned")
                            else:
                                success_count = sum(1 for r in test_results.values() if r.get("success", False))
                                total_tests = len(test_results)
                                if success_count == total_tests:
                                    st.success(f"✅ All {total_tests} functionality tests passed!")
                                else:
                                    st.warning(f"⚠️ {success_count}/{total_tests} tests passed")
                                # Show failed tests
                                failed_tests = [query for query, result in test_results.items() if not result.get("success", False)]
                                if failed_tests:
                                    with st.expander("Failed Tests", expanded=False):
                                        for failed_query in failed_tests:
                                            st.error(f"❌ {failed_query}: {test_results[failed_query].get('error', 'Unknown error')}")
                        except Exception as test_e:
                            st.error(f"❌ Error running functionality tests: {test_e}")
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as cleanup_e:
                        st.warning(f"Could not clean up {temp_file}: {cleanup_e}")
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {e}")
        import traceback
        st.code(traceback.format_exc())

def test_table_functionality():
    """Test table-specific functionality"""
    if 'conversation' not in st.session_state:
        st.error("No conversation available for testing")
        return
    st.subheader("🧪 Table Functionality Test")
    table_queries = [
        "What tables are in this document?",
        "Show me all numerical data from tables",
        "List the contents of any tables",
        "What data is available in structured format?",
        "Find any numbers or statistics in the document"
    ]
    chain = Chain()
    with st.spinner("Running table functionality tests..."):
        results = chain.test_chain_functionality(st.session_state.conversation, table_queries)
        for query, result in results.items():
            with st.expander(f"Test: {query}", expanded=False):
                if result.get("success", False):
                    st.success("✅ Test Passed")
                    st.write(f"**Sources:** {result['source_count']}")
                    st.write(f"**Table Sources:** {result['table_sources']}")
                    st.write(f"**Answer Length:** {result['answer_length']} characters")
                    st.write(f"**Contains Structured Data:** {result['contains_structured_data']}")
                    if result['contains_structured_data']:
                        st.info("✅ Response contains structured table data")
                    else:
                        st.warning("⚠️ Response may not contain structured data")
                    st.write("**Answer Preview:**")
                    st.code(result['answer_preview'], language=None)
                else:
                    st.error("❌ Test Failed")
                    st.write(f"**Error:** {result.get('error', 'Unknown error')}")

def debug_retrieval_system():
    """Debug the retrieval system"""
    if 'conversation' not in st.session_state:
        st.error("No conversation available for debugging")
        return
    st.subheader("🔍 Retrieval System Debug")
    vs = Vectorstore()
    vectorstore = vs.fetch_vectorstore()
    chain = Chain()
    debug_query = st.text_input("Enter a query to debug:", "Show me table data")
    if debug_query and st.button("🔍 Debug This Query"):
        with st.spinner("Analyzing retrieval process..."):
            docs = chain.debug_retrieval(vectorstore, debug_query)
            st.write(f"**Query Analysis Results for:** '{debug_query}'")
            # Analyze retrieved documents
            doc_analysis = {
                "total_docs": len(docs),
                "table_docs": len([d for d in docs if d.metadata.get("table_chunk", False)]),
                "image_docs": len([d for d in docs if d.metadata.get("image_chunk", False)]),
                "page_docs": len([d for d in docs if d.metadata.get("type") == "page"]),
                "overview_docs": len([d for d in docs if d.metadata.get("type") == "overview"]),
                "docs_with_table_markers": len([d for d in docs if any(marker in d.page_content for marker in ["===", "TABLE", "|", "STRUCTURED_DATA"])])
            }
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Retrieved", doc_analysis["total_docs"])
                st.metric("Table Docs", doc_analysis["table_docs"])
            with col2:
                st.metric("Image Docs", doc_analysis["image_docs"])
                st.metric("Page Docs", doc_analysis["page_docs"])
            with col3:
                st.metric("Overview Docs", doc_analysis["overview_docs"])
                st.metric("With Table Markers", doc_analysis["docs_with_table_markers"])
            # Quality assessment
            if any(kw in debug_query.lower() for kw in ['table', 'data', 'number']):
                if doc_analysis["table_docs"] > 0 or doc_analysis["docs_with_table_markers"] > 0:
                    st.success("✅ Good table content retrieval for table query")
                else:
                    st.error("❌ Table query but no table content retrieved")
            # Show actual retrieved content
            with st.expander("📋 Retrieved Documents Content", expanded=False):
                for i, doc in enumerate(docs[:5]):
                    st.write(f"**Document {i+1}:**")
                    st.write(f"Type: {doc.metadata.get('type', 'unknown')}")
                    st.write(f"Page: {doc.metadata.get('page_number', 'unknown')}")
                    st.code(doc.page_content[:500] + "...", language=None)
                    st.write("---")

if __name__ == "__main__":
    main()
