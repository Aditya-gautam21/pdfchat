import streamlit as st
from htmlTemplates import css, user_template, bot_template

class UserInput:
    def handle_userinput(self, user_question):
        # Enhanced debugging section
        with st.expander("🔍 Debug: Retrieved Content", expanded=False):
            try:
                retriever = st.session_state.conversation.retriever
                relevant_docs = retriever.get_relevant_documents(user_question)
                
                st.write(f"**Question:** {user_question}")
                st.write(f"**Retrieved {len(relevant_docs)} documents:**")
                
                # Count document types with new structure
                page_docs = len([d for d in relevant_docs if d.metadata.get("type") == "page"])
                overview_docs = len([d for d in relevant_docs if d.metadata.get("type") == "overview"])
                
                # Count content within pages
                docs_with_text = len([d for d in relevant_docs if d.metadata.get("has_text", False)])
                docs_with_tables = len([d for d in relevant_docs if d.metadata.get("has_tables", False)])
                docs_with_images = len([d for d in relevant_docs if d.metadata.get("has_images", False)])
                
                st.write(f"📄 Page documents: {page_docs}")
                st.write(f"📋 Overview documents: {overview_docs}")
                st.write(f"📝 Docs with text: {docs_with_text}")
                st.write(f"📊 Docs with tables: {docs_with_tables}")
                st.write(f"🖼️ **Docs with images: {docs_with_images}**")
                
                # Show context quality
                total_context_length = sum(len(doc.page_content) for doc in relevant_docs)
                st.write(f"📏 Total context length: {total_context_length} characters")
                
                for i, doc in enumerate(relevant_docs[:4]):
                    doc_type = doc.metadata.get("type", "unknown")
                    content_type = doc.metadata.get("content_type", "unknown")
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source", "unknown")
                    
                    st.write(f"**Chunk {i+1}:**")
                    st.write(f"- Type: **{doc_type}** ({content_type})")
                    st.write(f"- Page: {page_num}")
                    st.write(f"- Source: {source}")
                    
                    # Show content flags
                    flags = []
                    if doc.metadata.get("has_text", False):
                        flags.append("📝 Text")
                    if doc.metadata.get("has_tables", False):
                        flags.append("📊 Tables")
                    if doc.metadata.get("has_images", False):
                        flags.append("🖼️ Images")
                    
                    if flags:
                        st.write(f"- Content: {' | '.join(flags)}")
                    
                    st.write(f"- Preview: {doc.page_content[:300]}...")
                    st.write("---")
                    
            except Exception as e:
                st.error(f"Debug error: {e}")
        
        # CRITICAL: Pre-validate that we have relevant context
        try:
            retriever = st.session_state.conversation.retriever
            relevant_docs = retriever.get_relevant_documents(user_question)
            
            if not relevant_docs:
                st.warning("⚠️ No relevant content found in the PDF for this question.")
                return
                
            # Check if context is substantial enough
            total_context = " ".join([doc.page_content for doc in relevant_docs])
            if len(total_context.strip()) < 50:
                st.warning("⚠️ Very limited relevant content found for this question.")
                
        except Exception as e:
            st.error(f"Error checking context relevance: {e}")
        
        # Get the response
        response = st.session_state.conversation({
            'question': user_question,
            'chat_history': st.session_state.chat_history
        })
        
        answer = response['answer'] if isinstance(response, dict) else response
        
        # CRITICAL: Post-process answer to catch hallucinations
        if any(phrase in answer.lower() for phrase in [
            "i cannot find information about this in the provided pdf",
            "the provided documents do not contain",
            "i cannot find clear information about this in the pdf",
            "not mentioned in the provided context",
            "no information available in the pdf"
        ]):
            st.info("ℹ️ The assistant correctly identified that this information is not in the PDF.")
        
        # Enhanced source documents display
        if isinstance(response, dict) and 'source_documents' in response:
            with st.expander("📚 Sources Used in Answer", expanded=False):
                image_sources = 0
                table_sources = 0
                text_sources = 0
                
                for i, doc in enumerate(response['source_documents']):
                    doc_type = doc.metadata.get("type", "unknown")
                    content_type = doc.metadata.get("content_type", "unknown")
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source", "unknown")
                    
                    # Count content types
                    if doc.metadata.get("has_images", False):
                        image_sources += 1
                    if doc.metadata.get("has_tables", False):
                        table_sources += 1
                    if doc.metadata.get("has_text", False):
                        text_sources += 1
                    
                    st.write(f"**Source {i+1}:**")
                    st.write(f"- Type: {doc_type} ({content_type})")
                    st.write(f"- Page: {page_num} | File: {source}")
                    
                    # Content flags
                    content_flags = []
                    if doc.metadata.get("has_text", False):
                        content_flags.append("📝")
                    if doc.metadata.get("has_tables", False):
                        content_flags.append("📊")
                    if doc.metadata.get("has_images", False):
                        content_flags.append("🖼️")
                    
                    if content_flags:
                        st.write(f"- Contains: {' '.join(content_flags)}")
                    
                    st.write(f"- Content: {doc.page_content[:300]}...")
                    st.write("---")
                
                # Summary of sources used
                st.write("**Source Summary:**")
                if text_sources > 0:
                    st.success(f"✅ Used {text_sources} text-based sources")
                if table_sources > 0:
                    st.success(f"✅ Used {table_sources} sources with tables")
                if image_sources > 0:
                    st.success(f"✅ Used {image_sources} sources with image content")
                
                if image_sources == 0 and table_sources == 0:
                    st.warning("⚠️ Only text sources were used - no images or tables referenced")
        
        # Update history
        st.session_state.chat_history.append((user_question, answer))
        
        # Display conversation
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

    def get_conversation_history(self):
        with st.expander("Conversation History"):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                **Q{i+1}:** {q}
                **A{i+1}:** {a}
                ---
                """)
