import streamlit as st
from htmlTemplates import css, user_template, bot_template

class UserInput:
    def handle_userinput(self, user_question):
        """Enhanced user input handling with table-focused debugging"""
        
        # CRITICAL: Enhanced debugging for table queries
        with st.expander("🔍 Debug: Query Analysis & Retrieved Content", expanded=False):
            try:
                # Analyze query for table keywords
                table_keywords = [
                    'table', 'data', 'number', 'amount', 'total', 'sum', 'count', 
                    'value', 'figure', 'statistic', 'row', 'column', 'cell',
                    'compare', 'comparison', 'list', 'show me', 'what are',
                    'how many', 'how much', 'calculate', 'add up'
                ]
                
                query_lower = user_question.lower()
                detected_keywords = [kw for kw in table_keywords if kw in query_lower]
                
                if detected_keywords:
                    st.success(f"🎯 Table-related query detected! Keywords: {', '.join(detected_keywords)}")
                else:
                    st.info("📝 General query detected")
                
                # Get retriever and test it
                retriever = st.session_state.conversation.retriever
                relevant_docs = retriever.get_relevant_documents(user_question)
                
                st.write(f"**Question:** {user_question}")
                st.write(f"**Retrieved {len(relevant_docs)} documents:**")
                
                # Detailed content analysis
                page_docs = [d for d in relevant_docs if d.metadata.get("type") == "page"]
                table_docs = [d for d in relevant_docs if d.metadata.get("type") == "table"]
                overview_docs = [d for d in relevant_docs if d.metadata.get("type") == "overview"]
                
                # Critical table metrics
                dedicated_table_chunks = len([d for d in relevant_docs if d.metadata.get("table_chunk", False)])
                complete_tables = len([d for d in relevant_docs if d.metadata.get("is_complete_table", False)])
                docs_with_table_markers = len([d for d in relevant_docs if any(marker in d.page_content for marker in ["===", "TABLE", "|", "Row"])])
                
                st.write(f"📊 **CRITICAL TABLE METRICS:**")
                st.write(f"   - Dedicated table documents: **{len(table_docs)}**")
                st.write(f"   - Table chunks retrieved: **{dedicated_table_chunks}**")
                st.write(f"   - Complete tables: **{complete_tables}**")
                st.write(f"   - Docs with table markers: **{docs_with_table_markers}**")
                
                st.write(f"📄 Page documents: {len(page_docs)}")
                st.write(f"📋 Overview documents: {len(overview_docs)}")
                
                # Context quality analysis
                total_context_length = sum(len(doc.page_content) for doc in relevant_docs)
                table_context_length = sum(len(doc.page_content) for doc in relevant_docs if doc.metadata.get("table_chunk", False))
                
                st.write(f"📏 Total context: {total_context_length:,} characters")
                st.write(f"📊 Table context: {table_context_length:,} characters ({table_context_length/total_context_length*100:.1f}%)")
                
                # Show retrieved documents with table focus
                for i, doc in enumerate(relevant_docs[:8]):
                    doc_type = doc.metadata.get("type", "unknown")
                    content_type = doc.metadata.get("content_type", "unknown")
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source", "unknown")
                    
                    st.write(f"**Document {i+1}:**")
                    st.write(f"- Type: **{doc_type}** ({content_type})")
                    st.write(f"- Page: {page_num} | Source: {source}")
                    
                    # Enhanced content analysis
                    content_flags = []
                    if doc.metadata.get("has_text", False):
                        content_flags.append("📝 Text")
                    if doc.metadata.get("has_tables", False):
                        content_flags.append("📊 Tables")
                    if doc.metadata.get("has_images", False):
                        content_flags.append("🖼️ Images")
                    if doc.metadata.get("table_chunk", False):
                        content_flags.append("🔢 **TABLE CHUNK**")
                    if doc.metadata.get("is_complete_table", False):
                        content_flags.append("📋 **COMPLETE TABLE**")
                    if doc.metadata.get("contains_table_markers", False):
                        content_flags.append("🎯 **HAS TABLE MARKERS**")
                    
                    if content_flags:
                        st.write(f"- Content: {' | '.join(content_flags)}")
                    
                    # CRITICAL: Analyze actual content for table markers
                    content = doc.page_content
                    table_indicators = []
                    
                    if "===" in content:
                        table_indicators.append("Delimiters (===)")
                    if " | " in content:
                        table_indicators.append("Pipe separators (|)")
                    if "TABLE" in content.upper():
                        table_indicators.append("Table labels")
                    if "Row " in content:
                        table_indicators.append("Row indicators")
                    if any(word in content for word in ["Headers:", "DATA ROW"]):
                        table_indicators.append("Structured format")
                    
                    if table_indicators:
                        st.success(f"✅ Table indicators found: {', '.join(table_indicators)}")
                    elif doc.metadata.get("table_chunk", False):
                        st.warning("⚠️ Marked as table chunk but no clear indicators")
                    
                    # Show content preview with table highlighting
                    content_preview = content[:600]
                    if any(indicator in content for indicator in ["===", "|", "TABLE", "Row"]):
                        st.code(content_preview, language=None)  # Code block to preserve formatting
                    else:
                        st.write(f"- Preview: {content_preview}...")
                    
                    st.write("---")
                    
            except Exception as e:
                st.error(f"Debug error: {e}")
        
        # CRITICAL: Pre-validate table content availability
        try:
            retriever = st.session_state.conversation.retriever
            relevant_docs = retriever.get_relevant_documents(user_question)
            
            # Check for table-related query vs table content availability
            table_keywords = ['table', 'data', 'number', 'amount', 'show', 'list']
            is_table_query = any(kw in user_question.lower() for kw in table_keywords)
            
            table_docs_available = len([d for d in relevant_docs if d.metadata.get("table_chunk", False)]) > 0
            table_markers_found = any(
                any(marker in doc.page_content for marker in ["===", "TABLE", "|", "Row"])
                for doc in relevant_docs
            )
            
            if is_table_query and not table_docs_available and not table_markers_found:
                st.warning("⚠️ This appears to be a table-related question, but no table content was found in the relevant documents. The PDF might not contain tables, or they weren't extracted properly.")
            elif is_table_query and (table_docs_available or table_markers_found):
                st.success("✅ Table-related question with table content available!")
                
        except Exception as validation_e:
            st.error(f"Error validating content: {validation_e}")
        
        # Get the response with error handling
        try:
            with st.spinner("Analyzing PDF content..."):
                response = st.session_state.conversation({
                    'question': user_question,
                    'chat_history': st.session_state.chat_history
                })
            
            answer = response['answer'] if isinstance(response, dict) else response
            
            # Enhanced answer analysis
            if any(phrase in answer.lower() for phrase in [
                "i cannot find information about this in the provided pdf",
                "the provided documents do not contain",
                "i cannot find clear information about this in the pdf",
                "not mentioned in the provided context",
                "no information available in the pdf"
            ]):
                st.info("ℹ️ The assistant correctly identified that this information is not in the PDF.")
            
            # CRITICAL: Check if table data was actually used in the answer
            if isinstance(response, dict) and 'source_documents' in response:
                source_docs = response['source_documents']
                table_sources_used = [d for d in source_docs if d.metadata.get("table_chunk", False)]
                
                if table_sources_used and any(kw in user_question.lower() for kw in ['table', 'data', 'number']):
                    # Verify table data is actually in the answer
                    answer_has_numbers = any(char.isdigit() for char in answer)
                    answer_has_structure = any(marker in answer for marker in ["|", "Row", "Column"])
                    
                    if answer_has_numbers or answer_has_structure:
                        st.success("✅ Answer appears to contain table data!")
                    else:
                        st.warning("⚠️ Table sources were retrieved but answer doesn't seem to contain table data")
        
        except Exception as response_e:
            st.error(f"Error getting response: {response_e}")
            answer = f"Error: {response_e}"
        
        # Enhanced source documents display
        if isinstance(response, dict) and 'source_documents' in response:
            with st.expander("📚 Sources Used in Answer", expanded=False):
                source_analysis = {
                    "table_sources": 0,
                    "page_sources": 0,
                    "overview_sources": 0,
                    "table_chunks": 0,
                    "complete_tables": 0,
                    "total_table_content": 0
                }
                
                for i, doc in enumerate(response['source_documents']):
                    doc_type = doc.metadata.get("type", "unknown")
                    content_type = doc.metadata.get("content_type", "unknown")
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source", "unknown")
                    
                    # Count source types
                    if doc_type == "table":
                        source_analysis["table_sources"] += 1
                    elif doc_type == "page":
                        source_analysis["page_sources"] += 1
                    elif doc_type == "overview":
                        source_analysis["overview_sources"] += 1
                    
                    if doc.metadata.get("table_chunk", False):
                        source_analysis["table_chunks"] += 1
                    if doc.metadata.get("is_complete_table", False):
                        source_analysis["complete_tables"] += 1
                    
                    # Count table content
                    if any(marker in doc.page_content for marker in ["===", "TABLE", "|", "Row"]):
                        source_analysis["total_table_content"] += 1
                    
                    st.write(f"**Source {i+1}:**")
                    st.write(f"- Type: **{doc_type}** ({content_type})")
                    st.write(f"- Page: {page_num} | File: {source}")
                    
                    # Enhanced content flags
                    content_flags = []
                    if doc.metadata.get("has_text", False):
                        content_flags.append("📝")
                    if doc.metadata.get("has_tables", False):
                        content_flags.append("📊")
                    if doc.metadata.get("has_images", False):
                        content_flags.append("🖼️")
                    if doc.metadata.get("table_chunk", False):
                        content_flags.append("🔢**TABLE**")
                    if doc.metadata.get("is_complete_table", False):
                        content_flags.append("📋**COMPLETE**")
                    
                    if content_flags:
                        st.write(f"- Contains: {' '.join(content_flags)}")
                    
                    # Show table indicators in content
                    table_markers = []
                    if "===" in doc.page_content:
                        table_markers.append("Delimiters")
                    if " | " in doc.page_content:
                        table_markers.append("Pipes")
                    if "TABLE" in doc.page_content.upper():
                        table_markers.append("Labels")
                    if "Row " in doc.page_content:
                        table_markers.append("Rows")
                    
                    if table_markers:
                        st.write(f"- Table markers: {', '.join(table_markers)}")
                    
                    # Show formatted content preview
                    content_preview = doc.page_content[:500]
                    if any(marker in doc.page_content for marker in ["===", "|", "TABLE"]):
                        st.code(content_preview + "...", language=None)
                    else:
                        st.write(f"- Content: {content_preview}...")
                    
                    st.write("---")
                
                # CRITICAL: Source analysis summary
                st.write("**📊 SOURCE ANALYSIS:**")
                
                total_sources = len(response['source_documents'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Sources", total_sources)
                    st.metric("Table Sources", source_analysis["table_sources"])
                    st.metric("Table Chunks", source_analysis["table_chunks"])
                
                with col2:
                    st.metric("Complete Tables", source_analysis["complete_tables"])
                    st.metric("Content w/ Table Markers", source_analysis["total_table_content"])
                    st.metric("Page Sources", source_analysis["page_sources"])
                
                # Quality assessment
                if source_analysis["table_chunks"] > 0 or source_analysis["total_table_content"] > 0:
                    st.success("✅ Table content was successfully retrieved and used!")
                elif any(kw in user_question.lower() for kw in ['table', 'data', 'number']):
                    st.error("❌ Table-related query but no table content in sources!")
                    st.write("**Troubleshooting suggestions:**")
                    st.write("- Check if your PDF actually contains tables")
                    st.write("- Try rephrasing your question")
                    st.write("- Verify table extraction worked during upload")
        
        # Update chat history
        st.session_state.chat_history.append((user_question, answer))
        
        # Display conversation with enhanced table formatting
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                # Check if answer contains table-like content
                if any(marker in a for marker in ["|", "===", "Row", "TABLE"]):
                    # Display as code to preserve table formatting
                    st.code(a, language=None)
                else:
                    st.markdown(a)

    def get_conversation_history(self):
        """Display conversation history with table formatting preservation"""
        with st.expander("💬 Conversation History"):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                
                # Preserve table formatting in history
                if any(marker in a for marker in ["|", "===", "Row", "TABLE"]):
                    st.code(a, language=None)
                else:
                    st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")

    def test_table_query(self, test_queries=None):
        """Test the system with known table queries"""
        if test_queries is None:
            test_queries = [
                "What tables are in this document?",
                "Show me any numerical data",
                "List all the data from tables",
                "What numbers can you find?"
            ]
        
        st.subheader("🧪 Table Query Testing")
        
        if 'conversation' not in st.session_state:
            st.error("❌ No conversation chain available. Please upload and process PDFs first.")
            return
        
        for i, query in enumerate(test_queries):
            if st.button(f"Test Query {i+1}: {query}"):
                with st.spinner(f"Testing: {query}"):
                    try:
                        response = st.session_state.conversation({
                            'question': query,
                            'chat_history': []
                        })
                        
                        answer = response.get('answer', 'No answer')
                        sources = response.get('source_documents', [])
                        
                        # Analyze response
                        table_sources = len([d for d in sources if d.metadata.get("table_chunk", False)])
                        has_table_markers = any(
                            any(marker in doc.page_content for marker in ["===", "TABLE", "|", "Row"])
                            for doc in sources
                        )
                        
                        st.write(f"**Test Results for:** {query}")
                        st.write(f"- Answer length: {len(answer)} characters")
                        st.write(f"- Sources used: {len(sources)}")
                        st.write(f"- Table sources: {table_sources}")
                        st.write(f"- Has table markers: {has_table_markers}")
                        
                        if table_sources > 0 or has_table_markers:
                            st.success("✅ Table content found in sources!")
                        else:
                            st.warning("⚠️ No table content in sources")
                        
                        st.write("**Answer:**")
                        if any(marker in answer for marker in ["|", "===", "Row", "TABLE"]):
                            st.code(answer, language=None)
                        else:
                            st.write(answer)
                        
                        st.write("---")
                        
                    except Exception as e:
                        st.error(f"Test failed: {e}")