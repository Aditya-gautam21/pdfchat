# chain.py - UPDATED FOR BETTER TABLE RETRIEVAL

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from ollama_llm import OllamaLLM
from prompt import Prompt
import re
import time
from typing import List, Any

class Chain:
    def get_conversation_chain(self, vectorstore):
        """Create enhanced conversation chain with table-optimized settings"""
        llm = OllamaLLM(
            model="mistral:latest",
            temperature=0.0,
            streaming=True
        )
        chat_history = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            chat_memory=chat_history,
            output_key="answer"
        )
        question_generator_prompt, qa_prompt, refine_prompt = Prompt.provide_enhanced_prompts()
        question_generator = LLMChain(llm=llm, prompt=question_generator_prompt)
        qa_chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=qa_prompt
        )
        enhanced_retriever = self.create_intelligent_retriever(vectorstore)
        conversation_chain = ConversationalRetrievalChain(
            retriever=enhanced_retriever,
            memory=memory,
            question_generator=question_generator,
            combine_docs_chain=qa_chain,
            return_source_documents=True,
            verbose=True,
            max_tokens_limit=6000
        )
        return conversation_chain

    def create_intelligent_retriever(self, vectorstore):
        """Create intelligent retriever that adapts to query type"""
        class IntelligentRetriever(BaseRetriever):
            class Config:
                arbitrary_types_allowed = True
                extra = "forbid"

            def __init__(self, vectorstore, **kwargs):
                super().__init__(**kwargs)
                object.__setattr__(self, 'vectorstore', vectorstore)
                object.__setattr__(self, 'base_retriever', vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 15}
                ))
                # Define query patterns
                object.__setattr__(self, 'table_patterns', [
                    r'\b(table|data|number|amount|total|sum|count|value|figure|statistic)\b',
                    r'\b(show|list|display|what|how many|how much)\b',
                    r'\b(row|column|cell|header|compare|comparison)\b',
                    r'\b(calculate|add up|find|extract)\b'
                ])
                object.__setattr__(self, 'image_patterns', [
                    r'\b(image|picture|chart|graph|diagram|figure|illustration)\b',
                    r'\b(visual|photo|drawing|sketch)\b'
                ])
                object.__setattr__(self, 'overview_patterns', [
                    r'\b(overview|summary|about|general|introduction)\b',
                    r'\b(what is|tell me about|describe|explain)\b'
                ])

            def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                return self.get_relevant_documents(query)

            def get_relevant_documents(self, query: str) -> List[Document]:
                query_lower = query.lower()
                analysis = self._analyze_query(query_lower)
                print(f"🔍 Query Analysis: {analysis}")
                all_docs = self.base_retriever.get_relevant_documents(query)
                if analysis['is_table_query']:
                    return self._handle_table_query(all_docs, query, analysis)
                elif analysis['is_image_query']:
                    return self._handle_image_query(all_docs, query, analysis)
                elif analysis['is_overview_query']:
                    return self._handle_overview_query(all_docs, query, analysis)
                else:
                    return self._handle_general_query(all_docs, query, analysis)

            def _analyze_query(self, query_lower: str) -> dict:
                analysis = {
                    'is_table_query': False,
                    'is_image_query': False,
                    'is_overview_query': False,
                    'is_specific_data_query': False,
                    'contains_numbers': False,
                    'table_keywords': [],
                    'image_keywords': [],
                    'overview_keywords': []
                }
                # Check for table patterns
                for pattern in self.table_patterns:
                    matches = re.findall(pattern, query_lower)
                    if matches:
                        analysis['table_keywords'].extend(matches)
                        analysis['is_table_query'] = True
                # Check for image patterns
                for pattern in self.image_patterns:
                    matches = re.findall(pattern, query_lower)
                    if matches:
                        analysis['image_keywords'].extend(matches)
                        analysis['is_image_query'] = True
                # Check for overview patterns
                for pattern in self.overview_patterns:
                    matches = re.findall(pattern, query_lower)
                    if matches:
                        analysis['overview_keywords'].extend(matches)
                        analysis['is_overview_query'] = True
                # Check for specific data queries
                specific_patterns = [r'\bpage \d+\b', r'\btable \d+\b', r'\brow \d+\b']
                analysis['is_specific_data_query'] = any(re.search(pattern, query_lower) for pattern in specific_patterns)
                # Check for numbers in query
                analysis['contains_numbers'] = bool(re.search(r'\d+', query_lower))
                return analysis

            def _handle_table_query(self, docs: List[Document], query: str, analysis: dict) -> List[Document]:
                """UPDATED: Boost high-priority and complete tables"""
                print(f"📊 Handling table query with {len(docs)} initial docs")
                # Prioritize by metadata
                high_priority = [d for d in docs if d.metadata.get("priority") == "high"]
                complete_tables = [d for d in docs if d.metadata.get("is_complete_table", False) and d not in high_priority]
                table_docs = [d for d in docs if d.metadata.get("table_chunk", False) and d not in high_priority and d not in complete_tables]
                page_with_tables = [d for d in docs if d.metadata.get("has_tables", False) or d.metadata.get("contains_table_markers", False)]
                other_docs = [d for d in docs if d not in high_priority and d not in complete_tables and d not in table_docs and d not in page_with_tables]
                print(f" - High priority tables: {len(high_priority)}")
                print(f" - Dedicated table docs: {len(table_docs)}")
                print(f" - Pages with tables: {len(page_with_tables)}")
                print(f" - Other docs: {len(other_docs)}")
                final_docs = high_priority[:5] + complete_tables[:4] + table_docs[:3] + page_with_tables[:2] + other_docs[:2]
                print(f"📋 Final table query selection: {len(final_docs)} documents (High priority: {len(high_priority)})")
                return final_docs[:12]

            def _handle_image_query(self, docs: List[Document], query: str, analysis: dict) -> List[Document]:
                print(f"🖼️ Handling image query with {len(docs)} initial docs")
                image_docs = []
                page_with_images = []
                other_docs = []
                for doc in docs:
                    metadata = doc.metadata
                    content = doc.page_content
                    if metadata.get("image_chunk", False) or metadata.get("type") == "image":
                        image_docs.append(doc)
                    elif metadata.get("has_images", False) or "IMAGE" in content:
                        page_with_images.append(doc)
                    else:
                        other_docs.append(doc)
                final_docs = image_docs[:6] + page_with_images[:4] + other_docs[:2]
                print(f"🖼️ Final image query selection: {len(final_docs)} documents")
                return final_docs[:12]

            def _handle_overview_query(self, docs: List[Document], query: str, analysis: dict) -> List[Document]:
                print(f"📋 Handling overview query with {len(docs)} initial docs")
                overview_docs = []
                page_docs = []
                other_docs = []
                for doc in docs:
                    metadata = doc.metadata
                    if metadata.get("type") == "overview":
                        overview_docs.append(doc)
                    elif metadata.get("type") == "page":
                        page_docs.append(doc)
                    else:
                        other_docs.append(doc)
                final_docs = overview_docs[:2] + page_docs[:8] + other_docs[:2]
                print(f"📋 Final overview query selection: {len(final_docs)} documents")
                return final_docs[:12]

            def _handle_general_query(self, docs: List[Document], query: str, analysis: dict) -> List[Document]:
                print(f"📄 Handling general query with {len(docs)} initial docs")
                table_docs = [d for d in docs if d.metadata.get("table_chunk", False)]
                image_docs = [d for d in docs if d.metadata.get("image_chunk", False)]
                page_docs = [d for d in docs if d.metadata.get("type") == "page"]
                overview_docs = [d for d in docs if d.metadata.get("type") == "overview"]
                final_docs = page_docs[:6] + table_docs[:3] + image_docs[:2] + overview_docs[:1]
                print(f"📄 Final general query selection: {len(final_docs)} documents")
                return final_docs[:12]

        return IntelligentRetriever(vectorstore)

    def test_chain_functionality(self, conversation_chain, test_queries=None):
        """Comprehensive testing of chain functionality"""
        if test_queries is None:
            test_queries = [
                "What tables are in this document?",
                "Show me numerical data from the tables",
                "What images are in this document?",
                "Give me an overview of this document",
                "Find specific data points"
            ]
        results = {}
        if not conversation_chain:
            return {"error": "No conversation chain provided"}
        for query in test_queries:
            try:
                print(f"🧪 Testing: {query}")
                response = conversation_chain({
                    'question': query,
                    'chat_history': []
                })
                if isinstance(response, dict):
                    answer = response.get('answer', 'No answer')
                    source_docs = response.get('source_documents', [])
                else:
                    answer = str(response)
                    source_docs = []
                results[query] = {
                    "success": True,
                    "answer_length": len(answer),
                    "source_count": len(source_docs),
                    "table_sources": len([d for d in source_docs if d.metadata.get("table_chunk", False)]),
                    "image_sources": len([d for d in source_docs if d.metadata.get("image_chunk", False)]),
                    "contains_structured_data": any(marker in answer for marker in ["STRUCTURED_DATA", "|", "ROW", "TABLE", "Headers:", "DATA_"]),
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                    "has_meaningful_content": len(answer.strip()) > 10 and "error" not in answer.lower()
                }
                print(f" ✅ Success: {results[query]['source_count']} sources, {results[query]['answer_length']} chars")
            except Exception as e:
                results[query] = {
                    "success": False,
                    "error": str(e),
                    "answer_length": 0,
                    "source_count": 0,
                    "table_sources": 0,
                    "image_sources": 0,
                    "contains_structured_data": False,
                    "answer_preview": "",
                    "has_meaningful_content": False
                }
                print(f" ❌ Failed: {e}")
        # Add summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get("success", False))
        results["_summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "timestamp": time.time()
        }
        return results

    def debug_retrieval(self, vectorstore, query: str, k: int = 10):
        """Debug retrieval process for a specific query"""
        print(f"🔍 Debug retrieval for: '{query}'")
        retriever = self.create_intelligent_retriever(vectorstore)
        docs = retriever.get_relevant_documents(query)
        print(f"\n📊 Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"\nDoc {i+1}:")
            print(f" Type: {metadata.get('type', 'unknown')}")
            print(f" Source: {metadata.get('source', 'unknown')}")
            print(f" Page: {metadata.get('page_number', 'unknown')}")
            print(f" Table chunk: {metadata.get('table_chunk', False)}")
            print(f" Image chunk: {metadata.get('image_chunk', False)}")
            print(f" Has tables: {metadata.get('has_tables', False)}")
            print(f" Has images: {metadata.get('has_images', False)}")
            print(f" Content: {content_preview}...")
            # Check for table indicators
            table_indicators = []
            if "===" in doc.page_content:
                table_indicators.append("Delimiters")
            if "|" in doc.page_content:
                table_indicators.append("Pipes")
            if "STRUCTURED_DATA" in doc.page_content:
                table_indicators.append("Structured")
            if "TABLE" in doc.page_content.upper():
                table_indicators.append("Labels")
            if table_indicators:
                print(f" Table indicators: {', '.join(table_indicators)}")
        return docs
