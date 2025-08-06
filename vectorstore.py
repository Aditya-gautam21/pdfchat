# vectorstore.py - UPDATED FOR RELIABLE CLEARING

import os
import shutil
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder
import time
import hashlib
import gc
import uuid
import subprocess

class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = LocalHFEmbedder(model_path=model_path)
        self.collection_name = "pdf_documents"
        # Improved chunking strategies for different content types
        self.text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "!", "?", ";", " "],
            length_function=len
        )
        self.table_chunker = RecursiveCharacterTextSplitter(
            chunk_size=3000,  # Larger chunks for tables to preserve structure
            chunk_overlap=500,
            separators=["\n\n", "\n", "===", "---"],
            length_function=len
        )
        os.makedirs(self.persist_directory, exist_ok=True)

    def chroma_db_exists(self) -> bool:
        """Check if ChromaDB persistent storage exists"""
        if not os.path.exists(self.persist_directory):
            return False
        try:
            dir_contents = os.listdir(self.persist_directory)
            has_db_files = any(f.endswith('.sqlite3') or f.endswith('.parquet') or 'chroma' in f.lower() for f in dir_contents)
            return has_db_files and len(dir_contents) > 0
        except Exception as e:
            print(f"Error checking ChromaDB directory: {e}")
            return False

    def create_vectorstore(self, documents: list[Document]):
        """Create vectorstore with improved chunking for tables and images"""
        valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not valid_documents:
            raise ValueError("No valid documents with content to index.")
        print(f"🔄 Processing {len(valid_documents)} documents...")
        all_chunks = []
        chunk_stats = {
            "table_chunks": 0,
            "text_chunks": 0,
            "image_chunks": 0,
            "page_chunks": 0,
            "overview_chunks": 0
        }
        for doc_idx, doc in enumerate(valid_documents):
            doc_type = doc.metadata.get("type", "unknown")
            content_type = doc.metadata.get("content_type", "unknown")
            has_tables = doc.metadata.get("has_tables", False)
            has_images = doc.metadata.get("has_images", False)
            is_table = doc.metadata.get("table_chunk", False) or doc_type == "table"
            print(f"Processing document {doc_idx + 1}: {doc_type} (content: {content_type})")
            # Strategy 1: Handle dedicated table documents
            if is_table or doc_type == "table":
                if len(doc.page_content) <= 4000:
                    chunk = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_index": len(all_chunks),
                            "is_complete_table": True,
                            "table_chunk": True,
                            "chunk_type": "complete_table",
                            "searchable_content": self._extract_searchable_table_content(doc.page_content)
                        }
                    )
                    all_chunks.append(chunk)
                    chunk_stats["table_chunks"] += 1
                else:
                    table_chunks = self._split_table_carefully(doc)
                    for i, chunk in enumerate(table_chunks):
                        chunk.metadata.update({
                            "chunk_index": len(all_chunks) + i,
                            "table_chunk": True,
                            "chunk_within_table": i,
                            "total_table_chunks": len(table_chunks),
                            "chunk_type": "table_segment"
                        })
                    all_chunks.extend(table_chunks)
                    chunk_stats["table_chunks"] += len(table_chunks)
            # Strategy 2: Handle page documents with special content
            elif doc_type == "page":
                if has_tables and has_images:
                    chunks = self._create_multimodal_chunks(doc)
                    chunk_stats["page_chunks"] += len(chunks)
                elif has_tables:
                    chunks = self._create_table_aware_chunks(doc)
                    chunk_stats["page_chunks"] += len(chunks)
                elif has_images:
                    chunks = self._create_image_aware_chunks(doc)
                    chunk_stats["image_chunks"] += len(chunks)
                else:
                    chunks = self.text_chunker.split_documents([doc])
                    chunk_stats["text_chunks"] += len(chunks)
                for i, chunk in enumerate(chunks):
                    if not hasattr(chunk, 'metadata'):
                        chunk.metadata = doc.metadata.copy()
                    chunk.metadata.update({
                        "chunk_index": len(all_chunks) + i,
                        "chunk_within_page": i,
                        "total_page_chunks": len(chunks),
                        "chunk_type": "page_content"
                    })
                all_chunks.extend(chunks)
            # Strategy 3: Handle overview documents
            elif doc_type == "overview":
                chunk = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_index": len(all_chunks),
                        "chunk_type": "overview",
                        "is_overview": True
                    }
                )
                all_chunks.append(chunk)
                chunk_stats["overview_chunks"] += 1
            else:
                chunks = self.text_chunker.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata = doc.metadata.copy()
                    chunk.metadata.update({
                        "chunk_index": len(all_chunks) + i,
                        "chunk_type": "general"
                    })
                all_chunks.extend(chunks)
                chunk_stats["text_chunks"] += len(chunks)
        print(f"📊 Chunk Statistics:")
        for stat_name, count in chunk_stats.items():
            print(f" - {stat_name}: {count}")
        print(f" - Total chunks: {len(all_chunks)}")
        # Create or update vectorstore
        try:
            if self.chroma_db_exists():
                print("📂 Updating existing vectorstore...")
                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                vectorstore.add_documents(all_chunks)
            else:
                print("🆕 Creating new vectorstore...")
                vectorstore = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
            print("💾 Vectorstore created/updated successfully")
            try:
                test_results = vectorstore.similarity_search("test", k=1)
                print(f"✅ Vectorstore verification: {len(test_results)} documents accessible")
            except Exception as verify_e:
                print(f"⚠️ Vectorstore verification failed: {verify_e}")
            return vectorstore
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise

    def _extract_searchable_table_content(self, table_content: str) -> str:
        """Extract searchable keywords from table content"""
        keywords = []
        lines = table_content.split('\n')
        for line in lines:
            if 'HEADERS:' in line or 'DATA_' in line:
                cleaned = line.replace('HEADERS:', '').replace('DATA_', '').replace('|', ' ')
                words = [w.strip() for w in cleaned.split() if w.strip() and len(w) > 2]
                keywords.extend(words)
        return ' '.join(set(keywords[:20]))  # Unique and limited

    def _split_table_carefully(self, doc: Document) -> list[Document]:
        """Split table documents while preserving structure"""
        content = doc.page_content
        chunks = []
        table_sections = content.split('===')
        if len(table_sections) > 1:
            for i, section in enumerate(table_sections):
                if len(section.strip()) > 100:
                    chunk = Document(
                        page_content=section.strip(),
                        metadata={**doc.metadata, "table_section": i, "table_chunk": True}
                    )
                    chunks.append(chunk)
        else:
            table_chunks = self.table_chunker.split_documents([doc])
            for chunk in table_chunks:
                chunk.metadata.update(doc.metadata)
                chunk.metadata["table_chunk"] = True
            chunks = table_chunks
        return chunks

    def _create_multimodal_chunks(self, doc: Document) -> list[Document]:
        """Create chunks for pages with both tables and images"""
        content = doc.page_content
        chunks = []
        sections = []
        current_section = ""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('===') and ('TABLE' in line or 'IMAGE' in line):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        if current_section.strip():
            sections.append(current_section.strip())
        for i, section in enumerate(sections):
            if len(section) > 100:
                is_table_section = 'TABLE' in section and ('DATA_' in section or '|' in section)
                is_image_section = 'IMAGE' in section
                chunk = Document(
                    page_content=section,
                    metadata={
                        **doc.metadata,
                        "section_index": i,
                        "is_table_section": is_table_section,
                        "is_image_section": is_image_section,
                        "multimodal_chunk": True
                    }
                )
                if is_table_section:
                    chunk.metadata["table_chunk"] = True
                    chunk.metadata["contains_table_markers"] = True
                chunks.append(chunk)
        return chunks if chunks else self.text_chunker.split_documents([doc])

    def _create_table_aware_chunks(self, doc: Document) -> list[Document]:
        """Create chunks that preserve table structure"""
        table_aware_chunker = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=400,
            separators=["\n===", "\n\n", "\n"],
            length_function=len
        )
        chunks = table_aware_chunker.split_documents([doc])
        for chunk in chunks:
            chunk.metadata = doc.metadata.copy()
            if any(marker in chunk.page_content for marker in ["===", "|", "DATA_", "HEADERS:"]):
                chunk.metadata["contains_table_markers"] = True
                chunk.metadata["table_chunk"] = True
        return chunks

    def _create_image_aware_chunks(self, doc: Document) -> list[Document]:
        """Create chunks that preserve image context"""
        image_aware_chunker = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n===", "\n\n", "\n"],
            length_function=len
        )
        chunks = image_aware_chunker.split_documents([doc])
        for chunk in chunks:
            chunk.metadata = doc.metadata.copy()
            if "IMAGE_TEXT_START" in chunk.page_content:
                chunk.metadata["contains_image_content"] = True
                chunk.metadata["image_chunk"] = True
        return chunks

    def fetch_vectorstore(self):
        """Fetch existing vectorstore with proper error handling"""
        if not self.chroma_db_exists():
            raise FileNotFoundError("No vectorstore found in directory")
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            test_docs = vectorstore.similarity_search("test", k=1)
            print(f"✅ Vectorstore loaded with {len(test_docs)} test results")
            return vectorstore
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            print("🔄 Attempting to clear corrupted vectorstore...")
            self.clear_vectorstore()
            raise ValueError(f"Vectorstore was corrupted and has been cleared: {e}")

    def clear_vectorstore(self):
        """UPDATED: Enhanced clearing with retries and force deletion"""
        try:
            if os.path.exists(self.persist_directory):
                print("🧹 Clearing existing vectorstore...")
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        shutil.rmtree(self.persist_directory)
                        print(f"✅ Cleared on attempt {attempt + 1}")
                        break
                    except PermissionError as e:
                        print(f"⚠️ Permission error (attempt {attempt + 1}): {e}")
                        time.sleep(1)  # Wait and retry
                        if os.name == 'nt':
                            self._force_delete_windows(self.persist_directory)
                    except Exception as e:
                        print(f"❌ Error on attempt {attempt + 1}: {e}")
                else:
                    raise RuntimeError("Failed to clear after 3 attempts")
                os.makedirs(self.persist_directory, exist_ok=True)
                print("📁 Recreated empty directory")
            else:
                print("📂 No directory to clear")
            
            # NEW: Force garbage collection and verify no lingering data
            gc.collect()
            if self.verify_cleared():
                print("✅ Verified: Directory is completely empty")
            else:
                print("⚠️ Warning: Some files may still remain - manual deletion recommended")
        except Exception as e:
            print(f"❌ Final error clearing: {e}")
            self._emergency_clear()

        def _force_delete_windows(self, directory):
            """NEW: Windows-specific force delete"""
            try:
                subprocess.run(['rmdir', '/s', '/q', directory], shell=True, check=True)
                print("✅ Windows force delete successful")
            except subprocess.CalledProcessError as e:
                print(f"❌ Windows force failed: {e}")

    def _emergency_clear(self):
        """Emergency clear - rename directory and create new one"""
        try:
            if os.path.exists(self.persist_directory):
                backup_name = f"{self.persist_directory}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(self.persist_directory, backup_name)
                print(f"⚠️ Renamed problematic directory to {backup_name}")
                os.makedirs(self.persist_directory, exist_ok=True)
                print("✅ Emergency clear completed - created fresh directory")
        except Exception as e:
            print(f"❌ Emergency clear failed: {e}")

    def verify_cleared(self):
        """Verify that the vectorstore was actually cleared"""
        try:
            if not os.path.exists(self.persist_directory):
                return True
            for root, dirs, files in os.walk(self.persist_directory):
                if files:
                    return False
            return True
        except Exception as e:
            print(f"Error verifying clear: {e}")
            return False

    def get_vectorstore_stats(self):
        """Get detailed statistics about the vectorstore"""
        if not self.chroma_db_exists():
            return {"exists": False}
        try:
            vectorstore = self.fetch_vectorstore()
            all_docs = vectorstore.similarity_search("", k=1000)  # Get many docs
            stats = {
                "exists": True,
                "total_chunks": len(all_docs),
                "table_chunks": len([d for d in all_docs if d.metadata.get("table_chunk", False)]),
                "image_chunks": len([d for d in all_docs if d.metadata.get("image_chunk", False)]),
                "page_chunks": len([d for d in all_docs if d.metadata.get("type") == "page"]),
                "overview_chunks": len([d for d in all_docs if d.metadata.get("type") == "overview"]),
                "complete_tables": len([d for d in all_docs if d.metadata.get("is_complete_table", False)]),
                "docs_with_table_markers": len([d for d in all_docs if d.metadata.get("contains_table_markers", False)])
            }
            return stats
        except Exception as e:
            return {"exists": True, "error": str(e)}
