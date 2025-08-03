import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder
from chunks import Chunks

class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = LocalHFEmbedder(model_path=model_path)
        self.collection_name = "my_collection"
        # Match chunks.py configuration
        self.chunker = Chunks(chunk_size=1000, chunk_overlap=200)

    def chroma_db_exists(self) -> bool:
        """Check if ChromaDB persistent storage exists"""
        chroma_files = [
            "chroma-collections.parquet",
            "chroma-embeddings.parquet"
        ]
        return any(
            os.path.exists(os.path.join(self.persist_directory, file)) 
            for file in chroma_files
        )

    def clear_vectorstore(self):
        """Safely clear the vectorstore"""
        if self.chroma_db_exists():
            try:
                # Try to load and delete properly first
                try:
                    vectorstore = self.fetch_vectorstore()
                    del vectorstore
                except:
                    pass
                
                # Force cleanup
                import gc
                gc.collect()
                import time
                time.sleep(1)
                
                # Remove directory
                shutil.rmtree(self.persist_directory)
                print("🧹 Cleared existing vectorstore.")
            except Exception as e:
                print(f"❌ Error clearing vectorstore: {e}")
                raise e
        else:
            print("ℹ️ No vectorstore to clear.")

    def create_vectorstore(self, documents: list[Document]):
        """Create or append to existing vectorstore"""
        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        
        if not documents:
            raise ValueError("No valid documents with content to index.")

        all_chunks = []
        for doc in documents:
            # Special handling for different document types
            doc_type = doc.metadata.get("type", "unknown")
            
            if doc_type == "page":
                # For page documents with combined content, use larger chunks
                page_chunker = Chunks(chunk_size=1500, chunk_overlap=300)
                chunks = page_chunker.get_text_chunks(doc.page_content)
            elif doc_type == "overview":
                # For overview documents, use medium chunks
                overview_chunker = Chunks(chunk_size=1200, chunk_overlap=250)
                chunks = overview_chunker.get_text_chunks(doc.page_content)
            else:
                # Default chunking
                chunks = self.chunker.get_text_chunks(doc.page_content)
            
            # Preserve metadata and add chunk info
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata = doc.metadata.copy()
                chunk.metadata["chunk_index"] = len(all_chunks)
                chunk.metadata["chunk_within_doc"] = chunk_idx
                chunk.metadata["total_chunks_in_doc"] = len(chunks)
            
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks generated from documents.")

        print(f"📊 Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Debug info for different content types
        page_chunks = len([c for c in all_chunks if c.metadata.get("type") == "page"])
        overview_chunks = len([c for c in all_chunks if c.metadata.get("type") == "overview"])
        image_content_chunks = len([c for c in all_chunks if c.metadata.get("has_images", False)])
        
        print(f"📄 Page chunks: {page_chunks}")
        print(f"📋 Overview chunks: {overview_chunks}")
        print(f"🖼️ Chunks with image content: {image_content_chunks}")

        try:
            if self.chroma_db_exists():
                # Append to existing vectorstore
                print("📝 Appending to existing vectorstore...")
                vectorstore = self.fetch_vectorstore()
                vectorstore.add_documents(all_chunks)
                print(f"✅ Added {len(all_chunks)} new chunks to existing vectorstore")
            else:
                # Create new vectorstore
                print("🆕 Creating new vectorstore...")
                vectorstore = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                print(f"✅ Created new vectorstore with {len(all_chunks)} chunks")
            
            # Ensure persistence
            vectorstore.persist()
            print("💾 Vectorstore persisted to disk")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise e

    def fetch_vectorstore(self):
        """Fetch existing vectorstore from persistent storage"""
        try:
            if not self.chroma_db_exists():
                raise ValueError("No existing vectorstore found")
                
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test if it's working
            try:
                collection = vectorstore._collection
                doc_count = collection.count()
                print(f"📊 Loaded vectorstore with {doc_count} chunks")
            except:
                print("📊 Loaded vectorstore (count unavailable)")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ Error fetching vectorstore: {e}")
            raise e

    def get_vectorstore_stats(self):
        """Get statistics about the vectorstore"""
        if not self.chroma_db_exists():
            return {"exists": False}
        
        try:
            vectorstore = self.fetch_vectorstore()
            collection = vectorstore._collection
            
            # Get metadata statistics
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            stats = {
                "exists": True,
                "total_chunks": len(metadatas),
                "document_types": {},
                "sources": set(),
                "content_types": {}
            }
            
            for metadata in metadatas:
                # Count document types
                doc_type = metadata.get("type", "unknown")
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
                
                # Track sources
                source = metadata.get("source", "unknown")
                stats["sources"].add(source)
                
                # Count content types
                content_type = metadata.get("content_type", "unknown")
                stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            stats["sources"] = list(stats["sources"])
            return stats
            
        except Exception as e:
            return {"exists": True, "error": str(e)}
