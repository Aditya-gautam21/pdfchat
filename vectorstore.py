import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder  # your custom embedding class

class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalHFEmbedder(model_path="D:/models/all-MiniLM-L6-v2")
        self.collection_name = "my_collection"

    def chroma_db_exists(self) -> bool:
        return os.path.exists(os.path.join(self.persist_directory, "chroma-collections.parquet"))

    def clear_vectorstore(self):
        if self.chroma_db_exists():
            try:
                # Try loading and deleting all references first
                vectorstore = self.fetch_vectorstore()
                del vectorstore  # Remove the object so SQLite releases the lock

                import gc
                gc.collect()  # Force garbage collection to close DB connections

                # Wait briefly to ensure file handles are released
                import time
                time.sleep(1)

                shutil.rmtree(self.persist_directory)
                print("🧹 Cleared existing vectorstore.")
            except Exception as e:
                print(f"❌ Error clearing vectorstore: {e}")
        else:
            print("ℹ️ No vectorstore to clear.")


    def create_vectorstore(self, documents: list[Document]):
    # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not documents:
            raise ValueError("No valid documents with content to index.")

        if self.chroma_db_exists():
            vectorstore = self.fetch_vectorstore()
            vectorstore.add_documents(documents)
        else:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            vectorstore.persist()
        return vectorstore

    
    def fetch_vectorstore(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )
