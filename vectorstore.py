import os
import shutil
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder

class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalHFEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        # Large chunks to avoid cutting context
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=8000,  # Increased for full pages
            chunk_overlap=1000,
            length_function=len
        )
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore = None  # Track vectorstore instance

    def chroma_db_exists(self) -> bool:
        return os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0

    def create_vectorstore(self, documents: list[Document]):
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        if not valid_documents:
            raise ValueError("No valid documents.")
        # Minimal chunking for full context
        chunks = self.chunker.split_documents(valid_documents)
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_directory)
        return self.vectorstore

    def fetch_vectorstore(self):
        if not self.chroma_db_exists():
            raise FileNotFoundError("No vectorstore.")
        self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return self.vectorstore

    def clear_vectorstore(self):
        # Clear any existing vectorstore instance
        self.vectorstore = None
        # Remove the directory if it exists
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory, ignore_errors=True)
            except Exception as e:
                print(f"Error clearing directory: {e}")
        # Recreate an empty directory to ensure clean state
        os.makedirs(self.persist_directory, exist_ok=True)
