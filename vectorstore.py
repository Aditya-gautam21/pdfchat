import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder

class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalHFEmbedder(model_path="D:/models/all-MiniLM-L6-v2")
        self.collection_name = "my_collection"

    def chroma_db_exists(self) -> bool:
        return os.path.exists(os.path.join(self.persist_directory, "chroma-collections.parquet"))

    def create_vectorstore(self, documents: list[Document]):
        if self.chroma_db_exists():
            print("✅ Existing vectorstore found. Loading...")
            return self.fetch_vectorstore()

        print("🛠️ Creating new vectorstore...")

        # Optional re-chunking (skip if already chunked)
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
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
