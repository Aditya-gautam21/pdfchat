import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from embeddings import LocalHFEmbedder
from langchain_community.embeddings import HuggingFaceEmbeddings


class Vectorstore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = LocalHFEmbedder(model_path="D:/models/all-MiniLM-L6-v2")

    def create_vectorstore(self, documents: list[str]):
        if os.path.exists(os.path.join(self.persist_directory, "index")):
            print("✅ Loading existing vectorstore...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding=self.embedding_model,
                collection_name=self.collection_name
            )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            docs = [Document(page_content=chunk.page_content) for chunk in documents]
            chunks = splitter.split_documents(docs)

            
            vectorstore = Chroma.from_documents(
                collection_name="my_collection",
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            vectorstore.add_texts(["example document 1", "example document 2"])
            vectorstore.persist()
        return vectorstore

    def fetch_vectorstore(self):
        
        return Chroma(
            collection_name="my_collection",
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )