from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class Chunks:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):  # Increased
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
            length_function=len
        )

    def get_text_chunks(self, text: str) -> list[Document]:
        raw_chunks = self.text_splitter.split_text(text)
        seen = set()
        cleaned_chunks = []
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) < 200:  # Reduced minimum
                continue
            if chunk not in seen:
                seen.add(chunk)
                cleaned_chunks.append(Document(page_content=chunk))
        
        return cleaned_chunks
