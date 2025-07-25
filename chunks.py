from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class Chunks:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],  # Prioritize natural splits
            length_function=len
        )

    def get_text_chunks(self, text: str) -> list[Document]:
        # Split into text chunks
        text_chunks = self.text_splitter.split_text(text)

        # Remove duplicates and filter too-short chunks
        seen = set()
        cleaned_chunks = []
        for chunk in text_chunks:
            chunk = chunk.strip()
            if len(chunk) < 100:  # Filter out very small fragments
                continue
            if chunk not in seen:
                seen.add(chunk)
                cleaned_chunks.append(Document(page_content=chunk))

        return cleaned_chunks
