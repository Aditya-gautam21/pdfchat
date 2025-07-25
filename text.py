import fitz  

class Text:
    def get_pdf_text(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""