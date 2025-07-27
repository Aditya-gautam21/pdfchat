import fitz  
import pdfplumber
import os

class Data:
    def get_pdf_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_tables(self, pdf_path: str) -> str:
        tables_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    table_text = f"\n### Table from Page {page_num+1}:\n"
                    for row in table:
                        row_text = " | ".join(str(cell or "") for cell in row)
                        table_text += f"{row_text}\n"
                    tables_text += table_text
        return tables_text

    def extract_images(self, pdf_path: str, output_dir: str = "extracted_images") -> list[str]:
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_paths = []

        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(image_path)

        return image_paths


    def extract_all_content(self, pdf_path: str):
        text = self.get_pdf_text(pdf_path)
        tables = self.extract_tables(pdf_path)
        image_paths = self.extract_images(pdf_path)

        combined = f"## Extracted Text\n{text}\n\n## Extracted Tables\n{tables}"
        return combined, image_paths

