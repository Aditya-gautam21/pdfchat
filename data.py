import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np


class Data:
    def __init__(self):
        # Set Tesseract path for Windows if needed
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME'))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def preprocess_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter())
            
            return image
        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            return image

    def get_pdf_text_by_page(self, pdf_path: str) -> dict[int, str]:
        """Extract text from PDF page by page for better context preservation"""
        pages_text = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Try extracting text normally first
                        page_text = page.extract_text() or ""
                        
                        if not page_text.strip():
                            # If no text found, use OCR on the page image
                            print(f"Using OCR for page {page_num + 1}")
                            pil_image = page.to_image(resolution=300).original
                            
                            # Preprocess image for better OCR
                            pil_image = self.preprocess_image_for_ocr(pil_image)
                            
                            # Use custom OCR config for better results
                            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?:;-()[]{}/"'
                            ocr_text = pytesseract.image_to_string(pil_image, config=custom_config)
                            pages_text[page_num] = ocr_text.strip()
                        else:
                            pages_text[page_num] = page_text.strip()
                            
                    except Exception as e:
                        print(f"Error processing page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            
        return pages_text

    def get_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF with OCR fallback for scanned pages"""
        pages_text = self.get_pdf_text_by_page(pdf_path)
        combined_text = ""
        
        for page_num, text in pages_text.items():
            if text:
                combined_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
        
        return combined_text

    def extract_tables_by_page(self, pdf_path: str) -> dict[int, list[str]]:
        """Extract tables from PDF page by page for better context"""
        page_tables = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables[page_num] = []
                    try:
                        page_table_data = page.extract_tables()
                        
                        if page_table_data:
                            # Process structured tables
                            for table_idx, table in enumerate(page_table_data):
                                if table and len(table) > 0:
                                    table_text = f"\n=== TABLE {table_idx + 1} FROM PAGE {page_num + 1} ===\n"
                                    
                                    # Convert table to markdown-like format
                                    for row_idx, row in enumerate(table):
                                        if row:  # Skip empty rows
                                            # Clean and join cells
                                            cleaned_row = []
                                            for cell in row:
                                                cell_text = str(cell or "").strip().replace('\n', ' ')
                                                cleaned_row.append(cell_text)
                                            
                                            row_text = " | ".join(cleaned_row)
                                            table_text += f"{row_text}\n"
                                            
                                            # Add separator after header row
                                            if row_idx == 0:
                                                separator = " | ".join(["---"] * len(cleaned_row))
                                                table_text += f"{separator}\n"
                                    
                                    table_text += f"=== END TABLE {table_idx + 1} ===\n"
                                    page_tables[page_num].append(table_text)
                        
                        else:
                            # Try OCR-based table extraction for scanned pages
                            try:
                                pil_image = page.to_image(resolution=300).original
                                pil_image = self.preprocess_image_for_ocr(pil_image)
                                
                                # Use table-specific OCR configuration
                                table_config = r'--oem 3 --psm 6'
                                ocr_text = pytesseract.image_to_string(pil_image, config=table_config)
                                
                                # Simple heuristic to detect table-like content
                                lines = ocr_text.split('\n')
                                table_lines = []
                                
                                for line in lines:
                                    line = line.strip()
                                    # Look for lines with multiple words/numbers separated by spaces
                                    if line and len(line.split()) >= 2:
                                        # Check if line might be tabular (contains numbers or multiple segments)
                                        if any(char.isdigit() for char in line) or len(line.split()) >= 3:
                                            table_lines.append(line)
                                
                                if len(table_lines) >= 2:  # At least 2 rows to consider it a table
                                    table_text = f"\n=== TABLE FROM PAGE {page_num + 1} (OCR) ===\n"
                                    for line in table_lines:
                                        table_text += f"{line}\n"
                                    table_text += f"=== END TABLE (OCR) ===\n"
                                    page_tables[page_num].append(table_text)
                                    
                            except Exception as ocr_e:
                                print(f"OCR table extraction failed for page {page_num + 1}: {ocr_e}")
                                
                    except Exception as e:
                        print(f"Error extracting tables from page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            
        return page_tables

    def extract_tables(self, pdf_path: str) -> list[str]:
        """Extract tables from PDF with OCR fallback - maintains backward compatibility"""
        page_tables = self.extract_tables_by_page(pdf_path)
        all_tables = []
        
        for page_num, tables in page_tables.items():
            all_tables.extend(tables)
            
        return all_tables

    def extract_and_process_images_by_page(self, pdf_path: str, output_dir: str = "extracted_images") -> dict[int, list[str]]:
        """Extract images by page and convert them to text using OCR"""
        os.makedirs(output_dir, exist_ok=True)
        page_images = {}
        
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_images[page_num] = []
                try:
                    images = page.get_images(full=True)
                    
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Save image file
                            image_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{image_ext}")
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            
                            # Process image with OCR
                            try:
                                pil_image = Image.open(io.BytesIO(image_bytes))
                                
                                # Skip very small images (likely decorative)
                                if pil_image.width < 100 or pil_image.height < 100:
                                    continue
                                
                                # Preprocess for better OCR
                                pil_image = self.preprocess_image_for_ocr(pil_image)
                                
                                # Extract text from image
                                image_config = r'--oem 3 --psm 6'
                                image_text = pytesseract.image_to_string(pil_image, config=image_config)
                                
                                if image_text.strip():
                                    formatted_text = f"\n=== IMAGE {img_index + 1} FROM PAGE {page_num + 1} ===\n"
                                    formatted_text += f"Image file: {image_path}\n"
                                    formatted_text += f"Content: {image_text.strip()}\n"
                                    formatted_text += f"=== END IMAGE {img_index + 1} ===\n"
                                    page_images[page_num].append(formatted_text)
                                    
                            except Exception as ocr_e:
                                print(f"OCR failed for image {img_index + 1} on page {page_num + 1}: {ocr_e}")
                                
                        except Exception as img_e:
                            print(f"Error processing image {img_index + 1} on page {page_num + 1}: {img_e}")
                            continue
                            
                except Exception as page_e:
                    print(f"Error processing images on page {page_num + 1}: {page_e}")
                    continue
                    
            doc.close()
            
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {e}")
            
        return page_images

    def extract_and_process_images(self, pdf_path: str, output_dir: str = "extracted_images") -> list[str]:
        """Extract images and convert them to text using OCR - maintains backward compatibility"""
        page_images = self.extract_and_process_images_by_page(pdf_path, output_dir)
        all_images = []
        
        for page_num, images in page_images.items():
            all_images.extend(images)
            
        return all_images

    def create_combined_page_content(self, page_num: int, text: str, tables: list[str], images: list[str]) -> str:
        """Combine all content from a page into a single coherent document"""
        combined_content = f"=== PAGE {page_num + 1} CONTENT ===\n\n"
        
        # Add main text
        if text and text.strip():
            combined_content += f"TEXT CONTENT:\n{text.strip()}\n\n"
        
        # Add tables inline
        if tables:
            combined_content += "TABLES:\n"
            for table in tables:
                combined_content += f"{table}\n"
            combined_content += "\n"
        
        # Add image text inline
        if images:
            combined_content += "IMAGES WITH TEXT:\n"
            for image in images:
                combined_content += f"{image}\n"
            combined_content += "\n"
        
        combined_content += f"=== END PAGE {page_num + 1} ===\n"
        
        return combined_content

    def extract_all_content(self, pdf_path: str) -> tuple[list[Document], list[str]]:
        """Extract all content from PDF with improved context preservation"""
        documents = []
        filename = os.path.basename(pdf_path)
        
        print(f"Processing {filename}...")
        
        try:
            # Extract content by page for better context
            pages_text = self.get_pdf_text_by_page(pdf_path)
            page_tables = self.extract_tables_by_page(pdf_path)
            page_images = self.extract_and_process_images_by_page(pdf_path)
            
            # Strategy 1: Create combined page documents (maintains context)
            for page_num in range(max(
                max(pages_text.keys()) if pages_text else -1,
                max(page_tables.keys()) if page_tables else -1,
                max(page_images.keys()) if page_images else -1
            ) + 1):
                
                text = pages_text.get(page_num, "")
                tables = page_tables.get(page_num, [])
                images = page_images.get(page_num, [])
                
                # Only create document if there's actual content
                if text.strip() or tables or images:
                    combined_content = self.create_combined_page_content(page_num, text, tables, images)
                    
                    # Only add if there's substantial content
                    if len(combined_content.strip()) > 50:
                        documents.append(Document(
                            page_content=combined_content,
                            metadata={
                                "source": filename,
                                "type": "page",
                                "page_number": page_num + 1,
                                "content_type": "combined_page",
                                "has_text": bool(text.strip()),
                                "has_tables": bool(tables),
                                "has_images": bool(images)
                            }
                        ))
            
            # Strategy 2: Also create full document overview for global questions
            all_text = ""
            all_tables = []
            all_images = []
            
            for page_text in pages_text.values():
                if page_text.strip():
                    all_text += page_text + "\n\n"
            
            for tables in page_tables.values():
                all_tables.extend(tables)
            
            for images in page_images.values():
                all_images.extend(images)
            
            # Create document summary if content exists
            if all_text.strip() or all_tables or all_images:
                summary_content = f"=== DOCUMENT OVERVIEW: {filename} ===\n\n"
                
                if all_text.strip():
                    # Truncate very long text for overview
                    text_preview = all_text[:2000] + "..." if len(all_text) > 2000 else all_text
                    summary_content += f"MAIN CONTENT PREVIEW:\n{text_preview}\n\n"
                
                if all_tables:
                    summary_content += f"DOCUMENT CONTAINS {len(all_tables)} TABLES\n\n"
                
                if all_images:
                    summary_content += f"DOCUMENT CONTAINS {len(all_images)} IMAGES WITH TEXT\n\n"
                
                summary_content += f"=== END DOCUMENT OVERVIEW ===\n"
                
                documents.append(Document(
                    page_content=summary_content,
                    metadata={
                        "source": filename,
                        "type": "overview",
                        "content_type": "document_summary",
                        "total_pages": len(pages_text),
                        "total_tables": len(all_tables),
                        "total_images": len(all_images)
                    }
                ))

            # Collect image paths for reference
            image_paths = []
            if os.path.exists("extracted_images"):
                image_paths = [
                    os.path.join("extracted_images", f) 
                    for f in os.listdir("extracted_images") 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

            print(f"✅ Created {len(documents)} contextual documents")
            print(f"✅ Pages processed: {len(pages_text)}")
            print(f"✅ Tables extracted: {sum(len(tables) for tables in page_tables.values())}")
            print(f"✅ Images processed: {sum(len(images) for images in page_images.values())}")
            
            return documents, image_paths
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return [], []

    def get_content_summary(self, documents: list[Document]) -> str:
        """Generate a summary of extracted content for debugging"""
        summary = "Content Extraction Summary:\n"
        summary += "=" * 50 + "\n"
        
        page_docs = [d for d in documents if d.metadata.get("type") == "page"]
        overview_docs = [d for d in documents if d.metadata.get("type") == "overview"]
        
        summary += f"📄 Page documents: {len(page_docs)}\n"
        summary += f"📋 Overview documents: {len(overview_docs)}\n"
        summary += f"📄 Total documents: {len(documents)}\n"
        
        # Count content types across all pages
        pages_with_text = len([d for d in page_docs if d.metadata.get("has_text", False)])
        pages_with_tables = len([d for d in page_docs if d.metadata.get("has_tables", False)])
        pages_with_images = len([d for d in page_docs if d.metadata.get("has_images", False)])
        
        summary += f"📝 Pages with text: {pages_with_text}\n"
        summary += f"📊 Pages with tables: {pages_with_tables}\n"
        summary += f"🖼️ Pages with images: {pages_with_images}\n"
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        summary += f"📝 Total content size: {total_chars:,} characters\n"
        
        # Average document size
        if documents:
            avg_size = total_chars // len(documents)
            summary += f"📏 Average document size: {avg_size:,} characters\n"
        
        return summary
