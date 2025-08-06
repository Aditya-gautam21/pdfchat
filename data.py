# data.py - UPDATED FOR TABLE PRIORITY

import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import cv2
import json  # NEW: For structured table output

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
        """Enhanced image preprocessing for better OCR results"""
        try:
            if image.mode != 'L':
                image = image.convert('L')
            img_array = np.array(image)
            denoised = cv2.fastNlMeansDenoising(img_array)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            enhanced_image = Image.fromarray(processed)
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.2)
            return enhanced_image
        except Exception as e:
            print(f"Warning: Advanced image preprocessing failed: {e}")
            try:
                if image.mode != 'L':
                    image = image.convert('L')
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.5)
                return image
            except Exception as e2:
                print(f"Warning: Basic preprocessing also failed: {e2}")
                return image

    def format_table_for_llm(self, table_data, page_num, table_idx, extraction_method="Unknown"):
        """UPDATED: Format as queryable JSON-like structure for easy data retrieval"""
        if not table_data or len(table_data) == 0:
            return ""
        table_text = f"\n{'='*70}\n"
        table_text += f"TABLE {table_idx + 1} FROM PAGE {page_num + 1} (Method: {extraction_method})\n"
        table_text += f"{'='*70}\n\n"
        # Clean and process table data
        cleaned_table = []
        for row in table_data:
            if row:
                cleaned_row = [str(cell or "").strip().replace('\n', ' ').replace('\r', '') for cell in row]
                if any(cell.strip() for cell in cleaned_row):
                    cleaned_table.append(cleaned_row)
        if not cleaned_table:
            return ""
        # Ensure consistent columns
        max_cols = max(len(row) for row in cleaned_table)
        for row in cleaned_table:
            row.extend([""] * (max_cols - len(row)))
        # NEW: Create JSON-like structure for easy querying
        try:
            headers = cleaned_table[0]
            data_rows = cleaned_table[1:]
            structured_data = {
                "headers": headers,
                "rows": data_rows
            }
            table_text += "STRUCTURED_DATA_JSON:\n"
            table_text += json.dumps(structured_data, indent=2) + "\n"
        except Exception as e:
            table_text += f"Error creating JSON: {e}\n"
        # Keep existing formatted text
        table_text += "TABLE DATA:\n"
        table_text += f"HEADERS: {' | '.join(cleaned_table[0])}\n"
        table_text += "-" * 60 + "\n"
        for row_idx, row in enumerate(cleaned_table[1:], 1):
            table_text += f"DATA_ROW {row_idx:02d}: {' | '.join(row)}\n"
        table_text += "\nSTRUCTURED_DATA_START\n"
        for row_idx, row in enumerate(cleaned_table):
            if row_idx == 0:
                table_text += f"HEADERS: {' | '.join(row)}\n"
            else:
                table_text += f"DATA_{row_idx:02d}: {' | '.join(row)}\n"
        table_text += "STRUCTURED_DATA_END\n"
        table_text += f"\n{'='*70}\n"
        table_text += f"END TABLE {table_idx + 1}\n"
        table_text += f"{'='*70}\n\n"
        return table_text

    def validate_extracted_table(self, table, page_num):
        """UPDATED: Enhanced validation for real tables (high priority)"""
        if not table or len(table) < 2:  # At least 2 rows
            return False
        num_cols = len(table[0])
        if any(len(row) != num_cols for row in table[1:]):  # Consistent columns
            return False
        # Check for numeric data (high priority for data queries)
        has_numbers = any(any(char.isdigit() for char in str(cell)) for row in table for cell in row)
        return has_numbers or len(table) >= 3  # Prioritize tables with data

    def extract_tables_enhanced(self, pdf_path: str) -> dict[int, list[str]]:
        """UPDATED: Prioritize table extraction, skip image-like tables"""
        page_tables = {}
        filename = os.path.basename(pdf_path)
        print(f"🔍 Extracting tables (high priority) from {filename}...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables[page_num] = []
                    try:
                        # Method 1: PDFPlumber (primary)
                        page_table_data = page.extract_tables()
                        valid_tables = 0
                        for table_idx, table in enumerate(page_table_data or []):
                            if table and self.validate_extracted_table(table, page_num):
                                formatted_table = self.format_table_for_llm(table, page_num, table_idx, "PDFPlumber")
                                if formatted_table:
                                    page_tables[page_num].append(formatted_table)
                                    valid_tables += 1
                        if valid_tables > 0:
                            print(f"✅ Found {valid_tables} valid tables on page {page_num + 1}")
                            continue  # Skip other methods if successful
                        # Fallback: Text-based detection
                        page_text = page.extract_text() or ""
                        if self.detect_table_patterns(page_text):
                            formatted_table = self.extract_table_from_text(page_text, page_num)
                            if formatted_table:
                                page_tables[page_num].append(formatted_table)
                                print(f"✅ Fallback text table on page {page_num + 1}")
                        # Fallback: OCR if no text
                        if not page_text.strip():
                            pil_image = page.to_image(resolution=300).original
                            pil_image = self.preprocess_image_for_ocr(pil_image)
                            custom_config = r'--oem 3 --psm 6'
                            ocr_text = pytesseract.image_to_string(pil_image, config=custom_config)
                            if self.detect_table_patterns(ocr_text):
                                formatted_table = self.extract_table_from_ocr(ocr_text, page_num)
                                if formatted_table:
                                    page_tables[page_num].append(formatted_table)
                                    print(f"✅ Fallback OCR table on page {page_num + 1}")
                    except Exception as e:
                        print(f"⚠️ Error extracting tables from page {page_num + 1}: {e}")
        except Exception as e:
            print(f"❌ Error extracting tables from {pdf_path}: {e}")
        total_valid_tables = sum(len(tables) for tables in page_tables.values())
        print(f"📊 Total valid tables extracted: {total_valid_tables}")
        return page_tables

    def detect_table_patterns(self, text: str) -> bool:
        """Detect if text contains table-like patterns"""
        lines = text.split('\n')
        table_indicators = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) >= 3:  # At least 3 columns
                numbers = sum(1 for word in words if any(char.isdigit() for char in word))
                if numbers >= 2:
                    table_indicators += 1
                if '|' in line or '\t' in line or '  ' in line:
                    table_indicators += 1
                if any(pattern in line.lower() for pattern in ['total', 'amount', 'date', 'name', 'id', 'no.']):
                    table_indicators += 1
        return table_indicators >= 3

    def extract_table_from_text(self, text: str, page_num: int) -> str:
        """Extract table structure from plain text"""
        lines = text.split('\n')
        table_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) >= 2:
                numbers = sum(1 for word in words if any(char.isdigit() for char in word))
                if numbers >= 1 or len(words) >= 3:
                    table_lines.append(line)
        if len(table_lines) >= 2:
            table_text = f"\n{'='*70}\n"
            table_text += f"TABLE FROM PAGE {page_num + 1} (TEXT EXTRACTION)\n"
            table_text += f"{'='*70}\n\n"
            table_text += "EXTRACTED TABLE DATA:\n"
            for i, line in enumerate(table_lines):
                table_text += f"Row {i + 1:2d}: {line}\n"
            table_text += f"\n{'='*70}\n"
            table_text += f"END TABLE FROM PAGE {page_num + 1}\n"
            table_text += f"{'='*70}\n\n"
            return table_text
        return ""

    def extract_table_from_ocr(self, ocr_text: str, page_num: int) -> str:
        """Extract table structure from OCR text"""
        return self.extract_table_from_text(ocr_text, page_num).replace("(TEXT EXTRACTION)", "(OCR EXTRACTION)")

    def get_pdf_text_by_page(self, pdf_path: str) -> dict[int, str]:
        """Extract text from PDF page by page for better context preservation"""
        pages_text = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if not page_text.strip():
                            print(f"📄 Using OCR for page {page_num + 1}")
                            pil_image = page.to_image(resolution=300).original
                            pil_image = self.preprocess_image_for_ocr(pil_image)
                            custom_config = r'--oem 3 --psm 6'
                            ocr_text = pytesseract.image_to_string(pil_image, config=custom_config)
                            pages_text[page_num] = ocr_text.strip()
                        else:
                            pages_text[page_num] = page_text.strip()
                    except Exception as e:
                        print(f"⚠️ Error processing page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            print(f"❌ Error reading PDF {pdf_path}: {e}")
        return pages_text

    def extract_and_process_images_by_page(self, pdf_path: str, output_dir: str = "extracted_images") -> dict[int, list[str]]:
        """UPDATED: Skip OCR if table was already extracted on the page"""
        os.makedirs(output_dir, exist_ok=True)
        page_images = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_images[page_num] = []
                try:
                    images = page.get_images(full=True)
                    print(f"🖼️ Found {len(images)} images on page {page_num + 1}")
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            image_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{image_ext}")
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            # Process image with OCR (skip if table-heavy, but for now, always do)
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            if pil_image.width < 100 or pil_image.height < 100:
                                continue
                            pil_image = self.preprocess_image_for_ocr(pil_image)
                            image_config = r'--oem 3 --psm 6'
                            image_text = pytesseract.image_to_string(pil_image, config=image_config)
                            if image_text.strip():
                                formatted_text = f"\n{'='*60}\n"
                                formatted_text += f"IMAGE {img_index + 1} FROM PAGE {page_num + 1}\n"
                                formatted_text += f"{'='*60}\n"
                                formatted_text += f"IMAGE_TEXT_START\n"
                                formatted_text += f"{image_text.strip()}\n"
                                formatted_text += f"IMAGE_TEXT_END\n"
                                formatted_text += f"{'='*60}\n"
                                formatted_text += f"END IMAGE {img_index + 1}\n"
                                formatted_text += f"{'='*60}\n\n"
                                page_images[page_num].append(formatted_text)
                        except Exception as img_e:
                            print(f"⚠️ Error processing image {img_index + 1} on page {page_num + 1}: {img_e}")
                            continue
                except Exception as page_e:
                    print(f"⚠️ Error processing images on page {page_num + 1}: {page_e}")
                    continue
            doc.close()
        except Exception as e:
            print(f"❌ Error extracting images from {pdf_path}: {e}")
        return page_images

    def extract_all_content(self, pdf_path: str) -> tuple[list[Document], list[str]]:
        """UPDATED: High priority on tables, dedicated metadata"""
        documents = []
        filename = os.path.basename(pdf_path)
        print(f"📄 Processing {filename}...")
        try:
            pages_text = self.get_pdf_text_by_page(pdf_path)
            page_tables = self.extract_tables_enhanced(pdf_path)
            page_images = self.extract_and_process_images_by_page(pdf_path)
            # Strategy 1: Create dedicated table documents (CRITICAL for table queries)
            table_count = 0
            for page_num, tables in page_tables.items():
                for table_idx, table_content in enumerate(tables):
                    if table_content.strip():
                        table_count += 1
                        is_complete = len(table_content) < 4000 and "STRUCTURED_DATA_START" in table_content
                        documents.append(Document(
                            page_content=table_content,
                            metadata={
                                "source": filename,
                                "type": "table",
                                "content_type": "table_data",
                                "page_number": page_num + 1,
                                "table_index": table_idx + 1,
                                "table_chunk": True,
                                "has_tables": True,
                                "is_structured_data": True,
                                "is_complete_table": is_complete,
                                "contains_table_markers": True,
                                "priority": "high"  # NEW: For retrieval priority
                            }
                        ))
            print(f"✅ Created {table_count} dedicated table documents")
            # Strategy 2: Create image documents
            image_count = 0
            for page_num, images in page_images.items():
                for image_idx, image_content in enumerate(images):
                    if image_content.strip():
                        image_count += 1
                        documents.append(Document(
                            page_content=image_content,
                            metadata={
                                "source": filename,
                                "type": "image",
                                "content_type": "image_ocr",
                                "page_number": page_num + 1,
                                "image_index": image_idx + 1,
                                "image_chunk": True,
                                "has_images": True
                            }
                        ))
            print(f"✅ Created {image_count} image documents")
            # Strategy 3: Create comprehensive page documents
            max_page = max(
                max(pages_text.keys()) if pages_text else -1,
                max(page_tables.keys()) if page_tables else -1,
                max(page_images.keys()) if page_images else -1
            )
            for page_num in range(max_page + 1):
                text = pages_text.get(page_num, "")
                tables = page_tables.get(page_num, [])
                images = page_images.get(page_num, [])
                if text.strip() or tables or images:
                    page_content = f"{'='*70}\n"
                    page_content += f"PAGE {page_num + 1} FROM {filename}\n"
                    page_content += f"{'='*70}\n\n"
                    if text.strip():
                        page_content += f"PAGE_TEXT_START\n{text.strip()}\nPAGE_TEXT_END\n\n"
                    if tables:
                        page_content += f"TABLES_SUMMARY:\nThis page contains {len(tables)} table(s) with structured data.\nTable data is available in dedicated table documents.\n\n"
                    if images:
                        page_content += f"IMAGES_SUMMARY:\nThis page contains {len(images)} image(s) with extracted text.\nImage content is available in dedicated image documents.\n\n"
                    page_content += f"{'='*70}\n"
                    page_content += f"END PAGE {page_num + 1}\n"
                    page_content += f"{'='*70}\n"
                    documents.append(Document(
                        page_content=page_content,
                        metadata={
                            "source": filename,
                            "type": "page",
                            "content_type": "page_with_context",
                            "page_number": page_num + 1,
                            "has_text": bool(text.strip()),
                            "has_tables": bool(tables),
                            "has_images": bool(images),
                            "table_count": len(tables),
                            "image_count": len(images)
                        }
                    ))
            # Strategy 4: Create document overview
            total_pages = len(pages_text) if pages_text else 0
            total_tables = sum(len(tables) for tables in page_tables.values())
            total_images = sum(len(images) for images in page_images.values())
            if total_pages > 0:
                overview_content = f"{'='*70}\n"
                overview_content += f"DOCUMENT OVERVIEW: {filename}\n"
                overview_content += f"{'='*70}\n\n"
                overview_content += f"DOCUMENT_STATISTICS:\n- Total Pages: {total_pages}\n- Total Tables: {total_tables}\n- Total Images: {total_images}\n- Dedicated Table Documents: {table_count}\n- Image Documents: {image_count}\n\n"
                if pages_text:
                    first_page = list(pages_text.values())[0]
                    if first_page:
                        overview_content += f"SAMPLE_CONTENT (First Page):\n{first_page[:500]}...\n\n"
                if total_tables > 0:
                    overview_content += f"TABLE_CONTENT_AVAILABLE:\nThis document contains {total_tables} tables with structured data.\nUse table-specific queries to access this data.\n\n"
                if total_images > 0:
                    overview_content += f"IMAGE_CONTENT_AVAILABLE:\nThis document contains {total_images} images with extracted text.\nUse image-specific queries to access this content.\n\n"
                overview_content += f"{'='*70}\n"
                overview_content += f"END DOCUMENT OVERVIEW\n"
                overview_content += f"{'='*70}\n"
                documents.append(Document(
                    page_content=overview_content,
                    metadata={
                        "source": filename,
                        "type": "overview",
                        "content_type": "document_summary",
                        "total_pages": total_pages,
                        "total_tables": total_tables,
                        "total_images": total_images,
                        "table_documents": table_count,
                        "image_documents": image_count
                    }
                ))
            # Collect image paths
            image_paths = []
            if os.path.exists("extracted_images"):
                image_paths = [
                    os.path.join("extracted_images", f)
                    for f in os.listdir("extracted_images")
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
            print(f"✅ Total documents created: {len(documents)}")
            print(f" - Table documents: {table_count}")
            print(f" - Image documents: {image_count}")
            print(f" - Page documents: {len([d for d in documents if d.metadata.get('type') == 'page'])}")
            print(f" - Overview documents: {len([d for d in documents if d.metadata.get('type') == 'overview'])}")
            return documents, image_paths
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def get_content_summary(self, documents: list[Document]) -> str:
        """Generate a summary of extracted content for debugging"""
        summary = "Content Extraction Summary:\n" + "=" * 50 + "\n"
        page_docs = [d for d in documents if d.metadata.get("type") == "page"]
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        image_docs = [d for d in documents if d.metadata.get("type") == "image"]
        overview_docs = [d for d in documents if d.metadata.get("type") == "overview"]
        summary += f"📄 Page documents: {len(page_docs)}\n"
        summary += f"📊 Dedicated table documents: {len(table_docs)}\n"
        summary += f"🖼️ Image documents: {len(image_docs)}\n"
        summary += f"📋 Overview documents: {len(overview_docs)}\n"
        summary += f"📄 Total documents: {len(documents)}\n"
        pages_with_text = len([d for d in page_docs if d.metadata.get("has_text", False)])
        pages_with_tables = len([d for d in page_docs if d.metadata.get("has_tables", False)])
        pages_with_images = len([d for d in page_docs if d.metadata.get("has_images", False)])
        table_chunks = len([d for d in documents if d.metadata.get("table_chunk", False)])
        summary += f"📝 Pages with text: {pages_with_text}\n"
        summary += f"📊 Pages with tables: {pages_with_tables}\n"
        summary += f"🖼️ Pages with images: {pages_with_images}\n"
        summary += f"🔢 Table chunks: {table_chunks}\n"
        total_chars = sum(len(doc.page_content) for doc in documents)
        summary += f"📝 Total content size: {total_chars:,} characters\n"
        if documents:
            avg_size = total_chars // len(documents)
            summary += f"📏 Average document size: {avg_size:,} characters\n"
        return summary
