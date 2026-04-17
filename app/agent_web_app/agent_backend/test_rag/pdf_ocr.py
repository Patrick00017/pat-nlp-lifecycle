import fitz # PyMuPDF
import pymupdf
from langchain_core.documents import Document
# Open the PDF document
doc = pymupdf.open("usermanual.pdf")
# Select a page
for page in doc:
    # Perform OCR on the page
    text_page = Document(page_content=page.get_text()) # Specify the language (e.g., English)
    # text_page = page.get_textpage_ocr()
    # Extract text from the OCR result
    # text = text_page.extract_text()
    print(text_page)
    print(type(text_page))