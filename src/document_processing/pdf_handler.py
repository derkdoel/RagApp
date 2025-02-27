#pdf_handler.py

import pymupdf4llm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFHandler:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def extract_markdown(self):
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            logger.info(f"Processing PDF: {self.pdf_path}")

            markdown_text = pymupdf4llm.to_markdown(self.pdf_path)

            return markdown_text

        except Exception as e:
            raise Exception(f"Error splitting PDF: {str(e)}")
