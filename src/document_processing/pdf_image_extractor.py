import pymupdf  # PyMuPDF
import base64
import io
from PIL import Image
import os
import requests
import json
from typing import List, Dict, Tuple, Any, Optional

class ImageExtractorGPT:
    def __init__(self, openai_api_key: str, gpt_model: str = "gpt-4-vision-preview"):
        """
        Initialize the image extractor with OpenAI API key.
        
        Args:
            openai_api_key: Your OpenAI API key
            gpt_model: The GPT model to use (must support image inputs)
        """
        self.openai_api_key = openai_api_key
        self.gpt_model = gpt_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = "extracted_images") -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of dictionaries containing image info (path, page, position, etc.)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF
        doc = pymupdf.open(pdf_path)
        image_info_list = []
        
        # Loop through each page
        for page_num, page in enumerate(doc):
            # Get images from page
            image_list = page.get_images(full=True)
            
            # Process each image on the page
            for img_index, img in enumerate(image_list):
                # Get image details
                xref = img[0]  # XRef number
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image position on page
                rect = page.get_image_bbox(img)
                
                # Save the image
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store image info
                image_info = {
                    "path": image_path,
                    "page_num": page_num + 1,
                    "position": {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
                    "size": {"width": rect.width, "height": rect.height},
                    "extracted_text": None,
                    "description": None,
                    "surrounding_text": self._get_surrounding_text(page, rect)
                }
                
                image_info_list.append(image_info)
        
        return image_info_list
    
    def _get_surrounding_text(self, page, rect, margin: int = 50) -> str:
        """
        Get text surrounding an image to provide context.
        
        Args:
            page: PDF page object
            rect: Rectangle defining image position
            margin: Margin around image to consider for context
            
        Returns:
            Text surrounding the image
        """
        # Expand rectangle to get surrounding text
        expanded_rect = pymupdf.Rect(
            rect.x0 - margin, rect.y0 - margin, 
            rect.x1 + margin, rect.y1 + margin
        )
        
        # Get text within the expanded rectangle
        text = page.get_text("text", clip=expanded_rect)
        return text
    
    def process_image_with_gpt(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image with GPT to extract text and generate description.
        
        Args:
            image_info: Dictionary containing image information
            
        Returns:
            Updated image_info with extracted text and description
        """
        # Encode image to base64
        with open(image_info["path"], "rb") as img_file:
            img_data = img_file.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')
        
        # Create payload for GPT API
        payload = {
            "model": self.gpt_model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert in analyzing business reports, particularly financial figures and charts. 
                    For each image, provide:
                    1. Any visible text in the image (tables, labels, numbers, titles)
                    2. A detailed description of what the image shows (chart type, trends, key data points)
                    3. The financial implications or insights from this figure
                    4. Structure your response as JSON with keys: "extracted_text", "description", "financial_insights"
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"This image is from page {image_info['page_num']} of a KPN annual report. Surrounding text context: {image_info['surrounding_text'][:500]}..."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_img}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        # Call OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        # Parse response
        result = response.json()
        
        try:
            # Extract response content and parse JSON
            content = result["choices"][0]["message"]["content"]
            parsed_content = json.loads(content)
            
            # Update image info
            image_info["extracted_text"] = parsed_content.get("extracted_text", "")
            image_info["description"] = parsed_content.get("description", "")
            image_info["financial_insights"] = parsed_content.get("financial_insights", "")
            
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error processing image {image_info['path']}: {e}")
            image_info["error"] = str(e)
        
        return image_info
    
    def process_all_images(self, image_info_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all extracted images with GPT.
        
        Args:
            image_info_list: List of dictionaries containing image information
            
        Returns:
            Updated list with extracted text and descriptions
        """
        processed_images = []
        
        for image_info in image_info_list:
            processed_image = self.process_image_with_gpt(image_info)
            processed_images.append(processed_image)
            
        return processed_images
    
    def prepare_for_vector_db(self, image_info_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare processed image data for insertion into vector database.
        
        Args:
            image_info_list: List of processed image information
            
        Returns:
            List of documents ready for vector database insertion
        """
        vector_docs = []
        
        for img_info in image_info_list:
            # Create a comprehensive text representation for embedding
            combined_text = f"""
            FIGURE FROM PAGE {img_info['page_num']}
            
            EXTRACTED TEXT:
            {img_info['extracted_text']}
            
            DESCRIPTION:
            {img_info['description']}
            
            FINANCIAL INSIGHTS:
            {img_info.get('financial_insights', '')}
            
            SURROUNDING CONTEXT:
            {img_info['surrounding_text']}
            """
            
            # Create document for vector DB
            vector_doc = {
                "id": f"image_{img_info['page_num']}_{os.path.basename(img_info['path'])}",
                "text": combined_text.strip(),
                "metadata": {
                    "source": "image",
                    "page_num": img_info['page_num'],
                    "image_path": img_info['path'],
                    "position": img_info['position']
                }
            }
            
            vector_docs.append(vector_doc)
            
        return vector_docs