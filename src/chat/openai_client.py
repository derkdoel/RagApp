#openai_client.py

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import base64

class OpenAIClient:
    """A simple client for getting responses from OpenAI's chat models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", embedding_model="text-embedding-3-small"):
        # Load environment variables from .env file
        load_dotenv(find_dotenv())
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via OPENAI_API_KEY environment variable")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model
        
    def get_response(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Get a response from the model using a simple prompt.
        
        Args:
            prompt: The user's question or prompt
            system_prompt: Optional system prompt to set the AI's behavior
            
        Returns:
            The model's response as a string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error getting response from OpenAI: {str(e)}")
            
    def stream_response(self, prompt: str, system_prompt: str = "You are a helpful assistant."):
        """
        Stream a response from the model using a simple prompt.
        
        Args:
            prompt: The user's question or prompt
            system_prompt: Optional system prompt to set the AI's behavior
            
        Returns:
            A generator that yields response chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"Error streaming response from OpenAI: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
            
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def get_image_response(self, 
                        images: List[str], 
                        prompt: str, 
                        system_prompt: str = "You are an expert in analyzing business reports and financial figures.",
                        response_format: str = "text") -> str:
        """
        Get a response from the vision model based on images and a prompt.

        Args:
            images: List of image paths or base64-encoded images
            prompt: The specific question or instruction about the images
            system_prompt: Optional system prompt to set the AI's behavior
            response_format: Format of the response (text or json_object)
            
        Returns:
            The model's description of the images as a string
        """
        try:
            # Prepare the messages with images
            messages = [{"role": "system", "content": system_prompt}]
            
            # Create the user message with text and images
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
            
            # Add each image to the user message
            for img in images:
                # Check if the image is a file path or base64 data
                if os.path.exists(img):
                    # If it's a file path, read and encode it
                    with open(img, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                elif img.startswith(('data:', 'http')):
                    # Already a data URL or web URL
                    image_url = img
                else:
                    # Assume it's already base64 encoded
                    image_url = f"data:image/jpeg;base64,{img}"
                
                # Add the image to the message content
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            messages.append(user_message)
            
            # Set up response format if needed
            kwargs = {}
            if response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            
            # Call the OpenAI API
            completion = self.client.chat.completions.create(
                model="gpt-4-vision-preview",  # Use Vision model
                messages=messages,
                max_tokens=1000,
                **kwargs
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error getting image response from OpenAI: {str(e)}")


# Example usage:
if __name__ == "__main__":
    client = OpenAIClient()
    
    # Simple prompt with default system prompt
    response = client.get_response("Write a haiku about programming.")
    print(response)
    
    # Custom system prompt
    response = client.get_response(
        prompt="Write a haiku about programming.",
        system_prompt="You are a poet who specializes in technical topics."
    )
    print(response)
    
    # Streaming example
    print("\nStreaming response:")
    for chunk in client.stream_response("Tell me a 4 sentence story about AI."):
        print(chunk, end="", flush=True)

    # Embedding example
    texts = ["Hello, world!", "OpenAI is cool"]
    embeddings = client.get_embeddings(texts)
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Each embedding has dimension: {len(embeddings[0])}")