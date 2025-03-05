import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import MarkdownTextSplitter
from dotenv import load_dotenv, find_dotenv
from document_processing.pdf_handler import PDFHandler
import uuid
import datetime

class VectorDatabase:
    def __init__(self, collection_name="default_collection", 
                 embedding_model="text-embedding-3-small", persist_directory="./chroma_db"):
        """
        Initialize a vector database for single PDF storage and retrieval.
        
        Args:
            collection_name: Name of the Chroma collection
            embedding_model: OpenAI embedding model to use
            persist_directory: Directory to persist the Chroma database
        """

        load_dotenv(find_dotenv())
        
        # Use MarkdownTextSplitter instead of MarkdownHeaderTextSplitter
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Set up ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Set up OpenAI embedding function
        if os.getenv("OPENAI_API_KEY") is not None:
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=embedding_model
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF file, extract markdown, split into chunks, and store in vector DB.
        Since we're only using one PDF, we'll capture essential metadata automatically.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of IDs for the stored chunks
        """
        # First, clear any existing data since we only work with one PDF
        try:
            self.delete_collection()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except:
            # Collection might not exist yet
            pass
        
        # Create PDFHandler with the given path
        pdf_handler = PDFHandler(pdf_path=pdf_path)

        # Extract markdown from PDF
        markdown_text = pdf_handler.extract_markdown()  
        
        # Split the markdown text into chunks - now returns a list of strings
        chunks = self.markdown_splitter.split_text(markdown_text)
        
        # Create unique IDs for each chunk
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Create metadata for the single PDF
        pdf_metadata = {
            "filename": os.path.basename(pdf_path),
            "file_path": os.path.abspath(pdf_path),
            "processed_date": datetime.datetime.now().isoformat(),
            "total_chunks": len(chunks),
        }
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            # Create chunk-specific metadata
            chunk_metadata = pdf_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_size_chars": len(chunk),
                "chunk_position": f"{i+1}/{len(chunks)}",
                "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
            metadatas.append(chunk_metadata)
        
        # Add chunks to the collection
        self.collection.add(
            documents=chunks,  # chunks is now a list of strings, so we can use it directly
            ids=chunk_ids,
            metadatas=metadatas
        )
        
        print(f"Processed PDF: {pdf_metadata['filename']}")
        print(f"Created {len(chunks)} chunks")
        
        return chunk_ids, pdf_metadata
    
    def search(self, query, n_results=3):
        """
        Search for chunks similar to the query.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Search results from the collection
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def delete_collection(self):
        """Delete the current collection from the database."""
        self.client.delete_collection(name=self.collection_name)
    
    def get_collection_info(self):
        """Get information about the collection and PDF."""
        count = self.collection.count()
        
        # Get metadata from first chunk to retrieve PDF info
        if count > 0:
            first_chunk = self.collection.get(ids=["chunk_0"], include=["metadatas"])
            pdf_info = {k: v for k, v in first_chunk['metadatas'][0].items() 
                       if k not in ['chunk_index', 'chunk_size_chars', 'chunk_position', 'content_preview']}
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "pdf_info": pdf_info
            }
        else:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "pdf_info": None
            }