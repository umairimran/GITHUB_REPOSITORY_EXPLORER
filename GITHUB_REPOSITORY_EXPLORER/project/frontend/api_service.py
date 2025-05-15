import httpx
from typing import List, Dict, Any, Optional
import json
import requests
import time
import streamlit as st

class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=100.0)

    def check_api_connection(self):
        """Check if the API is ready and database is connected"""
        try:
            # Using base URL since there's no health endpoint
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def create_code_embeddings(self, base_dir: str, ignored_items: Optional[List[str]] = None, output_file: str = "code_files.json") -> Dict:
        """Create code embeddings for the specified directory."""
        try:
            # Construct the JSON body
            data = {
                "base_dir": base_dir,
                "ignored_items": ignored_items or [],
                "output_file": output_file
            }
            # Log the request data for debugging
            print("Request data:", data)
            # Send the request with JSON body
            response = self.client.post(f"{self.base_url}/make-code-embeddings", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error creating code embeddings: {str(e)}")

    def delete_all_collections(self) -> Dict:
        """Delete all collections from the backend."""
        try:
            response = self.client.post(f"{self.base_url}/delete-all-collections")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error deleting all collections: {str(e)}")

    def search_code(self, query: str) -> Dict:
        """Search code using the specified query."""
        try:
            data = {"query": query}
            response = self.client.post(f"{self.base_url}/search-code", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error searching code: {str(e)}")

    def chat(self, user_id: str, query: str, collection_name: str) -> Dict:
        """Chat with the system using the specified query and collection name."""
        try:
            data = {
                "user_id": user_id,
                "query": query,
                "collection_name": collection_name
            }
            response = self.client.post(f"{self.base_url}/chat", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error during chat: {str(e)}")

    def get_all_documents(self) -> Dict:
        """Get all document names from the backend."""
        try:
            response = self.client.get(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error getting documents: {str(e)}")

    def delete_all_chat_history(self) -> Dict:
        """Delete all chat history from the backend."""
        try:
            response = self.client.delete(f"{self.base_url}/chat-history")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error deleting chat history: {str(e)}")

    def __del__(self):
        """Close the HTTP client when the object is destroyed."""
        self.client.close()



