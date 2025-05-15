from llama_index.core.node_parser import CodeSplitter
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_c as tsc
import json
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
from groq import Groq
from database import VectorDB
import tree_sitter_cpp as tscpp
import tree_sitter_ruby as tsruby
import tree_sitter_typescript as tstypescript
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import json
import os
import threading
import concurrent.futures
from tqdm import tqdm
import logging
import numpy as np
import requests
import os
from groq import Groq
from logger import setup_logging
import dotenv
dotenv.load_dotenv()

API_KEY=os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=API_KEY)
logger=setup_logging()

class ModelCacheManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name="microsoft/codebert-base", cache_dir="../model_cache"):
        if not self._initialized:
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.tokenizer = None
            self.model = None
            self._initialized = True
            # Ensure the cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("ModelCacheManager initialized")

    def _get_cached_path(self, file_type):
        return os.path.join(self.cache_dir, file_type)

    def load_model(self):
        """Load the model and tokenizer from the cache, or download if not available."""
        if self.tokenizer is not None and self.model is not None:
            logger.info("Using already loaded model and tokenizer")
            return self.tokenizer, self.model
            
        try:
            tokenizer_cache_path = self._get_cached_path("tokenizer")
            model_cache_path = self._get_cached_path("model")
            
            if os.path.exists(tokenizer_cache_path) and os.path.exists(model_cache_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
                self.model = AutoModel.from_pretrained(model_cache_path)
                logger.info("Model and tokenizer loaded from local cache.")
            else:
               
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("Model and tokenizer downloaded.")
                # Cache the model and tokenizer
                self.tokenizer.save_pretrained(tokenizer_cache_path)
                self.model.save_pretrained(model_cache_path)
                logger.info("Model and tokenizer saved to local cache.")
            
            return self.tokenizer, self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None

    def clear_cache(self):
        """Clear the loaded model and tokenizer from memory."""
        self.tokenizer = None
        self.model = None
        logger.info("Model and tokenizer cache cleared from memory.")

    def delete_cache_files(self):
        """Delete the model and tokenizer cached files from disk."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cache files at {self.cache_dir} have been deleted.")
        except Exception as e:
            logger.error(f"Error deleting cache files: {e}")

    def reload_model(self):
        """Reload the model and tokenizer after clearing the cache."""
        return self.load_model()


class Embeddings:

    def __init__(self):
        print("Initializing Embeddings")
        self.model_cache_manager = ModelCacheManager()
        self.tokenizer, self.model = self.model_cache_manager.load_model()
        self.chroma_client = VectorDB()
        logger.info("Embeddings initialized")

    def generate_embedding(self, code):
        # Use the stored tokenizer and model directly
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def set_parser(self,language):
        if language == "python":
            LANGUAGE = Language(tspython.language())
            parser = Parser(LANGUAGE)
        elif language == "javascript":
            LANGUAGE = Language(tsjavascript.language())
            parser = Parser(LANGUAGE)
        elif language == "java":
            LANGUAGE = Language(tsjava.language())
            parser = Parser(LANGUAGE)
        elif language == "go":
            LANGUAGE = Language(tsgo.language())
            parser = Parser(LANGUAGE)
        elif language == "c":
            LANGUAGE = Language(tsc.language())
            parser = Parser(LANGUAGE)
        elif language == "cpp":
            LANGUAGE = Language(tscpp.language())
            parser = Parser(LANGUAGE)
        elif language == "ruby":
            LANGUAGE = Language(tsruby.language())
            parser = Parser(LANGUAGE)

        elif language == "typescript":
            LANGUAGE = Language(tstypescript.language())
            parser = Parser(LANGUAGE)
        return parser
    def store_code_chunks_in_db(self,file_path,language):
        try:
            with open(file_path,"r") as file:
                code=file.read()
           
            result=self.convert_code_to_embeddings_with_meta_data(code,file_path,language)
            
        except Exception as e:
            logger.error(f"Error storing code chunks in database: {e}")


    def read_selected_files(self, file_path):
        try:
            with open(file_path, "r") as file:
                json_data = json.load(file)

            def process_file(file_path, language):
                try:
                    logger.info(f"Processing file {file_path}")
                    self.store_code_chunks_in_db(file_path, language)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

            futures = []
            with ThreadPoolExecutor(max_workers=8) as executor:  # Auto-manages threading
                for language, details in json_data.items():
                    for path in details["files"]:
                        futures.append(executor.submit(process_file, path, language))

                for future in as_completed(futures):
                    future.result()  # This raises any thread exception if occurred

            logger.info("Code chunks stored in database")

        except Exception as e:
            logger.error(f"Error reading selected files: {e}")

    def generate_code_description(self,code):
        #description = self.describe_code(code)
        description="this is a code description"
        full_code_with_description = f"Code:\n{code}\n\nDescription:\n{description}"
        return full_code_with_description
       
    def describe_code(self,code_snippet):
        """
        Describes the functionality of the given code snippet using a language model (e.g., LLaMA).

        Parameters:
            code_snippet (str): The code to describe.
            client (object): The client used to interact with the language model API.

        Returns:
            str: The generated description of what the code does.
        """
        # Define the prompt for code description
        prompt = f"Explain in detail what the following code does:\n\n{code_snippet}"

        # Send the prompt to the chat completion API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant who explains code clearly and concisely."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Use the specific model for generating code explanations
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        # Extract and return the description from the model's response
        return chat_completion.choices[0].message.content
    def get_chunk_metadata_with_embedding(self,code_chunk,file_path,language,start_line,end_line,id):
        
        document={
            "id":id,
            "embedding":self.generate_embedding(self.generate_code_description(code_chunk)),
            "metadata":{
                "file_path":file_path,
                "language":language,
                "start_line":start_line,
                "end_line":end_line,
                "code":code_chunk,
            }
        }
        return document
    def convert_code_to_embeddings_with_meta_data(self, code, file_path, language):
        parser = self.set_parser(language)
        splitter = CodeSplitter(
            language=language,
            chunk_lines=5,
            parser=parser,
            chunk_lines_overlap=2,
            max_chars=500
        )
        chunks = splitter.split_text(code)
        code_lines = code.splitlines()
        collection_name = self.get_valid_collection_name(file_path)

        # Shared mutable state
        current_line = [0]  # Use list to make it mutable inside threads
        line_lock = threading.Lock()
        chroma_lock = threading.Lock()

        def process_chunk(i, chunk):
            logger.info(f"Processing chunk {i+1} of {len(chunks)}")

            with line_lock:
                start_line = current_line[0] + 1
                end_line = start_line + len(chunk.splitlines()) - 1
                current_line[0] = end_line

            document = self.get_chunk_metadata_with_embedding(chunk, file_path, language, start_line, end_line, i)

            with chroma_lock:
                self.chroma_client.add_document(collection_name=collection_name, document=document)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]

            for future in as_completed(futures):
                future.result()

        logger.info(f"Code chunks stored in database for {file_path}")

    def get_valid_collection_name(self,file_path):
        # Generate a valid collection name by hashing the file path (or just take part of it)
        collection_name = file_path.replace("\\", "/")  # Replace backslashes with forward slashes
        collection_name = collection_name.split("/")[-1]  # Take the last part (file name)
        collection_name = collection_name.split(".")[0]  # Remove file extension   
        return collection_name
