from llama_index.core.node_parser import CodeSplitter
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_c as tsc
import json
import dotenv
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

logger=logging.getLogger(__name__)
class ModelCacheManager:
    def __init__(self, model_name="microsoft/codebert-base", cache_dir="../model_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None

        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cached_path(self, file_type):
        return os.path.join(self.cache_dir, file_type)

    def load_model(self):
        """Load the model and tokenizer from the cache, or download if not available."""
        try:
            tokenizer_cache_path = self._get_cached_path("tokenizer")
            model_cache_path = self._get_cached_path("model")
            
            if os.path.exists(tokenizer_cache_path) and os.path.exists(model_cache_path):
                print("Loading model and tokenizer from local cache.")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
                self.model = AutoModel.from_pretrained(model_cache_path)
            else:
                print("Downloading model and tokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                # Cache the model and tokenizer
                self.tokenizer.save_pretrained(tokenizer_cache_path)
                self.model.save_pretrained(model_cache_path)
            
            return self.tokenizer, self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def clear_cache(self):
        """Clear the loaded model and tokenizer from memory."""
        self.tokenizer = None
        self.model = None
        print("Model and tokenizer cache cleared from memory.")

    def delete_cache_files(self):
        """Delete the model and tokenizer cached files from disk."""
        try:
            shutil.rmtree(self.cache_dir)
            print(f"Cache files at {self.cache_dir} have been deleted.")
        except Exception as e:
            print(f"Error deleting cache files: {e}")

    def reload_model(self):
        """Reload the model and tokenizer after clearing the cache."""
        return self.load_model()


class Embeddings:

    def __init__(self):
        print("Initializing Embeddings")
        self.model_cache_manager=ModelCacheManager()
        self.model_cache_manager.load_model()
        self.chroma_client=VectorDB()

    def generate_embedding(self,code):
        model_cache_manager=ModelCacheManager()
        model_cache_manager.load_model()
        tokenizer=model_cache_manager.tokenizer
        model=model_cache_manager.model
        inputs=tokenizer(code,return_tensors="pt",truncation=True,padding=True)
        with torch.no_grad():
            outputs=model(**inputs)
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

    def read_selected_files(self,file_path):
        try:
            with open(file_path,"r") as file:
                json_data=json.load(file)
            for language,details in json_data.items():
                for file_path in details["files"]:
                    print(file_path)
                    logger.info(f"Processing file {file_path}")
                    self.store_code_chunks_in_db(file_path,language)
                    return None
            logger.info("Code chunks stored in database")
        except Exception as e:
            logger.error(f"Error reading selected files: {e}")
    def generate_code_description(self,code):
        return code
        pass
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
    def convert_code_to_embeddings_with_meta_data(self,code,file_path,language):
        parser=self.set_parser(language)
        splitter=CodeSplitter(
            language=language,
            chunk_lines=5,
            parser=parser,
            chunk_lines_overlap=2,
            max_chars=500
        )
        code_lines=code.splitlines()
        chunks=splitter.split_text(code)
        chunks_to_store=[]
        current_line=0
        for i,eachChunk in enumerate(chunks):
           
            logger.info(f"Processing chunk {i+1} of {len(chunks)}")
            start_line = current_line + 1
            end_line = start_line + len(eachChunk.splitlines()) - 1
            current_line = end_line
            print(start_line,end_line)
            chunk_document=(self.get_chunk_metadata_with_embedding(eachChunk,file_path,language,start_line,end_line,i))
            
            self.chroma_client.add_document(collection_name=self.get_valid_collection_name(file_path),document=chunk_document)

        logger.info(f"Code chunks stored in database for {file_path}")


    def get_valid_collection_name(self,file_path):
        # Generate a valid collection name by hashing the file path (or just take part of it)
        collection_name = file_path.replace("\\", "/")  # Replace backslashes with forward slashes
        collection_name = collection_name.split("/")[-1]  # Take the last part (file name)
        collection_name = collection_name.split(".")[0]  # Remove file extension   
        return collection_name

            
if __name__ == "__main__":
    
    embeddings=Embeddings()
    embeddings.read_selected_files("code_files.json")
