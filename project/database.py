import chromadb
from datetime import datetime
from logger import setup_logging
from groq import Groq
import dotenv
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime


dotenv.load_dotenv()
API_KEY=os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=API_KEY)
logger=setup_logging()
class VectorDB:
    def __init__(self, db_path="./chroma_db", collection_name="code_snippets"):
        self.db_path = db_path
        logger.info(f"VectorDB initialized with path {self.db_path}")
       

    def get_client(self):
        return chromadb.PersistentClient(path=self.db_path)

    def create_collection(self, collection_name):
        client = self.get_client()
        return client.get_or_create_collection(name=collection_name)


    def delete_collection(self, collection_name):
        client = self.get_client()
        client.delete_collection(name=collection_name)
        logger.info(f"Collection {collection_name} deleted")

    def get_all_collections(self):
        client = self.get_client()
        return client.list_collections() 
    
    def add_document(self, collection_name, document):
        # Create the collection (or get an existing one)
        collection = self.create_collection(collection_name)

        # Add the document to the collection
        collection.add(
            ids=[str(document["id"])],
            embeddings=document["embedding"],
            metadatas=[document["metadata"]],
            documents=[document["metadata"]["code"]],
                 # Correctly using 'texts' instead of 'documents'
        )

        logger.info(f"Document with ID {document['id']} added to collection {collection_name}.")
    def get_all_documents(self, collection_name):
        collection = self.create_collection(collection_name)
        documents = collection.get(include=['embeddings', 'metadatas', 'documents'])

        num_docs = len(documents['documents'])
        logger.info(f"Total documents in '{collection_name}': {num_docs}")

        return documents

    def delete_all_collections(self):
        client = self.get_client()
        all_collections = client.list_collections()
        for collection in all_collections:
            client.delete_collection(collection.name)
        logger.info("All collections deleted")
    def generate_related_keywords(self, query):
        prompt = f"""
        Original query: "{query}"
        Most Relevant Keywords, Synonyms, and Phrases and variations.
        Generate a list of semantically related keywords that enhance the search intent.
        Output only the keywords as a comma-separated list without any additional text.
        Do not rewrite or expand the query—only provide related keywords.
        """
        
        # Make API call to retrieve related keywords
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides related search keywords."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=500
        )
        
        related_keywords = response.choices[0].message.content.strip()
        updated_query = f"{query} {related_keywords}"

        return updated_query
    def generate_embedding(self, query):
        # Placeholder for embedding generation logic, which should return an embedding for the 
        from code_embeddings import Embeddings
        embeddings=Embeddings()
        
        embedding=embeddings.generate_embedding(query)
        return embedding


    def first_rank_using_embeddings(self, query, collection_name=None):
        query_embedding = self.generate_embedding(query)  # (1, 768)
        
        all_embeddings = []
        all_metadatas = []

        all_collections = self.get_all_collections()

        if not all_collections:
            logger.warning("No collections found in database")
            return [], [], []

        lock = threading.Lock()

        def process_collection(each_collection):
            nonlocal all_embeddings, all_metadatas

            try:
                # Use collection.name instead of collection object
                collection_name_str = each_collection.name
                
                # Check if the collection matches the specified collection name
                if collection_name and collection_name != collection_name_str:
                    return

                print(f"Collection {collection_name_str} queried")
                collection = self.create_collection(collection_name_str)  # Use the string name

                first_rank_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=50,
                    include=["documents", "metadatas", "embeddings"],
                )
                
                if not first_rank_results["embeddings"]:
                    return

                embeddings = np.squeeze(np.array(first_rank_results["embeddings"]), axis=0)  # (N, 768)
                if embeddings.size > 0:  # Avoid empty arrays
                    with lock:  # Ensure thread-safety while modifying shared lists
                        all_embeddings.append(embeddings)
                
                with lock:  # Ensure thread-safety while modifying shared lists
                    for each_metadata in first_rank_results["metadatas"]:
                        for metadata in each_metadata:
                            all_metadatas.append(metadata)

            except Exception as e:
                logger.error(f"Error processing collection {each_collection}: {e}")

        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:  # Auto-manages threads
            for each_collection in all_collections:
                futures.append(executor.submit(process_collection, each_collection))

            for future in as_completed(futures):
                future.result()  # This raises any exceptions if they occurred in the thread

        if not all_embeddings or not all_metadatas:
            logger.warning("No results found across any collections")
            return [], [], []

        all_embeddings = np.vstack(all_embeddings)  # (Total_N, 768)
        all_metadatas = np.array(all_metadatas)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), all_embeddings)
        print(f"Similarities shape: {similarities.shape}")

        # Ensure we do not exceed the number of available documents
        max_results = min(50, len(all_metadatas))
        top_indices = np.argsort(similarities[0])[::-1][:max_results]

        # Fetch only valid indices
        top_embeddings = [all_embeddings[i] for i in top_indices]
        top_metadatas = [all_metadatas[i] for i in top_indices]
        top_similarities = similarities[0][top_indices]

        return top_embeddings, top_metadatas, top_similarities

    def second_rank_with_high_similarity(self, ranked_results, query_embedding, similarity_threshold=0.8):
        second_rank_results = []

        # Extract embeddings, metadata, and documents from the ranked results
        top_embeddings = [result["embedding"] for result in ranked_results]
        top_metadatas = [result["metadata"] for result in ranked_results]
        top_similarities = []  # Will hold the similarity scores

        # Convert list of embeddings to a NumPy array for efficient processing
        top_embeddings = np.array(top_embeddings)
        
        print(np.array(top_embeddings).shape)  # Should print (50, 768)
        
        # Compute cosine similarities for all embeddings
        cosine_similarities = cosine_similarity(query_embedding.reshape(1, -1), top_embeddings)  # (1, 50)
        
        print(np.array(cosine_similarities).shape)  # Should print (1, 50)

        # Process each result based on similarity threshold
        for idx, similarity in enumerate(cosine_similarities[0]):
            # If similarity is above the threshold, store it
            if similarity >= similarity_threshold:
                second_rank_results.append((ranked_results[idx], similarity))

            top_similarities.append(similarity)  # Store similarity score

        print(np.array(top_metadatas).shape)  # Should print (50,)

        # Sort results by the similarity score in descending order
        second_rank_results = sorted(second_rank_results, key=lambda x: x[1], reverse=True)

        # For debugging or further validation
        print("Top similarities:", top_similarities)

        return second_rank_results
    # Search With Two-Pass Ranking
    def search_with_two_pass_ranking(self, query, collection_name=None, similarity_threshold=0.8):
        # Preprocess the query
        preprocessed_query = self.generate_related_keywords(query)

        # First rank using embeddings (first pass)
        top_embeddings, top_metadatas, top_similarities = self.first_rank_using_embeddings(preprocessed_query,collection_name)
        print(np.array(top_embeddings).shape)
        print(np.array(top_metadatas).shape)
        print(np.array(top_similarities).shape)
        return json.loads(json.dumps(top_metadatas[:10]))   



def init_db():
    """Initialize the SQLite database and create necessary tables."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         index_name TEXT NOT NULL,
         query TEXT NOT NULL,
         response TEXT NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_conversation_history(index_name, limit=10):
    """Get the last 'limit' conversations for a given index."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        SELECT query, response FROM conversations 
        WHERE index_name = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (index_name, limit))
    history = c.fetchall()
    conn.close()
    return history

def store_conversation(index_name, query, response):
    """Store a new conversation in the database."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (index_name, query, response)
        VALUES (?, ?, ?)
    ''', (index_name, query, response))
    conn.commit()
    conn.close()

def delete_conversation_history(index_name):
    """Delete all conversation history for a given index."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM conversations WHERE index_name = ?', (index_name,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted conversation history for index: {index_name}") 

def delete_all_conversation_history():
    """Delete all conversation history from the database."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM conversations')
    conn.commit()
    conn.close()
    logger.info("All conversation history deleted successfully")

if __name__ == "__main__":
    print("=== Testing VectorDB ===")
    db = VectorDB()


    # List collections
    collections = db.get_all_collections()
    print("Collections:", collections)




