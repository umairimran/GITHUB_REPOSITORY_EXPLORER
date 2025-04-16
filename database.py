import chromadb
from datetime import datetime
import logging

logger=logging.getLogger(__name__)
class VectorDB:
    def __init__(self, db_path="./chroma_db", collection_name="code_snippets"):
        self.db_path = db_path
       

    def get_client(self):
        return chromadb.PersistentClient(path=self.db_path)

    def create_collection(self, collection_name):
        client = self.get_client()
        return client.get_or_create_collection(name=collection_name)


    def delete_collection(self, collection_name):
        client = self.get_client()
        client.delete_collection(name=collection_name)

    def get_all_collections(self):
        client = self.get_client()
        return [collection.name for collection in client.list_collections()]

    
    def add_document(self, collection_name, document):
        # Create the collection (or get an existing one)
        collection = self.create_collection(collection_name)

        # Add the document to the collection
        collection.add(
            ids=[str(document["id"])],
            embeddings=[document["embedding"]],
            metadatas=[document["metadata"]],
                 # Correctly using 'texts' instead of 'documents'
        )

        logger.info(f"Document with ID {document['id']} added to collection {collection_name}.")




if __name__ == "__main__":
    db=VectorDB()
    print(db.get_all_collections())
