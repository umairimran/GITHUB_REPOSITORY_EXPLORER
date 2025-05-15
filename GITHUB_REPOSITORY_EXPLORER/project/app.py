from fastapi import FastAPI
import uvicorn
from code_embeddings import Embeddings
from file_filter import filter_files_by_extension
from database import VectorDB,get_conversation_history,store_conversation,delete_conversation_history,init_db,delete_all_conversation_history
import json
from pydantic import BaseModel
from google import genai
from typing import List, Optional, Dict
import httpx

app=FastAPI()
init_db()
client = genai.Client(api_key="AIzaSyBBiWiE7DXrbmyUAjRW_tu4InxAic1xrUY")
def generate_response(query, history, collection_name):
    # Step 1: Search code using the vector database
    db = VectorDB()
    search_results = db.search_with_two_pass_ranking(query, collection_name)

    # Step 2: Check if any results found
    if search_results:
        # Step 3: Limit number of results for clarity
        limited_results = search_results

        # Step 4: Format recent conversation history for context
        formatted_history = ""
        if history:
            recent_history = history
            formatted_history = "\n".join([
                f"User: {q}\nAI: {a}" for q, a in recent_history
            ])

        # Step 5: Build prompt for the LLM
        prompt = (
            "You are an expert AI assistant specialized in Python codebases.\n"
            "Here is the recent conversation for context:\n\n"
            f"{formatted_history}\n\n"
            f"The user now asks:\nUser: {query}\n\n"
            "Below are code snippets from the project. For each snippet:\n"
            "if user ask about specific code then only explain that code and dont explain other code"
            "if user ask about what is this or summarize then explain the whole code"
            "- Explain what the code does in a clear and concise way.\n"
            
            "- Follow the explanation with a brief reference to the file and line numbers.\n\n"
        )

        # Add code snippets to the prompt with explanation requests
        for res in limited_results:
            prompt += (
                f"```python\n{res['code'].strip()}\n```\n"
                f"What this code does:\n"
                f"(📄 `{res['file_path']}`, lines {res['start_line']}-{res['end_line']})\n\n"
            )

        # Step 6: Generate explanation using the LLM
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Step 7: Return the final, trimmed response
        final_response = response.text.strip()

    else:
        final_response = "No relevant code found for your query."

    return final_response

class ChatRequest(BaseModel):
    user_id: str
    query: str
    collection_name: str

class EmbeddingsRequest(BaseModel): 
    base_dir: str
    ignored_items: list[str]
    output_file: str = "code_files.json"

@app.post("/make-code-embeddings")
async def create_embeddings(embeddings_request: EmbeddingsRequest):
    """
    Create embeddings for code files in the specified directory and store them in the database.
    
    Args:
        base_dir (str): Base directory path containing the code files
        ignored_items (list[str], optional): List of files/folders to ignore
        output_file (str, optional): Output JSON file name. Defaults to "code_files.json"
    
    Returns:
        dict: Status of the embedding creation process
    """
    try:
        print(f"Creating embeddings for base_dir: {embeddings_request.base_dir}")
        # Filter files based on extensions and ignored items
        filtered_files = filter_files_by_extension(
            base_dir=embeddings_request.base_dir,
            output_file=embeddings_request.output_file,
            ignored_items=embeddings_request.ignored_items if embeddings_request.ignored_items else None
        )

        # Create and store embeddings
        embeddings_object = Embeddings()
        embeddings_object.read_selected_files(embeddings_request.output_file)

        return {
            "status": "success",
            "message": "Code embeddings created successfully",
            "files_processed": filtered_files
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create embeddings: {str(e)}"
        }

@app.post("/delete-all-collections")
async def delete_all_collections():
    db=VectorDB()
    db.delete_all_collections()
    return {"message": "All collections deleted successfully"}


@app.post("/search-code")
async def search_code(query: str):
    db=VectorDB()
    results=db.search_with_two_pass_ranking(query)
    return {"results": results}


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        # Retrieve conversation history for the user
        history = get_conversation_history(chat_request.collection_name)
        
        # Generate a response using the modified function
        response = generate_response(chat_request.query, history,chat_request.collection_name)
        
        # Store the conversation
        store_conversation(chat_request.collection_name, chat_request.query, response)
        
        return {
            "collection_name": chat_request.collection_name,
            "query": chat_request.query,
            "response": response,
            "history": history
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process chat request: {str(e)}"
        }


@app.get("/")
def read_root():
    return {"message": "Hello World"}
@app.get("/documents")
async def get_all_documents():
    try:
        db = VectorDB()
        document_names = db.get_all_collections()
        return {"document_names": document_names}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/chat-history")
async def delete_all_chat_history():
    try:
        delete_all_conversation_history()
        return {"status": "success", "message": "All chat history deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # Replace with actual module name (e.g. main:app)
        host="127.0.0.1",
        port=8080,
        reload=True
    )





