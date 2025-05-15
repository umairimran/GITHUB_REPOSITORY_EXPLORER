import streamlit as st
import datetime
from api_service import APIClient

# Initialize API client
api_client = APIClient()

def initialize_chat_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'chat_collection' not in st.session_state:
        st.session_state.chat_collection = None

def add_to_chat_history(document_name, message, is_user=False, timestamp=None):
    """Add a message to the chat history"""
    if document_name not in st.session_state.chat_history:
        st.session_state.chat_history[document_name] = []
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    message_data = {
        "message": message,
        "is_user": is_user,
        "timestamp": timestamp
    }
    
    st.session_state.chat_history[document_name].append(message_data)

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    if not timestamp_str:
        return ""
    return timestamp_str

def chatbot():
    """Main chatbot function"""
    # Initialize chat state
    initialize_chat_state()

    # Get selected document
    selected_document = st.session_state.get("chat_collection")
   
    if not selected_document:
        st.info("Please select a document from the sidebar to start chatting.")
        return

    # Display chat history
    if selected_document in st.session_state.chat_history:
        for message in st.session_state.chat_history[selected_document]:
            time = format_timestamp(message.get("timestamp", ""))
            text = message["message"]
            
            if message.get("is_user", False):
                # User message
                with st.chat_message("user"):
                    st.markdown(f"<div style='font-size: 22px;'>{text}</div>", unsafe_allow_html=True)
                    st.caption(f"{time}")
            else:
                # Assistant message 
                with st.chat_message("assistant"):
                    st.markdown(f"<div style='font-size: 22px;'>{text}</div>", unsafe_allow_html=True)
                    st.caption(f"{time}")

    # Chat input
    user_query = st.chat_input("Ask something about the document...")
    
    if user_query:
        try:
            # Add user message to chat history
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            add_to_chat_history(selected_document, user_query, is_user=True, timestamp=current_time)
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = api_client.chat(
                    user_id="user123",  # Replace with actual user ID if available
                    query=user_query,
                    collection_name=selected_document
                )
                
                # Format response
                if isinstance(response, dict):
                    resp_text = response.get('answer') or response.get('response') or str(response)
                else:
                    resp_text = str(response)
                
                # Add bot response to chat history
                add_to_chat_history(selected_document, resp_text, timestamp=current_time)
                
                # Rerun to update the UI
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
