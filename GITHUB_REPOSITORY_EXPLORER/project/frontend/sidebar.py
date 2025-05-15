# sidebar.py

import streamlit as st
from api_service import APIClient
from typing import Optional, List, Dict

api_client = APIClient()

def create_sidebar():
    # Add logo and branding
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="background-color: #1976d2; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
            <span style="color: white; font-size: 22px; font-weight: bold;">🏢</span>
        </div>
        <h1 style="margin: 0; padding: 0; font-size: 24px;">Github Code Explorer</h1>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Collection For Chat")
    
    # Call the function to handle collection selection
    select_collection_for_chat()

    # Add section for creating code embeddings
    st.sidebar.subheader("Create Code Embeddings")
    create_code_embeddings_section()

    delete_all_embeddings_section()

def select_collection_for_chat():
    """Handles the selection of a collection for chat."""
    try:
        collections = api_client.get_all_documents()
        if collections:
            # Sort collection names to ensure consistent order
            collections = collections['document_names']
            collections.sort()
            
            # Collection selection section
            selected_collection = st.sidebar.selectbox(
                "Select a collection to chat with",
                collections,
                key="chat_collection",
                format_func=lambda x: x.replace('_', ' ').title() if isinstance(x, str) else x
            )
            
            # Update session state with the selected collection
            st.session_state.selected_collection = selected_collection
        else:
            st.sidebar.info("No collections available. Please upload a document first.")

    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")

def create_code_embeddings_section():
    """Allows the user to input a base directory and create code embeddings."""
    base_dir = st.sidebar.text_input("Enter Base Directory for Code Embeddings")
    if st.sidebar.button("Create Embeddings"):
        if base_dir:
            with st.spinner("Creating embeddings... This may take a while."):
                try:
                    # Log the base directory for debugging
                    st.sidebar.info(f"Base Directory: {str(base_dir)}")
                    # Call the API to start the embedding process
                    result = api_client.create_code_embeddings(base_dir)
                    st.sidebar.success("Project is being processed.")
                    st.rerun()
                except Exception as e:
                    if "timeout" in str(e).lower():
                        st.sidebar.error("The request timed out. Please try again later.")
                    else:
                        st.sidebar.error(f"Error creating embeddings: {e}")
        else:
            st.sidebar.warning("Please enter a base directory.")


def delete_all_embeddings_section():
    """Allows the user to delete all embeddings."""
    if st.sidebar.button("Delete All Embeddings"):
        api_client.delete_all_collections()
        api_client.delete_all_chat_history()
        st.sidebar.success("All embeddings deleted successfully.")
        st.rerun()

