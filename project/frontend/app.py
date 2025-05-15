import streamlit as st
from sidebar import create_sidebar
from chatbot import chatbot


# Placeholder for chatbot module
def chatbot_module():
    selected_document = st.session_state.chat_collection
    st.info(f"Selected document: {selected_document}")
    chatbot()
    pass

# Placeholder for sidebar module
def sidebar_module():
    create_sidebar()
    
   
def main():
    st.title("Github Code Explorer")
    st.write("Welcome to the Github Code Explorer. This app will help you explore code in Github.")

    # Call the sidebar module
    sidebar_module()
    # Call the chatbot module
    chatbot_module()
  

if __name__ == "__main__":
    main()
