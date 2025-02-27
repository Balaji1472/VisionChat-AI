import streamlit as st
from llm_chains import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from image_handler import handle_image
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create chat_sessions directory if it doesn't exist
os.makedirs(config["chat_history_path"], exist_ok=True)


def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def save_chat_history():
    if st.session_state.history != []:
        # Ensure filename has .json extension
        if st.session_state.session_key == "new_session" and st.session_state.new_session_started:
            if not st.session_state.new_session_key:
                st.session_state.new_session_key = get_timestamp() + ".json"
            file_path = os.path.join(config["chat_history_path"], st.session_state.new_session_key)
            save_chat_history_json(st.session_state.history, file_path)
        elif st.session_state.session_key != "new_session":
            file_path = os.path.join(config["chat_history_path"], st.session_state.session_key)
            save_chat_history_json(st.session_state.history, file_path)

def main():
    st.title("Conversational Image Recognition Chatbot")
    chat_container = st.container()
    st.sidebar.title("Chat Session")
    
    # Initialize all session state variables at the start
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.audio_processed = False
        st.session_state.new_session_started = False
        st.session_state.previous_session_key = "new_session"
        st.session_state.history = []
        st.session_state.last_audio = None

    # List only .json files from chat_sessions directory
    chat_sessions = ["new_session"] + [
        f for f in os.listdir(config["chat_history_path"]) 
        if f.endswith('.json')
    ]

    # Load existing session or create new one
    session_key = st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key")
    
    # Handle session changes
    if session_key != st.session_state.previous_session_key:
        chat_history = StreamlitChatMessageHistory(key="history")
        chat_history.clear()
        
        if session_key != "new_session":
            file_path = os.path.join(config["chat_history_path"], session_key)
            if os.path.exists(file_path):
                loaded_history = load_chat_history_json(file_path)
                for msg in loaded_history:
                    chat_history.add_message(msg)
        else:
            st.session_state.new_session_started = False
            st.session_state.new_session_key = None
        
        st.session_state.previous_session_key = session_key
    else:
        chat_history = StreamlitChatMessageHistory(key="history")

    llm_chain = load_chain(chat_history)

    # Chat interface
    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)
    
    upload_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    
    send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    # Handle input
    if (send_button or st.session_state.send_input) and st.session_state.user_question:
        st.session_state.new_session_started = True
        
        if upload_image:
            with st.spinner("Processing image..."):
                user_message = "Describe this image in detail please."
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                llm_answer = handle_image(upload_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)
        else:
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                response = llm_chain.run(st.session_state.user_question)
                st.chat_message("assistant").write(response)

        st.session_state.user_question = ""
        st.session_state.send_input = False

    # Display chat history
    if chat_history.messages:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == "__main__":
    main()