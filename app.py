import streamlit as st
import requests
from funcs import get_bot_response

# set the page title and layout
st.set_page_config(page_title="Hardtack.bot", layout="wide")

# initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

col1, col2 = st.columns(2)

with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(get_bot_response(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})