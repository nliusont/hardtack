import streamlit as st
from google.cloud import storage
from streamlit_float import *
from hardtack import get_bot_response
from hardtack.utils import format_recipe
from hardtack.storage import retrieve_file_from_gcs
import os
import time
import requests
from dotenv import load_dotenv
load_dotenv()

API_URL='http://127.0.0.1:8000/bot-response/'

# set the page title and layout
st.set_page_config(page_title="hardtack", layout="wide")

#----------Authentication----------
if 'authenticated' not in st.session_state or st.session_state['authenticated']==False:
    with st.form("login_form", clear_on_submit=True):
        st.title("🔒 Login Required")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if password == os.getenv('PASSWORD'):
                st.session_state["authenticated"] = True
                st.success("welcome, chef!")
                time.sleep(1)  # delay for smooth transition
                st.rerun()
            else:
                st.error("WRONG!")
else:
#----------Initialize Session State----------
    # initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'most_recent_query' not in st.session_state:
        st.session_state['most_recent_query'] = 'No queries run yet'

    if 'selected_recipe_uuid' not in st.session_state:
        st.session_state['selected_recipe_uuid'] = 'No recipe selected'
        st.session_state['selected_recipe'] = {}
    
    # initialize float UI
    float_init(theme=True, include_unstable_primary=False)

    # header
    st.title("hardtack")
    col1, col2 = st.columns(2)
    
    with col1:
        st.divider()
        #----------File Uploads----------
        with st.expander('Upload files'):
            # File uploader widget
            uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

            # Check if a file is uploaded
            if uploaded_files:
                # Display file details
                for file in uploaded_files:
                    st.write("Filename:", file.name)
                file_type = uploaded_files[0].type
                st.write("File type:", file_type)

                st.session_state['uploaded_file_type'] = file_type
                st.session_state['uploaded_files'] = uploaded_files
        
        #----------Chat Interface----------
        # display chat messages from history on app rerun
        for message in st.session_state['chat_history']:
            with st.chat_message(message[0]):
                st.markdown(message[1])

        # use a container to make the chat input stick to the bottom
        with st.container():
            # float UI for chat input
            with st.container():
                user_input = st.chat_input("Type your message here", key='content')
                button_b_pos = "0rem"
                button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                float_parent(css=button_css)
            
            if user_input:  # check if the user entered something
                # add user message to chat history
                st.session_state['chat_history'].append(('user', user_input))
                
                with st.chat_message("user", avatar="🧑‍🍳"):
                    st.markdown(user_input)
                
                with st.chat_message("assistant", avatar="🕵️"):
                    
                    # call session state vars to pass to endpoint
                    chat_history = st.session_state.get('chat_history', [])
                    # sanitize chat_history so it's JSON-serializable
                    chat_history = [
                        {"role": msg_type, "content": str(msg_text) if isinstance(msg_text, str) else "<Non-Serializable Content>"}
                        for msg_type, msg_text in st.session_state.get("chat_history", [])
]
                    most_recent_query = st.session_state.get('most_recent_query', '')
                    selected_recipe = st.session_state.get('selected_recipe', {})

                    try:
                        # send POST request to fastapi endpoint
                        response = requests.post(
                            API_URL,
                            json={
                                "message": user_input,
                                "chat_history": chat_history,
                                "most_recent_query": most_recent_query,
                                "selected_recipe": selected_recipe,
                                "model": "openai",
                            },
                        )

                        # check that request was successful
                        if response.status_code == 200:
                            # stream in chunks
                            for chunk in response.iter_lines():
                                if chunk:  # skip empty chunks
                                    st.write_stream((chunk.decode("utf-8") for chunk in response.iter_lines() if chunk))
                        else:
                            # error handling
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        # error handling
                        st.error(f"Failed to connect to the API: {e}")
                
                # add assistant response to chat history
                st.session_state['chat_history'].append(('assistant', response))
    with col2:
        st.divider()
        #----------Display Recipe----------
        if 'selected_recipe_uuid' in st.session_state:
            if st.session_state['selected_recipe_uuid']!='No recipe selected':
                recipe_uuid = st.session_state['selected_recipe_uuid'] 

                recipe_data = st.session_state['selected_recipe'] 
                
                # call the function to display the recipe
                format_recipe(recipe_data)
                
            else:
                st.warning("No recipe selected.")
