import streamlit as st
import requests
import json
from streamlit_float import *
from hardtack import get_bot_response
from hardtack.utils import format_recipe
import os

# set the page title and layout
st.set_page_config(page_title="hardtack", layout="wide")

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
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                if True:  # change to False if you want to disable streaming
                    response = st.write_stream(get_bot_response(user_input))
                else:
                    response = get_bot_response(user_input, stream=False)
            
            # add assistant response to chat history
            st.session_state['chat_history'].append(('assistant', response))

with col2:
    if 'selected_recipe_uuid' in st.session_state:
        if st.session_state['selected_recipe_uuid']!='No recipe selected':
            recipe_uuid = st.session_state['selected_recipe_uuid']
            
            # Construct the file path for the recipe JSON file
            file_path = os.path.join('data', 'raw', 'json', f'{recipe_uuid}.json')
            
            if os.path.isfile(file_path):
                try:
                    # Read and display the JSON file
                    with open(file_path, 'r') as file:
                        recipe_data = json.load(file)
                    
                    # Call the function to display the recipe
                    format_recipe(recipe_data)
                
                except Exception as e:
                    st.error(f"Error reading recipe file: {e}")
            else:
                st.warning(f"No recipe file found for UUID: {recipe_uuid}")
