# app/utils.py

import time
import random
import json
import re
import streamlit as st
import pandas as pd
from google.cloud import storage
import os

def simulate_stream(text):
    """
    Simulate streaming of text by splitting it into words and yielding them one by one with a small delay.

    Args:
        text (str): The text to be streamed.

    Yields:
        str: One word at a time, simulating the streaming process with delays.
    """
    # Split the text into words
    words = text.split(" ")

    # For each word, yield the word with a random short delay
    for word in words:
        yield word + " "
        # Sleep for a short random interval between 0.01 and 0.05 seconds
        time.sleep(random.uniform(0.01, 0.1))

def extract_function_call(message_content: str) -> dict:
    """
    Extract a function call from a message's content, if present. This function looks for specific JSON patterns
    that denote function calls.

    Args:
        message_content (str): The content of the message that may contain a function call.

    Returns:
        dict: A dictionary representing the function call, or None if no function call is found.
    """
    try:
        # Look for a JSON block that starts with {function_name: ...}
        match = re.search(r'(\{"function_name"\s*:\s*".+?",\s*"arguments"\s*:\s*\{.*?\}\})', message_content, re.DOTALL)
        if match:
            json_block = match.group(1).strip()  # Extract and strip the JSON portion
            
            # Parse the JSON into a Python dictionary
            function_call = json.loads(json_block)
            
            # Ensure the necessary keys are present
            if "function_name" in function_call and "arguments" in function_call:
                print(function_call)
                return function_call  # Return the parsed function call
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # If no match is found or an error occurs, return None
    return None

def handle_function_call(function_call: dict) -> str:
    """
    Handle the function call by calling the appropriate function from the registry.

    Args:
        function_call (dict): The parsed function call, containing the function name and arguments.

    Returns:
        str: The result of the function call, or an error message if the function call fails.
    """

    from hardtack.function_registry import FUNCTION_REGISTRY

    try:
        func_name = function_call["function_name"]
        args = function_call["arguments"]
        
        # Check if the function exists in the function registry
        if func_name in FUNCTION_REGISTRY:
            func = FUNCTION_REGISTRY[func_name]
            try:
                result = func(**args)  # Call the function with the provided arguments
                print(f"Function '{func_name}' called successfully with parameters: {function_call['arguments']}")
                return result
            except Exception as e:
                return f"Error calling function '{func_name}': {e}"
        else:
            return f"Error: Function '{func_name}' is not in the registry."
    except Exception as e:
        return f"Error handling function call: {e}"

def format_recipe(
        recipe_data: dict, 
        keys_to_display: list = [
            'ingredients',
            'cooking_steps',
            'recipe_notes',
            'user_notes'
        ]):
    """Display a recipe using Streamlit's st.markdown and custom formatting.
    
    Args:
        recipe_data (dict): The recipe data loaded from the JSON file.
        keys_to_display (list): List of keys to display in the main section. Keys not in this list will be shown in a collapsible expander at the bottom.
    """
    
    # display recipe title
    recipe_title = recipe_data.get('dish_name', 'Unnamed Recipe')

    st.subheader(f"{recipe_title}")

    # extract relevant information to display in a single line
    source_name = recipe_data.get('source_name', 'Unknown Source')
    user_rating = recipe_data.get('rating', None)
    active_time = recipe_data.get('active_time', None)
    total_time = recipe_data.get('total_time', None)

    # add checkmark if already cooked
    if user_rating is not None:
        recipe_title += " ✅"

    # create the single-line display
    inline_elements = []

    if source_name:
        inline_elements.append(f"**Source:** {source_name}")
    
    if active_time is not None and total_time is not None:
        inline_elements.append(f"**Time:** {active_time}/{total_time} min")
    
    if user_rating is not None and isinstance(user_rating, (int, float)):
        full_stars = int(user_rating)
        inline_elements.append(f"**Rating:** {'⭐️' * full_stars}")
    
    # combine the elements into a single line
    if inline_elements:
        inline_display = " — ".join(inline_elements)
        st.markdown(inline_display)
    
    # track keys that aren't displayed in the main section
    keys_not_displayed = []

    # loop through the specified keys to display, in order
    for key in keys_to_display:
        if key in recipe_data:
            value = recipe_data[key]  # get the value from the recipe

            # add a header for each section
            st.markdown(f"**{key.replace('_', ' ').title()}**")

            # handle different types of values
            if key == 'ingredients' and isinstance(value, dict):
                # display ingredients as a table using streamlit's st.table()
                ingredients_table = pd.DataFrame(
                    [(ingredient, details[0] if len(details) > 0 else '', details[1] if len(details) > 1 else '') 
                     for ingredient, details in value.items()],
                    columns=["Ingredient", "Amount", "Preparation"]
                )
                st.table(ingredients_table)  # display the table

            elif key == 'cooking_steps' and isinstance(value, list):
                # **numbered list for cooking steps**
                for i, step in enumerate(value, start=1):
                    st.markdown(f"{i}. {step}")

            elif isinstance(value, list):
                # **bullet list for other list items (like tags, recipe notes)**
                st.markdown('<ul>', unsafe_allow_html=True)
                for item in value:
                    st.markdown(f'<li>{item}</li>', unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)

            elif isinstance(value, dict):
                # **dictionary items (like some additional fields) as key-value pairs**
                st.markdown('<ul>', unsafe_allow_html=True)
                for subkey, subvalue in value.items():
                    formatted_value = ', '.join(subvalue) if isinstance(subvalue, list) else subvalue
                    st.markdown(f'<li>**{subkey}**: {formatted_value}</li>', unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)

            else:
                # **simple fields (str, int, float, bool)**
                if isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                else:
                    display_value = str(value)
                st.markdown(f"{display_value}")

    # identify keys that are not in keys_to_display
    for key, value in recipe_data.items():
        if key not in keys_to_display and key not in ['dish_name', 'cooked_already', 'source_name', 'user_rating', 'active_time', 'total_time']:
            keys_not_displayed.append((key, value))

    # show keys that were not displayed in the main section
    if keys_not_displayed:
        with st.expander("Additional Recipe Data"):
            for key, value in keys_not_displayed:
                # add a header for each additional key
                st.markdown(f"**{key.replace('_', ' ').title()}**")

                # handle different types of values
                if isinstance(value, list):
                    st.markdown('<ul>', unsafe_allow_html=True)
                    for item in value:
                        st.markdown(f'<li>{item}</li>', unsafe_allow_html=True)
                    st.markdown('</ul>', unsafe_allow_html=True)
                
                elif isinstance(value, dict):
                    st.markdown('<ul>', unsafe_allow_html=True)
                    for subkey, subvalue in value.items():
                        formatted_value = ', '.join(subvalue) if isinstance(subvalue, list) else subvalue
                        st.markdown(f'<li>**{subkey}**: {formatted_value}</li>', unsafe_allow_html=True)
                    st.markdown('</ul>', unsafe_allow_html=True)
                
                else:
                    if isinstance(value, bool):
                        display_value = "Yes" if value else "No"
                    else:
                        display_value = str(value)
                    st.markdown(f"{display_value}")

def list_files(bucket_name):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)