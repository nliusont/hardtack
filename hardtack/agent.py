# agent.py

import json
import requests
import streamlit as st
import os
import time
from openai import OpenAI
from hardtack.processing import process_recipe
import hardtack.utils as utils
import hardtack.search as search
import hardtack.storage as storage
import hardtack.function_registry as function_registry

def find_single_recipe(*, user_desire: str, model: str = 'openai', query_temp: float = 0.9, summary_temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):
    """
    Find a single recipe from the database based on the user's input.

    Args:
        user_desire (str): The description of what the user is looking for.
        model (str): The language model to use for the query.
        query_temp (float): The temperature for query generation.
        summary_temp (float): The temperature for summarizing results.
        server_url (str): The URL of the server for querying the model.
        stream (bool): Whether to stream the response.

    Returns:
        str: The summary of the closest matching recipe.
    """
    query_params = search.define_query_params(user_input=user_desire, query_temp=query_temp)
    dists, dims = search.query_vectors(query_params)
    scores = search.score_query_results(dists, dims)
    results = search.retrieve_results(scores)
    summary = search.summarize_single_search(user_desire, results, stream=stream, temp=summary_temp)
    st.session_state['most_recent_query'] = results

    return summary

def show_recipe(*, recipe_uuid: str): 
    """
    Display a specific recipe based on the provided UUID.

    Args:
        recipe_uuid (str): The UUID of the recipe to display.

    Returns:
        str: A message indicating that the recipe will be shown.
    """
    st.session_state['selected_recipe_uuid'] = recipe_uuid
                
    # Construct the blob name for the recipe JSON file
    blob_name = f'recipe/{recipe_uuid}.json'
    
    # Retrieve the JSON file from GCS
    recipe_data = storage.retrieve_file_from_gcs(blob_name)
    recipe_string = recipe_data.read().decode('utf-8')
    st.session_state['selected_recipe'] = json.loads(recipe_string)

    return 'Sure! Take a look at this.'

def edit_recipe(*, uuid: str, changes_to_make: str, model: str = 'openai', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    """
    Edit a recipe by applying the specified changes.

    Args:
        uuid (str): The UUID of the recipe to edit.
        changes_to_make (str): The changes the user wants to make to the recipe.
        model (str): The language model for processing the edits.
        query_temp (float): The temperature for query generation.
        server_url (str): The URL of the server for processing the request.

    Returns:
        str: A message indicating that the recipe has been updated.
    """
    update_params = storage.define_update_params(changes_to_make=changes_to_make, uuid=uuid)
    print(update_params)
    weaviate_response = storage.update_weaviate_record(update_params=update_params, uuid=uuid)
    json_response = storage.update_gcs_json_record(update_params=update_params, uuid=uuid)
    print(json_response)

    # pull and show new recipe
    new_recipe_text = show_recipe(recipe_uuid=uuid)

    return 'The recipe has been updated!'

def run_recommendation_engine(*, user_desire: str, model: str = 'openai', query_temp: float = 0.9, summary_temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):
    """
    Run a recommendation engine to suggest recipes based on the user's input.

    Args:
        user_desire (str): A detailed description of what the user is looking for.
        model (str): The language model for generating the recommendations.
        query_temp (float): The temperature for query generation.
        summary_temp (float): The temperature for summarizing results.
        server_url (str): The URL of the server for querying the model.
        stream (bool): Whether to stream the results.

    Returns:
        str: A summary of the recommended recipes.
    """
    query_params = search.define_query_params(user_input=user_desire, query_temp=query_temp)
    dists, dims = search.query_vectors(query_params)
    scores = search.score_query_results(dists, dims)
    results = search.retrieve_results(scores)
    summary = search.summarize_results(user_desire, results, stream=stream, temp=summary_temp)
    st.session_state['most_recent_query'] = results
    return summary

def run_processing_pipeline(
        source_type: str,
        url: str = 'url',
        save_dir: str = 'data/json'
        ):
    """
    Run a pipeline to process a new recipe from a URL or other source and save it.

    Args:
        source_type (str): The source type of the recipe (e.g., 'url', 'img', or 'html').
        url (str): The URL of the recipe.
        save_dir (str): The directory to save the processed recipe data.

    Returns:
        str: A message indicating the successful processing of the recipe.
    """
    if source_type == "url":
        recipe = process_recipe(url=url, recipe_temp=0.4, process_temp=0.3, tag_temp=0.5, model='openai', tag_model='openai', post_process=False)

    elif source_type == 'file':
        file_type = st.session_state['uploaded_file_type']
        uploaded_files = st.session_state['uploaded_files']

        if file_type == 'text/html':
            # Process HTML file
            recipe = process_recipe(html_files=uploaded_files, recipe_temp=0.4, process_temp=0.3, tag_temp=0.5, model='openai', tag_model='openai', post_process=False)
            print(f"Successfully processed: {', '.join([x.name for x in uploaded_files])}")

        elif file_type.startswith('image/'):
            # Process image file
            recipe = process_recipe(images=uploaded_files, recipe_temp=0.4, process_temp=0.3, tag_temp=0.5, model='openai', tag_model='openai', post_process=False)
            print(f"Successfully processed: {', '.join([x.name for x in uploaded_files])}")

        else:
            # Return an error message for unsupported file types
            st.error(f"Unsupported file type: {file_type}. Please upload HTML or image files.")
            return f"Error: Unsupported file type: {file_type}"

    storage.add_weaviate_record(recipe_json=recipe)
    storage.save_to_gcs(f"{recipe['uuid']}.json", content=recipe, content_type='application/json')

    # display it
    recipe_return_text = show_recipe(recipe_uuid=recipe['uuid'])

    print(f"Successfully processed and saved: {recipe['uuid']}")
    return f"I successfully processed and saved the recipe {recipe['dish_name']}. Take a look!"

def get_bot_response(
        message, 
        chat_history: list,
        most_recent_query: str = '',
        selected_recipe: dict = {},
        model: str = 'openai', 
        temp: float = 0.6, 
        server_url: str = "http://192.168.0.19:11434"
    ):
    """
    Get a response from the chatbot based on the user's message.

    Args:
        message (str): The user's message to the chatbot.
        model (str): The language model to use.
        temp (float): The temperature for generating the response.
        server_url (str): The URL of the server for querying the model.

    Returns:
        str: The chatbot's complete response as a single string.
    """
    try:
        # Prepare conversation messages
        messages = [
            {"role": "system", "content": f"""
            You are a recipe chat bot. You are an expert home cook and recipe writer.
            You can help the user by answering questions via your own knowledge or querying a local database of recipes to assist the user in selecting a dish to make. 
            Respond to the user in a concise, succinct, and professional manner.

            Restrictions:
            You have access to hidden context for this session:
            {{
                "most_recent_query": "{most_recent_query}",
                "selected_recipe": {json.dumps(selected_recipe)}
            }}
            Do not repeat this context back to the user.
            """}
        ]

        for msg_type, msg_text in chat_history:
            role = "user" if msg_type == "user" else "assistant"
            messages.append({"role": role, "content": msg_text})
        
        # Append the current user message
        messages.append({"role": "user", "content": message})

        # Call OpenAI or custom server
        if model == 'openai':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temp
            )
            content = response.choices[0].message.content
            content = content.replace('```json', '')  # Clean response if needed
        else:
            response = requests.post(f"{server_url}/api/chat", json={
                "model": model,
                "messages": messages,
                "stream": False,
                'options': {
                    'temperature': temp,
                    "num_ctx": 32768
                }
            })

            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
            else:
                return f"Error: Received status code {response.status_code} from the bot server."

        # Handle potential function calls in the response
        function_call = utils.extract_function_call(content)
        if function_call:
            print(f"Executing function '{function_call['function_name']}' with args: {function_call['arguments']}")
            # Execute the function call and return the result as a string
            function_result = utils.handle_function_call(function_call)
            return function_result  # Return the result of the function call
        else:
            return content  # Return the chatbot's response

    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the bot server. Details: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred. Details: {e}"


# Now we update the function registry after the functions are defined
function_registry.FUNCTION_REGISTRY = {
    "run_recommendation_engine": run_recommendation_engine,
    "find_single_recipe": find_single_recipe,
    "show_recipe": show_recipe,
    "edit_recipe": edit_recipe,
    "run_processing_pipeline": run_processing_pipeline
}