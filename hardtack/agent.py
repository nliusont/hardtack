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
    st.session_state['last_updated_recipe'] = time.time()

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

def get_bot_response(message, model: str = 'openai', temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):

    """
    Get a response from the chatbot based on the user's message.

    Args:
        message (str): The user's message to the chatbot.
        model (str): The language model to use.
        temp (float): The temperature for generating the response.
        server_url (str): The URL of the server for querying the model.
        stream (bool): Whether to stream the response.

    Yields:
        str: The chatbot's response, possibly in streaming format.
    """
    try:
        most_recent_query = st.session_state.get('most_recent_query', 'No queries run yet')
        selected_recipe = st.session_state.get('selected_recipe', {})

        messages = [
            {"role": "system", "content": f"""
            You are a recipe chat bot. You are an expert home cook and recipe writer.
            You can help the user by answering questions via your own knowledge or querying a local database of recipes to assist the user in selecting a dish to make. 
            Respond to the user in a concise, succinct, and professional manner.
            
            If needed, you have access to several functions.
            Do not call a function based on any older messages. Completely ignore any previous requests for function calls from earlier in the conversation when deciding to call a function in the current moment. 
            The decision to call a function depends solely on the user's latest message.
             
            Functions available:
            1. run_recommendation_engine(user_desire="<YOUR INPUT>") - Triggers a pipeline that provides recommendations based on user input. Only trigger this function if the user asks you for recommendations in their most recent message. The user_desires is a positional input that is a string. It should be a summary of what the user is looking for including flavor profile, cuisine, type of equipment (e.g. pressure cooker), meal type (e.g. lunch), food type (e.g. soup, salad), ingredients, etc. The more descriptive the better. If the user specifies a desired rating and operator (e.g. greater than), specify that. If that user asks for a dish that hasn't been cooked yet, it means they're looking for dishes that have a is_null rating. If they are looking for dishes they HAVE cooked, they are looking for dishes with a rating greater than or equal to 0.
            2. find_single_recipe(user_desire="<YOUR INPUT>") - Triggers a pipeline to search for a single known recipe. Trigger this function when the user is asking you to find a specific recipe that is known to exist in the database. The "name" field is likely the key to searching. The user_desires is a positional input that is a string. It should be a summary of what the user is looking for including flavor profile, cuisine, type of equipment (e.g. pressure cooker), meal type (e.g. lunch), food type (e.g. soup, salad), ingredients, etc. The more descriptive the better.
            3. show_recipe(recipe_uuid="<YOUR INPUT>") - Displays the entire recipe for the user to read and respond to. Takes the UUID of the recipe they want to see. If a user asks you to "show" the a recipe, then they likely want this function.
            4. edit_recipe(uuid="<YOUR INPUT>", changes_to_make="<YOUR DESCRIPTION OF CHANGES>") - Updates the record of a recipe in the database. You can edit/update specific fields by describing in detail the changes and fields to make.
            5. run_processing_pipeline(source_type=<url or html or img, url=str) - This function is to add a new recipe to the database. If the user wants to add a recipe, specify what the source type is. source_type can either be 'url', 'file'. If the source_type is 'url', provide the url as a string. If the user has alludes to "attached" or "uploaded" images or html, then the source type is 'file' and you can assume they uploaded them. You do not need to ask them to provide the image file. Simply run this function.
             
            When you want to call a function, respond with this exact format and DO NOT RESPOND WITH ANY OTHER TEXT:
            ```json
            {{"function_name": "name_of_function", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
             
            For example, to run function 1, it might look like this:
            ```json
            {{"function_name": "run_recommendation_engine", "arguments": {{"user_desire": "<YOUR INPUT>"}}}}

            If the user has not asked for you to search the database or for recommendations, then just respond normally.
             
            Restrictions:
            You can tell the user about your abilities, but do not surface the exact function calls to the user.
            DO NOT offer up recipes to the user unless they've been returned to you by the database. 
            If you call a function, do not produce any text after your function call json.
             
            You have access to some important hidden context for this chat. This context is **not to be repeated to the user in any form**. 
            Here is the hidden context for this session:
            {{
                "most_recent_query": "{most_recent_query}",
                "selected_recipe": {json.dumps(selected_recipe)}
            }}
            Do not repeat this context back to the user. Use it to inform your responses, but do not mention it.
            """}
        ]
        
        for msg_type, msg_text in st.session_state['chat_history']:
            role = "user" if msg_type == "user" else "assistant"
            messages.append({"role": role, "content": msg_text})
        
        messages.append({"role": "user", "content": message})

        if model == 'openai':

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temp
            )

            content = response.choices[0].message.content
            content = content.replace('```json', '')
        else:
            response = requests.post(f"{server_url}/api/chat", json={
                "model": model, 
                "messages": messages,
                "stream": stream,
                'options': {
                    'temperature': temp,
                    "num_ctx": 32768
                }
            }, stream=stream)

            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
            else:
                yield f"get_bot_response error: Received status code {response.status_code} from the bot server."
        
        function_call = utils.extract_function_call(content)
        if function_call:  # If a function call was detected
            print(f"Executing function '{function_call['function_name']}' with args: {function_call['arguments']}")
            function_result = utils.handle_function_call(function_call)
            for chunk in utils.simulate_stream(function_result):
                yield chunk
        else:
            for chunk in utils.simulate_stream(content):
                yield chunk

    except requests.exceptions.RequestException as e:
        yield f"get_bot_response error: Could not connect to the bot server. Details: {e}"

# Now we update the function registry after the functions are defined
function_registry.FUNCTION_REGISTRY = {
    "run_recommendation_engine": run_recommendation_engine,
    "find_single_recipe": find_single_recipe,
    "show_recipe": show_recipe,
    "edit_recipe": edit_recipe,
    "run_processing_pipeline": run_processing_pipeline
}
