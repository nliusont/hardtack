# app/search.py

import json
import requests
import weaviate
import streamlit as st
import os
from weaviate.classes.query import MetadataQuery


def define_query_params(user_input: str, model: str = 'qwen2.5', query_temp: float = 0.5, server_url: str = "http://192.168.0.19:11434"):
    """
    Define the parameters for a search query based on the user's input.

    Args:
        user_input (str): The user's input that will define the query parameters.
        model (str): The model used to generate the query parameters.
        query_temp (float): Temperature setting for generating the query.
        server_url (str): The URL of the server to query the model.

    Returns:
        dict: A dictionary containing the query parameters.
    """
    prompt = f"""
    You are an expert chef and recipe writer. You are interacting with a user who is looking for recipes in a database.
    Please generate a concise set of search parameters based on the user's input.

    User input:
    {user_input}
    """
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': query_temp,
                    "num_ctx": 32768
                },
            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])
    except Exception as e:
        print(f"Error communicating with the server: {e}")
        return {}


def query_vectors(query_params, collection_name='Recipe', num_matches=5):
    """
    Perform a similarity search using vectors based on the given query parameters.

    Args:
        query_params (dict): The parameters for the search query.
        collection_name (str): The name of the collection to search in.
        num_matches (int): The number of matches to return.

    Returns:
        dict: A dictionary containing the distances for the search results.
        list: A list of dimensions that were searched.
    """
    headers = {
        "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
    }
    client = weaviate.connect_to_local(headers=headers)
    collection = client.collections.get(collection_name)

    searched_dimensions = [dim for dim, terms in query_params.items() if len(terms) > 0]

    recipe_distances = {}
    for dimension, query_terms in query_params.items():
        if len(query_terms) > 0:
            search_query = ','.join(query_terms)
            response = collection.query.near_text(
                query=search_query,
                limit=num_matches,
                target_vector=[f"{dimension}_vector"], 
                return_metadata=MetadataQuery(distance=True)
            )

            for obj in response.objects:
                uuid = str(obj.uuid)
                dist = obj.metadata.distance
                if uuid not in recipe_distances:
                    recipe_distances[uuid] = {}

                    # Store the distance for the current dimension
                    recipe_distances[uuid][dimension] = dist
    client.close()

    return recipe_distances, searched_dimensions


def score_query_results(recipe_distances, searched_dimensions):
    """
    Calculate the scores for each recipe based on the query dimensions and their distances.

    Args:
        recipe_distances (dict): The distances for each recipe from the search query.
        searched_dimensions (list): The dimensions that were searched.

    Returns:
        dict: A dictionary with recipe UUIDs as keys and their combined scores as values.
    """
    combined_scores = {}

    for recipe_uuid, distances in recipe_distances.items():
        total_distance = 0
        for dim in searched_dimensions:
            total_distance += distances.get(dim, 1)  # If dimension is missing, assign a default distance of 1

        combined_scores[recipe_uuid] = total_distance / len(searched_dimensions)

    return combined_scores


def retrieve_results(combined_scores, top_n=3, json_dir='data/json'):
    """
    Retrieve the top search results based on the combined scores.

    Args:
        combined_scores (dict): The scores for each recipe.
        top_n (int): The number of top results to retrieve.
        json_dir (str): The directory where the recipe JSON files are stored.

    Returns:
        dict: A dictionary of the top search results, where each key is a recipe number (e.g., recipe_1, recipe_2).
    """
    sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1])
    top_recipes = sorted_scores[:top_n]

    combined_json = {}

    for i, (recipe_uuid, score) in enumerate(top_recipes, start=1):
        json_path = os.path.join(json_dir, f"{recipe_uuid}.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as file:
                    recipe_data = json.load(file)
                    combined_json[f"recipe_{i}"] = recipe_data  # Store it as recipe_1, recipe_2, etc.
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        else:
            print(f"Error: The file for UUID {recipe_uuid} does not exist at {json_path}")

    return combined_json


def summarize_results(
    user_input: str,
    results: str,
    model: str = 'llama3.1',
    temp: float = 0.4,
    server_url: str = "http://192.168.0.19:11434",
    stream: bool = False
):
    """
    Summarize the search results and present them to the user in a concise, professional manner.

    Args:
        user_input (str): The original user input.
        results (str): The search results to summarize.
        model (str): The model used to generate the summary.
        temp (float): The temperature setting for the model.
        server_url (str): The URL of the server.
        stream (bool): Whether to stream the response.

    Returns:
        str: The summary of the search results.
    """
    chat_history = st.session_state.get('chat_history', [])
    formatted_chat_history = ""

    max_messages = 10
    for role, message in chat_history[-max_messages:]:
        if role == 'user':
            formatted_chat_history += f"User: {message}\n"
        elif role == 'assistant':
            formatted_chat_history += f"Assistant: {message}\n"

    prompt = f"""
        You are an expert chef and recipe writer. You are interacting with a user that is looking for recipes in a database so they can make a dish.
        Following this message is a user request along with a JSON of several recipes that have been returned by a recommendation engine based on the request. 
        The recipes are ordered such that the closest match is first.

        Please review the results along with the user input and the chat history. Do the following:
        1. Decide which recipes to present to the user.  You do not have to return every recipe if you do not believe they match the user's interest.
        2. For the recipes you choose to present, briefly summarize them to the user to help the user decide on which to cook. 
        
        User input:
        {user_input}

        Search results:
        {results}

        Chat history context:
        {formatted_chat_history}
    """

    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temp,
                    "num_ctx": 32768
                }
            },
            stream=stream
        )

        if response.status_code == 200:
            data = response.json()
            return data['response']
        else:
            return f"Error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the server. Details: {e}"


def summarize_single_search(
    user_input: str,
    results: str,
    model: str = 'llama3.1',
    temp: float = 0.4,
    server_url: str = "http://192.168.0.19:11434",
    stream: bool = False
):
    """
    Summarize a single search result and present it to the user.

    Args:
        user_input (str): The original user input.
        results (str): The search result to summarize.
        model (str): The model used to generate the summary.
        temp (float): The temperature setting for the model.
        server_url (str): The URL of the server.
        stream (bool): Whether to stream the response.

    Returns:
        str: The summary of the search result.
    """
    chat_history = st.session_state.get('chat_history', [])
    formatted_chat_history = ""

    max_messages = 10
    for role, message in chat_history[-max_messages:]:
        if role == 'user':
            formatted_chat_history += f"User: {message}\n"
        elif role == 'assistant':
            formatted_chat_history += f"Assistant: {message}\n"

    prompt = f"""
        You are an expert chef and recipe writer. You are interacting with a user that is looking for a specific recipe in a database.
        Following this message is a user request along with a JSON of several recipes that have been returned possibly the recipe the user is seeking.
        The recipes are ordered such that the closest match is first.

        Please review the results along with the user input and the chat history. Do the following:
        1. Decide which recipe is most likely to the recipe the user is looking for.
        2. Present a summary of the single recipe and explain why it might match what they're seeking.

        User input:
        {user_input}

        Search results:
        {results}

        Chat history context:
        {formatted_chat_history}
    """

    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temp,
                    "num_ctx": 32768
                }
            },
            stream=stream
        )

        if response.status_code == 200:
            data = response.json()
            return data['response']
        else:
            return f"Error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the server. Details: {e}"
