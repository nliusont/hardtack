# app/search.py

import json
import requests
import weaviate
import streamlit as st
import os
from openai import OpenAI
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from hardtack.storage import retrieve_file_from_gcs


def define_query_params(user_input: str, model: str = 'openai', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
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
    You are an expert chef and recipe writer. You are interacting with a user that is looking for recipes in a database so they can make a dish.
    Please use a concise and professional tone with the user.

    You are able to search this database both with vector / semantic search and with explicit filtering.
    
    Vector searching is available for the following dimensions:
    - dish_name - The name of the dish.
    - tags - Tags that describe the dish. These cover meal types (e.g. dinner), dish types (e.g. soup, salad, sandwich), flavor profiles (e.g. spicy, sweet), cuisines (e.g. Indian, Mexican), and cooking styles or equipment (e.g. Pressure cooker, easy)
    - shopping_list - This is of the ingredients that go in the dish.
    - source_author - The name of the publication/author where the recipe was sourced. This can be things like a website, book, publication, app, or person.

    Explicit filtering is available for the following dimensions:
    - rating - the past rating of the dish by the user (float). Dishes that have not been cooked have a rating of Null while dishes that have been cooked have a numerical 0-5 rating. To search by rating, provide a list of the numerical value and the operator. Available operators are:
        - equal
        - not_equal
        - greater_than
        - greater_or_equal
        - less_than
        - less_or_equal
        - is_null

    If the user asks for dihes that have NOT been cooked yet, pass: [None, "is_null"]. 
    To simply search for recipes that have ALREADY been cooked, pass: [0.0, "greater_or_equal"]

    Your job is to decide which of the above dimensions you would like to search (one or many) and then to decide what your search query terms for that dimension should be.
    The system will take your decision and do a similarity search with embeddings of your search queries. Use your knowledge of cooking and food to generate relevant tags, dish names or ingredients to find a matching recipe.
    In order to generate tags, it might help for you to generate a list of adjectives that are similar to the descriptions provided by the user.

    For example, if the user is looking for desserts with walnuts and brown sugar you would decide to search the tags and shopping_list dimensions. You would return a JSON as below:
    {{
        "dish_name":[],
        "tags":["dessert", "sweet"],
        "shopping_list":["walnuts", "brown sugar"],
        "rating":[]
    }}

    Or if the user is looking for Coq Au Vin or similar dishes which have a rating of at least 3, you would return a JSON as below
    {{
        "dish_name":["Coq Au Vin"],
        "tags":["stew", "braised", "savory", "hearty", "rustic"],
        "shopping_list":["red wine"],
        "rating":[3.0, "greater_or_equal"]
    }}
    
    You do not need to explicitly defined a search dimension. The system knows you wish to search a given dimension if the list of query terms has more than one item. 
    Restrictions:
    Do not add ingredients to the shopping list that the user has not specified.
    Do not search for a specific dish name unless the user has specified it.

    User input:
    {user_input}
    """
    try:
        if model=='openai':

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=query_temp
            )
            
            content = response.choices[0].message.content
            content = content.replace('```json', '').replace('```', '')
            result = json.loads(content)
            return result
        else:
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
        print(f"define_query_params error communicating with the server: {e}")
        return {}


def query_vectors(query_params, collection_name='Recipe', num_matches=5, db='remote'):
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

    print(f"Querying Weaviate for: {query_params}")

    headers = {
        "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
    }
    if db=='local':
        client = weaviate.connect_to_local(headers=headers)
    else:
        weaviate_url = os.environ["WEAVIATE_URL"]
        weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers=headers,
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
              )  # Values in seconds
        )
        
    collection = client.collections.get(collection_name)

    # get total # of searched dimensions for scoring
    searched_dimensions = [dim for dim, terms in query_params.items() if len(terms) > 0 and dim != 'rating']

    operand_mapping = {
        "greater_than": lambda v: Filter.by_property("rating").greater_than(v),
        "greater_or_equal": lambda v: Filter.by_property("rating").greater_or_equal(v),
        "less_than": lambda v: Filter.by_property("rating").less_than(v),
        "less_or_equal": lambda v: Filter.by_property("rating").less_or_equal(v),
        "equal": lambda v: Filter.by_property("rating").equal(v),
        "no_equal": lambda v: Filter.by_property("rating").not_equal(v),
        "is_null": lambda _: Filter.by_property("rating").is_null(),
    }

    # determine if a rating filter is needed
    rating_filter = None
    if len(query_params.get('rating', [])) > 0:
        value, operator = query_params['rating']
        if operator in operand_mapping:
            rating_filter = operand_mapping[operator](value)

    recipe_distances = {}
    for dimension, query_terms in query_params.items():
        if len(query_terms) > 0 and dimension != 'rating':
            search_query = ','.join(query_terms)

            if rating_filter:
                response = collection.query.near_text(
                    query=search_query,
                    limit=num_matches,
                    target_vector=[f"{dimension}_vector"],
                    return_metadata=MetadataQuery(distance=True),
                    filters=rating_filter
                )
            else:
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
        blob_name = f"recipe/{recipe_uuid}.json"
        
        try:
            recipe_file = retrieve_file_from_gcs(blob_name)
            recipe_data = json.load(recipe_file)
            combined_json[f"recipe_{i}"] = recipe_data  # Store it as recipe_1, recipe_2, etc.
        except Exception as e:
            print(f"Error loading {blob_name}: {e}")

    return combined_json


def summarize_results(
    user_input: str,
    results: str,
    model: str = 'openai',
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

        Do not say anything like "I've reviewed the search results" or "Here are the results of your query" or "Based on your query". 
        As far as the user is concerned, you are recommending recipes as if they are your own knowledge.
        Present the results as your own personalized recommendations.
        
        Keep your summary concise, succinct and professional.
        Use markdown like newline breaks to format your response.

        User input:
        {user_input}

        Search results:
        {results}

        Chat history context:
        {formatted_chat_history}
    """

    try:
        if model=='openai':

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=temp
            )
            
            content = response.choices[0].message.content
            return content
        else:
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
                return f"summarize_results error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"summarize_results error: Could not connect to the server. Details: {e}"


def summarize_single_search(
    user_input: str,
    results: str,
    model: str = 'openai',
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
        You are an expert chef and recipe writer. You are interacting with a user that is looking for a specifc recipe in a database so they can make a dish.
        Following this message is a user request along with a JSON of several recipes that have been returned possibly the recipe the user is seeking. 
        The recipes are ordered such that the closest match is first.

        Please review the results along with the user input and the chat history. Do the following:
        1. Decide which recipe is most likely to the recipe the user is looking for.
        2. Present a summary of the single recipe and explain why it might match what they're seeking.

        Do not say anything like "I've reviewed the search results" or "Here are the results of your query" or "Based on your query". 
        Present the recipe if you are a librarian returning with a book. Do not present the entire recipe to the user. You may ask them if they want you to show it or display it to them.
        
        Keep your summary concise, succinct and professional.
        Use markdown like newline breaks to format your response.

        User input:
        {user_input}

        Search results:
        {results}

        Chat history context:
        {formatted_chat_history}
    """

    try:
        if model=='openai':

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=temp
            )
            
            content = response.choices[0].message.content
            return content
        else:
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
                return f"summarize_single_search error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"summarize_single_search error: Could not connect to the server. Details: {e}"
