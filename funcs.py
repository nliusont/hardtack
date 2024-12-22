from PIL import Image
import pytesseract
import requests
import json
import os
import pillow_heif
from bs4 import BeautifulSoup
from datetime import datetime
import uuid
import weaviate
from weaviate.classes.query import MetadataQuery
import streamlit as st
import re
import time
import random
import pandas as pd
import base64
import io
from PIL import Image
from dotenv import load_dotenv
from fake_useragent import UserAgent
load_dotenv()

def extract_recipe(
        text: str, 
        model: str = 'llama3.1',
        tag_model: str = 'qwen2.5', 
        recipe_temp: float = 0.3, 
        tag_temp: float = 0.75, 
        server_url: str = "http://192.168.0.19:11434") -> dict:
    """
    Extract structured recipe information using an Ollama server.

    Parameters:
        raw_text (str): The text extracted from OCR.
        server_url (str): The URL of the Ollama server.

    Returns:
        dict: Structured recipe information.
    """
    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is a recipe for a dish. Please extract the below information from the text and store it as indicated:\
    - dish_name - The name of the dish, not the name of the recipe.
    - ingredients - stored as a dictionary of ingredients in the form k:[v1, v2] such as <ingredient_name>: [<quantity>, <preparation>]. For example, 'white onion': ['1 cup', 'diced'], 'all purpose flour':['1 cup'], 'walnuts':['2 cups', 'chopped']. If there is no preparation, skip it. The ingredients will be used as search indices, so they should be standardized. If the dish provides both imperial and metric quantities, only choose the imperial units.
    - shopping list - stored as a list of ingredients. Do not include quantities or preparations. For example, 'poblano peppers, roughly chopped, seeds and stems discarded', should be stored as 'poblano peppers'. 
    Or 'loosely packed fresh cilantro leaves and fine stems' should be stored as 'cilantro'. Essentially, it should be the keys of the ingredients dictionary.
    - cooking_steps - stored as list
    - active_time - amount of time actively cooking in minutes
    - total_time - total cook time from start to finish in minutes
    - source_name - NYTimes, Hellofresh, Blue Apron, Serious Eats, etc.
    - author - The name of the person who developed the recipe, if available.

    Return the result as a JSON object, structured as below:

    {{
    "dish_name": str(),
    "ingredients": dict(),
    "date_added": str(),
    "cooking_steps": list(str())
    "active_time": int(),
    "total_time": int(),
    "source_name": str(),
    "author": str(),
    "shopping_list": list(str()),
    }}



    Text for extraction: 
    {text}
    """
    print('Requesting recipe extraction...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,  # replace with the model name you're using
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature':recipe_temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        output = json.loads(result['response'])
        tags_and_notes = interpret_recipe(text=text, model=tag_model, temp=tag_temp, server_url=server_url)
        output['tags'] = tags_and_notes['tags']
        output['recipe_notes'] = tags_and_notes['recipe_notes']

        return output

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}
    
def interpret_recipe(text: str, model: str = 'qwen2.5', temp: float = 0.5, server_url: str = "http://192.168.0.19:11434") -> list:
    '''Recipe extract requires low temp, but tag extraction requires high temp. So send a separate call for tags.
    Tag extraction is much better with qwen2.5'''

    prompt = f"""
        You are an expert chef and recipe writer.

        Following this message is a recipe for a dish. Your task is to produce:
        
        1. A list of 12 to 15 tags representing the dish’s cuisine, flavor profile, meal type, cooking method, seasonality, dietary considerations, difficulty level, and occasion. 
        The tags are keywords that help classify the dish across multiple dimensions. The tags will be used in semantic search. The tags should help a RAG model find this dish easily.
        
        Generate a list of 12 to 15 tags that classify the dish across multiple dimensions. Requirements for tags:

        - Do not use the dish name as a tag (e.g., if the recipe is "Chicken Noodle Soup," do not tag it as "Chicken Noodle Soup"). The RAG model already separately searches by dish name.
        - Do not use simple ingredient names as tags (e.g., "Chicken," "Noodles," "Carrots" on their own are too generic unless they define the dish type). The RAG model already separately searches by ingredient.
        - Consider tags for:
            - Cuisine/Heritage (e.g., American, Italian, Mexican, Homestyle, French, Middle Eastern, Classic, Modern, Traditional, Jewish, Southeast Asian, Mediterranean, Street food, Fusion)
            - Flavor/Texture Profile (e.g., Savory, Spicy,  Tangy, Smoky, Rich, Crunchy, Herbaceous, Umami, Tangy, Sweet, Creamy, Crispy)
            - Dish Type (e.g., Soup, Roast, Bread, Salad, Sandwich, Casserole, Stew, Pasta, Curry, Sushi, Burrito, Wrap, Dip, BBQ, Pizza)
            - Cooking Method (e.g., Stir-Fry, One-Pot, Oven, Air fryer, Pressure cooker, Grilled, Boiled)
            - Meal Type (e.g., Side Dish, Dessert, Breakfast, Lunch, Dinner, Party appetizer, Brunch)
            - Dietary Notes (e.g., Vegetarian, Gluten-Free, Dairy-Free, Low-Carb, Vegan)
            - Difficulty/Accessibility (e.g., Quick, Easy, Meal-Prep Friendly, Complex, Simple ingredients, Make-ahead friendly, Advanced)
            - Seasonality/Context/Vibes (e.g., Fall, Summer, Comfort Food, Family-Style, Holidays, Indulgent, Healthy, Cozy, Light, Comforting)
            
        2. Recipe Notes: Provide practical cooking tips that help someone prepare the dish. These can include tips on substitutions, techniques, presentation, or serving suggestions. 
        Do not include the dish name or any history, just helpful tips.

        Return the tags as a list in JSON as follows:
        {{
        "tags": [ ... ],
        "recipe_notes": [ ... ]
        }}

        Do not return any other text except for this JSON structure.

        Text for extraction: 
        {text}
        """
    print('Requesting recipe interpretation...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,  # replace with the model name you're using
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature':temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return []

def post_process_recipe(recipe: str, text: str, model: str = 'llama3.1', temp: float = 0.2, server_url: str = "http://192.168.0.19:11434"):
    
    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is an extracted recipe for a dish that was extracted from an image or website by an LLM. Below that is the raw text that the LLM extracted from.
    Your job is to further refine and clean the recipe per the instructions below.
    The purpose of this is to catalogue the recipe as part of a database. Therefore, everything must be standardize and universal.

    Please check and clean the following:
    - dish_name - This should be the name of the food, not the name of the recipe.
    - cooking_steps - This should be a simple list, where each item in the list is a step. There is no need to number or demarcate steps.
    - ingredients - Ensure that the ingredients are stored in one dictionary with each ingredient as a key in form k:[v1, v2] such as <ingredient_name>: [<quantity>, <preparation>]. If the dish provides both imperial and metric quantities, only choose the imperial units.
        - Example: 'all purpose flour':['1 cup'], 'walnut':['2 cups', 'chopped']
    - tags - Review the list of and enforce logical constraints:
        - Remove "Vegetarian" if the dish contains meat.  
        - Remove "Vegan" if the dish contains meat or dairy (cheese or eggs).
        - Remove "Quick" if the dish cook time is long.
        - Remove "Easy" if the active time is long or the instructions are complex.  
        - Remove "Gluten free" if it has gluten/wheat products in the ingredients.
        - Return the tags as a clean list with no extra explanations. ONLY REMOVE TAGS IF THEY BREAK A LOGICAL CONSTRAINT.
    - shopping_list - This is of the ingredients that go in the dish. It should be a simple list, not a dictionary. It is just the name of the ingredients, they do not include preparations or quantities. Essentially, it should be the keys of the ingredients dictionary.
    - recipe_notes - Provide any additional guidance that someone making the recipe should know (e.g. notes on process, tips, instructions, etc). This should be information not included in your cooking steps. DO NOT INCLUDE RECIPE HISTORY.
    The purpose of the shopping list is to provide a standard list of ingredients that can be indexed against. Therefore, review and edit this list so it is standardized.

    Return your cleaned version of the recipe as a JSON in the below structure. If a field that is below doesn't exist in the extracted recipe, create it.

    {{
        "dish_name": str(),
        "ingredients": dict(),
        "date_added": str(),
        "cooking_steps": list(str())
        "active_time": int(),
        "total_time": int(),
        "source_name": str(),
        "author": str(),
        "shopping_list": list(str()),
        "tags": list(str()),
        "recipe_notes": list(str())
    }}

    Extracted recipe:
    {recipe}

    Raw text source:
    {text}
    """
    print('Cleaning things up...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,  # replace with the model name you're using
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature':temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}

def fetch_html_from_url(url):

    from fake_useragent import UserAgent

    # Randomized user agent
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/',
        'Cache-Control': 'no-cache'
    }
    try:
        # send GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url} - {e}")
        return None
    
    return response

def parse_html(html):
    # Check if html is a Response object, if so, extract the content
    if hasattr(html, 'content'):
        html = html.content

    # Parse the page content
    soup = BeautifulSoup(html, 'lxml')

    # Remove script, style, and hidden elements
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    # Extract all text from visible elements
    text = soup.get_text(separator=' ')

    # Clean the text
    cleaned_text = clean_text(text)
    return cleaned_text

def clean_text(text):
    import re
    # Step 1: Remove extra spaces within each line
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with a single space
    
    # Step 2: Remove extra blank lines (more than one consecutive newline)
    text = re.sub(r'\n{2,}', '\n\n', text)  # Collapse multiple blank lines into one blank line
    
    # Step 3: Strip leading and trailing whitespace on each line
    text = '\n'.join([line.strip() for line in text.split('\n')])
    
    return text.strip()

def process_recipe(
        *, 
        url: str = None, 
        images: list = None, 
        html_file: str = None, 
        model: str = 'llama3.1', 
        tag_model: str = 'qwen2.5',
        recipe_temp: float = 0.4, 
        tag_temp: float = 0.5, 
        process_temp: float = 0.3,
        save_dir: str = 'data/raw/', 
        post_process: bool = True):
    # Count how many of the mutually exclusive arguments are provided
    provided_sources = sum([url is not None, images is not None, html_file is not None])
    if provided_sources != 1:
        raise ValueError("You must provide exactly one of 'url', 'images', or 'html'.")
    
    identifier = str(uuid.uuid4())
    
    if url:
        html = fetch_html_from_url(url)
        cleaned_text = parse_html(html)
        file_path = os.path.join(f'{save_dir}/parsed_html/', f'{identifier}.txt')
        with open(file_path, "wb" if isinstance(cleaned_text, bytes) else "w") as f:
            f.write(cleaned_text)
    elif images:
        cleaned_text = extract_text_from_images(images)
    elif html_file:
        scraped_text = parse_html(html_file)
        cleaned_text = clean_text(scraped_text)

    recipe = extract_recipe(text=cleaned_text, recipe_temp=recipe_temp, tag_temp=tag_temp, model=model, tag_model=tag_model)
    if post_process:
        processed_recipe = post_process_recipe(recipe, cleaned_text, temp=process_temp, model=model)
    else:
        processed_recipe = recipe
    if processed_recipe is None:
        raise ValueError("Failed to post-process the recipe with the LLM.")
    
    processed_recipe['date_added'] = datetime.now().strftime('%Y-%m-%d')

    if url:
        processed_recipe['url'] = url
    if images:
        processed_recipe['image_paths'] = images
    # if html_file:
    #     processed_recipe['html_file'] = html_file

    processed_recipe['uuid'] = identifier
    processed_recipe['user_notes'] = str()
    processed_recipe['user_rating'] = None
    processed_recipe['cooked_already'] = bool()
    return processed_recipe

def define_query_params(user_input: str, model: str = 'qwen2.5', query_temp: float = 0.5, server_url: str = "http://192.168.0.19:11434"):
    
    prompt = f"""
    You are an expert chef and recipe writer. You are interacting with a user that is looking for recipes in a database so they can make a dish.
    Please use a concise and professional tone with the user.

    You are able to search this database according to the following dimensions:
    - dish_name - The name of the dish.
    - tags - Tags that describe the dish. These cover meal types (e.g. dinner), dish types (e.g. soup, salad, sandwich), flavor profiles (e.g. spicy, sweet), cuisines (e.g. Indian, Mexican), and cooking styles or equipment (e.g. Pressure cooker, easy)
    - shopping_list - This is of the ingredients that go in the dish.

    Your job is to decide which of the above dimensions you would like to search (one or many) and then to decide what your search query terms for that dimension should be.
    The system will take your decision and do a similarity search with embeddings of your search queries. Use your knowledge of cooking and food to generate relevant tags, dish names or ingredients to find a matching recipe.
    In order to generate tags, it might help for you to generate a list of adjectives that are similar to the descriptions provided by the user.

    For example, if the user is looking for desserts with walnuts and brown sugar you would decide to search the tags and shopping_list dimensions. You would return a JSON as below:
    {{
        'dish_name':[],
        'tags':['dessert', sweet],
        'shopping_list':['walnuts', 'brown sugar']
    }}

    Or if the user is looking for Coq Au Vin or similar dishes, you would return a JSON as below
    {{
        'dish_name':['Coq Au Vin],
        'tags':['stew', 'braised', 'savory', 'hearty', 'rustic'],
        'shopping_list':['red wine']
    }}
    
    You do not need to explicitly defined a search dimension. The system knows you wish to search a given dimension if the list of query terms has more than one item. 
    
    Restrictions:
    Do not add ingredients to the shopping list that the user has not specified.
    Do not search for a specific dish name unless the user has specified it.

    User input:
    {user_input}
    """
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,  # replace with the model name you're using
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature':query_temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        print(result['response'])
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}
    
def query_vectors(query_params, collection_name='Recipe', num_matches=5):

    headers = {
    "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
    }
    client = weaviate.connect_to_local(headers=headers)
    collection = client.collections.get(collection_name)

    searched_dimensions = [dim for dim, terms in query_params.items() if len(terms) > 0]

    recipe_distances = {}
    for dimension, query_terms in query_params.items():
        if len(query_terms)>0:
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
    combined_scores = {}

    for recipe_uuid, distances in recipe_distances.items():
        # Only use dimensions that were actually searched
        total_distance = 0
        for dim in searched_dimensions:
            total_distance += distances.get(dim, 1)  # If dimension is missing, assign distance of 1

        combined_scores[recipe_uuid] = total_distance / len(searched_dimensions)

    return combined_scores

def retrieve_results(combined_scores, top_n=3, json_dir='data/raw/json/'):
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
    
    # Get the chat history from session state, but limit the number of past messages to avoid too large a context
    chat_history = st.session_state.get('chat_history', [])
    formatted_chat_history = ""

    # Only include the last N messages to avoid token limits
    max_messages = 10  # Adjust as needed based on token limits
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
            # Handle non-streaming case
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
    
    # Get the chat history from session state, but limit the number of past messages to avoid too large a context
    chat_history = st.session_state.get('chat_history', [])
    formatted_chat_history = ""

    # Only include the last N messages to avoid token limits
    max_messages = 10  # Adjust as needed based on token limits
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
            # Handle non-streaming case
            data = response.json()
            print(data['response'])
            return data['response']
        else:
            return f"Error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the server. Details: {e}"

def get_bot_response(message, model: str = 'llama3.1', temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):
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
            1. run_recommendation_engine(user_desire="<YOUR INPUT>") - Triggers a pipeline that provides recommendations based on user input. Only trigger this function if the user asks you for recommendations in their most recent message. The user_desires is a positional input that is a string. It should be a summary of what the user is looking for including flavor profile, cuisine, type of equipment (e.g. pressure cooker), meal type (e.g. lunch), food type (e.g. soup, salad), ingredients, etc. The more descriptive the better.
            2. find_single_recipe(user_desire="<YOUR INPUT>") - Triggers a pipeline to search for a single known recipe. Trigger this function when the user is asking you to find a specific recipe that is known to exist in the database. The "name" field is likely the key to searching. The user_desires is a positional input that is a string. It should be a summary of what the user is looking for including flavor profile, cuisine, type of equipment (e.g. pressure cooker), meal type (e.g. lunch), food type (e.g. soup, salad), ingredients, etc. The more descriptive the better.
            3. show_recipe(recipe_uuid="<YOUR INPUT>") - Displays the entire recipe for the user to read and respond to. Takes the UUID of the recipe they want to see. If a user asks you to "show" the a recipe, then they likely want this function.
            4. edit_recipe(uuid="<YOUR INPUT>", changes_to_make="<YOUR DESCRIPTION OF CHANGES>") - Updates the record of a recipe in the database. You can edit/update specific fields by describing in detail the changes and fields to make.
            5. run_processing_pipeline(source_type=<url or html or img, url=str) - This function is to add a new recipe to the database. If the user wants to add a recipe, specify what the source type is. source_type can either be 'url', 'img' or 'html'. If the source_type is 'url', provide the url as a string.
             
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
            function_call = extract_function_call(content)
            
            if function_call:  # If a function call was detected
                print(f"Executing function '{function_call['function_name']}' with args: {function_call['arguments']}")
                function_result = handle_function_call(function_call)
                for chunk in simulate_stream(function_result):
                    yield chunk
            else:
                for chunk in simulate_stream(content):
                    yield chunk

        else:
            yield f"Error: Received status code {response.status_code} from the bot server."
    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to the bot server. Details: {e}"

def run_recommendation_engine(*, user_desire: str, model: str = 'llama3.1', query_temp: float = 0.9, summary_temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):
    query_params = define_query_params(user_input=user_desire, query_temp=query_temp)
    dists, dims = query_vectors(query_params)
    scores = score_query_results(dists, dims)
    results = retrieve_results(scores)
    summary = summarize_results(user_desire, results, stream=stream, temp=summary_temp)
    st.session_state['most_recent_query'] = results
    return summary

def find_single_recipe(*, user_desire: str, model: str = 'llama3.1', query_temp: float = 0.9, summary_temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = False):
    query_params = define_query_params(user_input=user_desire, query_temp=query_temp)
    dists, dims = query_vectors(query_params)
    scores = score_query_results(dists, dims)
    results = retrieve_results(scores)
    summary = summarize_single_search(user_desire, results, stream=stream, temp=summary_temp)
    st.session_state['most_recent_query'] = results
    return summary

def extract_function_call(message_content: str) -> dict:
    try:
        # Step 1: Extract JSON block from the assistant's message
        match = re.search(r'(\{"function_name"\s*:\s*".+?",\s*"arguments"\s*:\s*\{.*?\}\})', message_content, re.DOTALL)
        if match:
            json_block = match.group(1).strip()  # Extract the JSON portion and strip whitespace
            
            # Step 2: Parse the JSON into a Python dictionary
            function_call = json.loads(json_block)
            
            # Step 3: Ensure the necessary keys are present
            if "function_name" in function_call and "arguments" in function_call:
                return function_call  # Return the parsed function call
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # If no match is found or an error occurs, return None
    return None

def handle_function_call(function_call: dict) -> str:
    try:
        func_name = function_call["function_name"]
        args = function_call["arguments"]
        
        if func_name in FUNCTION_REGISTRY:
            func = FUNCTION_REGISTRY[func_name]
            try:
                result = func(**args)  # Call the function with arguments
                print(f"Function '{func_name}' called successfully.")
                return result
            except Exception as e:
                return f"Error calling function '{func_name}': {e}"
        else:
            return f"Error: Function '{func_name}' is not in the registry."
    except Exception as e:
        return f"Error handling function call: {e}"

def simulate_stream(text):
    # comment: split text into words
    words = text.split(" ")
    # comment: for each word, yield the word with a random short delay
    for word in words:
        # comment: yield word
        yield word + " "
        # comment: sleep for a short random interval between 0.01 and 0.05 seconds
        time.sleep(random.uniform(0.01, 0.1))

def define_update_params(changes_to_make: str, uuid: str, model: str = 'llama3.1', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    
    prompt = f"""
    You are an expert chef and recipe writer. You are allowing a user to interact with a database of recipes.
    Please use a concise and professional tone with the user.

    You are able to update fields in the database. In the database, each recipe is a JSON file. The name of the recipe is it's UUID.
    You are being called because the user wants to edit a field for a given recipe. Your job is to take the user's input and generate the input arguments to the function to update a field.

    Below is the template structure of each recipe:

    {{
        "dish_name": str(), # Name of the dish
        "ingredients": dict(), # A dictionary of ingredients in the recipe with quantities and preparations
        "date_added": str(), # Date added to database
        "cooking_steps": list(str()) # A list of the cooking steps, stored as strings
        "active_time": int(), # The amount of time in minutes of active time
        "total_time": int(), # The total time from start to finish to make the meal
        "source_name": str(), # The name of the source (NYT Cooking, Kitchn, Mob, Instagram)
        "author": str(), # The name of the person who developed the recipe
        "shopping_list": list(str()), # The list of ingredients in the recipe, stripped of quantities and preparations
        "tags": list(str()), # A list of keywords describing the dish
        "recipe_notes": list(str()), # A list of notes for additional guidance, alternatives, and tips for making the dish
        "user_notes" : str(), # Notes that the user has supplied
        "user_rating": float(), # The user's numerical rating of the dish
        "cooked_already": bool() # Whether the user has made the dish before
        "uuid": str() # UNIQUE IDENTIFIER FOR THE RECIPE
    }}
    
    Below this text you will recieve three things:
    
    1. The recipe to be edited (in it's current form).
    2. The user input

    Return a JSON formatted as below:

    {{
        "uuid":{uuid}, # Pull from existing recipe
        "update_params": dict()
    }}

    **NEVER CREATE NEW KEYS. Only update existing keys in the existing JSON structure. Your keys in update_params must be an existing key in the recipe JSON. Use your best judgement.

    Keep in mind, that if you are appending an item to a list, the new value must be the old list + new item.
    For example, if the user wants to add the tag "Spicy" and the existing tags are ["American", "Easy"],
    then you should pass the entire new list of ["American", "Easy", "Spicy"]

    Recipe:
    {st.session_state['selected_recipe']}

    User input:
    {changes_to_make}
    """
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,  # replace with the model name you're using
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature':query_temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        print(result['response'])
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}

def update_weaviate_record(update_params: dict, uuid: str, class_name: str = "Recipe"):
    
    """
    Updates a recipe in the Weaviate database using the update parameters.

    Args:
        update_params (dict): A dictionary of the fields to update and their new values.
        uuid (str): The UUID of the recipe to be updated.
        class_name (str): The Weaviate class name (e.g., "Recipe").

    Returns:
        dict: The API response from Weaviate.
    """
    try:
        # Connect to the local Weaviate instance
        headers = {
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
        }
        client = weaviate.connect_to_local(headers=headers)
        collection = client.collections.get('Recipe')

        # Perform the update
        collection.data.update(
            uuid=update_params['uuid'],
            properties=update_params['update_params']
        )
        client.close()
        print(f"Successfully updated Weaviate record for UUID {uuid}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}
    
def update_local_json_record(update_params: dict, uuid: str, data_path: str = "data/raw/json"):
    """
    Updates a recipe stored in a local JSON file using the update parameters.

    Args:
        update_params (dict): A dictionary of the fields to update and their new values.
        uuid (str): The UUID of the recipe to be updated.
        data_path (str): The path to the directory where the recipe JSON files are stored.

    Returns:
        dict: The result of the update, either success message or error details.
    """
    try:
        # Build the path to the JSON file
        file_path = os.path.join(data_path, f"{uuid}.json")

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found for UUID: {uuid} at {file_path}")

        # Load the existing JSON file
        with open(file_path, 'r') as file:
            recipe_data = json.load(file)

        # Update the fields specified in update_params
        for key, value in update_params['update_params'].items():
            # Otherwise, replace the value directly
            recipe_data[key] = value

        # Save the updated data back to the same JSON file
        with open(file_path, 'w') as file:
            json.dump(recipe_data, file, indent=4)

        print(f"Successfully updated local JSON record for UUID {uuid}")
        return {"status": "success", "message": f"Updated JSON file for UUID {uuid}"}

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return {"error": f"File not found: {e}"}

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return {"error": f"Error parsing JSON file: {e}"}

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}

def show_recipe(*, recipe_uuid: str): 
    st.session_state['selected_recipe_uuid'] = recipe_uuid
    return 'Sure! Take a look at this.'

def edit_recipe(*, uuid: str, changes_to_make: str, model: str = 'llama3.1', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    update_params = define_update_params(changes_to_make=changes_to_make, uuid=uuid)
    weaviate_response = update_weaviate_record(update_params=update_params, uuid=uuid)
    json_response = update_local_json_record(update_params=update_params, uuid=uuid)
    return 'The recipe has been updated!'

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
    already_cooked = recipe_data.get('cooked_already', False)

    # add checkmark if already cooked
    if already_cooked:
        recipe_title += " ✅"

    st.subheader(f"{recipe_title}")

    # extract relevant information to display in a single line
    source_name = recipe_data.get('source_name', 'Unknown Source')
    user_rating = recipe_data.get('user_rating', None)
    active_time = recipe_data.get('active_time', None)
    total_time = recipe_data.get('total_time', None)

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

def resize_and_encode_images(image_paths: list, max_dimension: int = 1120) -> list:
    """
    Resize images so that the largest dimension is `max_dimension` and encode them as base64 strings.
    
    Args:
        image_paths (list): List of paths to the images to be resized.
        max_dimension (int): Maximum dimension (height or width) for the resized images.
        
    Returns:
        list: List of base64-encoded images.
    """
    encoded_images = []

    for image_path in image_paths:
        try:
            print(f"Processing image: {image_path}")
            
            # Resize image
            with Image.open(image_path) as image:
                original_size = image.size
                max_side = max(original_size)
                
                if max_side > max_dimension:
                    ratio = max_dimension / max_side
                    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                    print(f"Resizing image from {original_size} to {new_size}")
                    image = image.resize(new_size, Image.LANCZOS)

                # Convert to bytes and encode as base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")  # Save the image as PNG in memory, not raw bytes
                image_bytes = buffered.getvalue()
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                encoded_images.append(encoded_image)
                        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if not encoded_images:
        print("No images were successfully processed.")
    
    return encoded_images


def extract_text_from_images(
        image_paths: list, 
        model: str = 'llama3.2-vision', 
        server_url: str = "http://192.168.0.19:11434", 
        temp: float = 0.3
    ) -> dict:
    """
    Extract structured recipe information directly from images using the Llama 3.2 Vision model.

    Args:
        image_paths (list): A list of file paths to the images.
        model (str): The Llama Vision model to use.
        server_url (str): The URL of the server where the LLM is running.
        temp (float): Temperature for the LLM.
        
    Returns:
        dict: Extracted recipe information as a structured JSON object.
    """
    
    # Step 1: Resize images and encode them in base64
    encoded_images = resize_and_encode_images(image_paths)
    if not encoded_images:
        print("No images available for extraction.")
        return {}

    # Step 2: Construct the Llama Vision prompt
    prompt = f"""
    You are an expert chef and recipe writer.
    Perform OCR and extract the structured recipe information from the following image.
    Extract the exact text and respect borders and columns so text that is continuous.
    """

    extracted_text = []

    for enc_img in encoded_images:
        # Step 3: Prepare the API payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [enc_img]  # This is now base64-encoded images
                }
            ],
            "stream": False,
            "options": {
                'temperature': temp,
                'num_ctx': 32768
            }
        }

        try:
            print('Requesting recipe extraction from images via /api/chat...')
            response = requests.post(f"{server_url}/api/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            output = result.get("message", {}).get("content", "")
            extracted_text.append(output)

        except Exception as e:
            print(f"Error communicating with Llama 3.2 Vision server: {e}")
            return {}
        
    return extracted_text

def run_processing_pipeline(
        source_type: str,
        url: str,
        save_dir: str = 'data/raw/json'
        ):
    
    if source_type=="url":
        recipe = process_recipe(url=url, recipe_temp=0.4, process_temp=0.3, tag_temp=0.5, model='llama3.1', tag_model='qwen2.5', post_process=False)
        add_weaviate_record(recipe_json=recipe)
        
        recipe_save_path = os.path.join(save_dir, f"{recipe['uuid']}.json")
        with open(recipe_save_path, 'w', encoding='utf-8') as f:
            json.dump(recipe, f, indent=4)
        
        print(f"Successfully processed and saved: {recipe_save_path}")

    return f"Successfully processed and saved: {recipe_save_path}"

def add_weaviate_record(
        recipe_json: dict,
        collection: str = 'Recipe'
    ):
    try:
        recipe_obj = {
            "dish_name": recipe_json["dish_name"],
            "shopping_list": recipe_json["shopping_list"],
            "tags": recipe_json["tags"],
            "source_name": recipe_json["source_name"],
            "author_name": recipe_json["author"],
            "user_rating": None,
            "cooked_already": False,
            "user_notes": recipe_json["user_notes"],
            "active_time": int(recipe_json["active_time"]),
            "total_time": int(recipe_json["total_time"]),
            "cooking_steps": recipe_json["cooking_steps"],
            "recipe_notes": recipe_json["recipe_notes"],
        }

        headers = {
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
        }

        client = weaviate.connect_to_local(headers=headers)
        recipes = client.collections.get(collection)
        uuid = recipes.data.insert(
            properties=recipe_obj,
            uuid=recipe_json['uuid']
        )

        client.close()

        print(f"Recipe {uuid} successfully added to hardtack-weaviate.")
    except Exception as e:
        uuid = recipe_json['uuid']
        print(f'Recipe {uuid} could not be added due to error: {e}')
        


###### FUNCTION REGISTRY #####
# Create a function registry (dictionary) for the chatbot
FUNCTION_REGISTRY = {
    "run_recommendation_engine": run_recommendation_engine,
    "find_single_recipe": find_single_recipe,
    "show_recipe": show_recipe,
    "edit_recipe":edit_recipe,
    "run_processing_pipeline": run_processing_pipeline
}