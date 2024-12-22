# app/processing.py

import json
import requests
import uuid
from datetime import datetime
from app.acquisition import extract_text_from_images, fetch_html_from_url
from app.utils import simulate_stream
from app.storage import define_update_params


def extract_recipe(
        text: str, 
        model: str = 'llama3.1',
        tag_model: str = 'qwen2.5', 
        recipe_temp: float = 0.3, 
        tag_temp: float = 0.75, 
        server_url: str = "http://192.168.0.19:11434") -> dict:
    """
    Extract structured recipe information from raw text using an LLM (Language Model).

    Args:
        text (str): The raw text extracted from an image or webpage.
        model (str): The model used to extract the recipe details.
        tag_model (str): The model used to extract recipe tags.
        recipe_temp (float): Temperature for recipe extraction.
        tag_temp (float): Temperature for tag extraction.
        server_url (str): The URL of the server where the model is hosted.

    Returns:
        dict: A structured recipe with details like ingredients, cooking steps, and tags.
    """
    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is a recipe for a dish. Please extract the below information from the text and store it as indicated:
    - dish_name: The name of the dish, not the name of the recipe.
    - ingredients: Stored as a dictionary of ingredients with quantities and preparation details.
    - shopping_list: A list of ingredients without quantities.
    - cooking_steps: List of cooking steps.
    - active_time: Active cooking time in minutes.
    - total_time: Total cooking time in minutes.
    - source_name: Name of the source (e.g., NYTimes, Blue Apron).
    - author: The author of the recipe, if available.

    Return the result as a JSON object.
    Text for extraction: {text}
    """
    print('Requesting recipe extraction...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': recipe_temp,
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
        print(f"Error communicating with LLM server: {e}")
        return {}


def interpret_recipe(text: str, model: str = 'qwen2.5', temp: float = 0.5, server_url: str = "http://192.168.0.19:11434") -> list:
    """
    Extract tags and notes for a recipe using a model.

    Args:
        text (str): The raw recipe text.
        model (str): The model used for extracting tags.
        temp (float): Temperature setting for the model.
        server_url (str): The URL of the server.

    Returns:
        dict: A dictionary with 'tags' and 'recipe_notes'.
    """
    prompt = f"""
        You are an expert chef and recipe writer.
        Following this message is a recipe for a dish. Your task is to produce:
        1. A list of tags that describe the dish’s cuisine, flavor profile, meal type, and more.
        2. Practical cooking tips that help someone prepare the dish.
        
        Text for extraction:
        {text}
    """
    print('Requesting recipe interpretation...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': temp,
                    "num_ctx": 32768},
            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}


def post_process_recipe(recipe: str, text: str, model: str = 'llama3.1', temp: float = 0.2, server_url: str = "http://192.168.0.19:11434"):
    """
    Clean and refine the extracted recipe details to ensure consistency and standardization.

    Args:
        recipe (str): The extracted recipe data.
        text (str): The raw text used to extract the recipe.
        model (str): The language model to use.
        temp (float): The temperature setting for the model.
        server_url (str): The server URL for processing.

    Returns:
        dict: The cleaned and standardized recipe.
    """
    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is an extracted recipe for a dish. Your job is to clean and refine the recipe as per the instructions below:
    - Standardize ingredient names, quantities, and preparation methods.
    - Ensure cooking steps are in a clear, easy-to-follow format.
    - Review the tags and ensure they make sense for the dish.
    - Clean the shopping list and remove unnecessary information.
    
    Recipe:
    {recipe}

    Raw text:
    {text}
    """
    print('Cleaning up recipe...')
    try:
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': temp,
                    "num_ctx": 32768},
            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}


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
    """
    Process a new recipe from a URL, images, or HTML file and extract the structured data.

    Args:
        url (str): The URL of the recipe.
        images (list): List of images for OCR extraction.
        html_file (str): An HTML file containing the recipe.
        model (str): The model used for extracting recipe data.
        tag_model (str): The model used for extracting recipe tags.
        recipe_temp (float): Temperature for recipe extraction.
        tag_temp (float): Temperature for tag extraction.
        process_temp (float): Temperature for post-processing.
        save_dir (str): Directory where the processed recipe will be saved.
        post_process (bool): Whether to post-process the recipe after extraction.

    Returns:
        dict: The processed recipe with all structured information.
    """
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

    processed_recipe['date_added'] = datetime.now().strftime('%Y-%m-%d')

    if url:
        processed_recipe['url'] = url
    if images:
        processed_recipe['image_paths'] = images

    processed_recipe['uuid'] = identifier
    processed_recipe['user_notes'] = str()
    processed_recipe['user_rating'] = None
    processed_recipe['cooked_already'] = bool()

    return processed_recipe
