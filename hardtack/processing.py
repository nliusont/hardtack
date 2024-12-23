# app/processing.py

import json
import requests
import uuid
import os
from openai import OpenAI
from datetime import datetime
from hardtack.acquisition import extract_text_from_images, fetch_html_from_url, parse_html, clean_text


def extract_recipe(
        text: str, 
        model: str = 'openai',
        recipe_temp: float = 0.3, 
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
        api_key (str): API key for OpenAI models if used.

    Returns:
        dict: A structured recipe with details like ingredients, cooking steps, and tags.
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
        if model == 'openai':

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=recipe_temp
            )
            
            content = response.choices[0].message.content
            content = content.replace('```json', '').replace('```', '')
            output = json.loads(content)

        else:
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

        return output

    except Exception as e:
        print(f"Error communicating with LLM server: {e}")
        return {}



def interpret_recipe(text: str, model: str = 'openai', temp: float = 0.5, server_url: str = "http://192.168.0.19:11434") -> dict:
    """
    Extract tags and notes for a recipe using a model.

    Args:
        text (str): The raw recipe text.
        model (str): The model used for extracting tags.
        temp (float): Temperature setting for the model.
        server_url (str): The URL of the server.
        api_key (str): API key for OpenAI models if used.

    Returns:
        dict: A dictionary with 'tags' and 'recipe_notes'.
    """
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
        if model == 'openai':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=temp
            )

            content = response.choices[0].message.content
            content = content.replace('```json', '').replace('```', '')
            return json.loads(content)

        else:
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
        print(f"Error communicating with LLM server: {e}")
        return {}


def post_process_recipe(recipe: str, text: str, model: str = 'openai', temp: float = 0.2, server_url: str = "http://192.168.0.19:11434"):
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
        html_files: list = None, 
        model: str = 'openai', 
        tag_model: str = 'openai',
        recipe_temp: float = 0.4, 
        tag_temp: float = 0.5, 
        process_temp: float = 0.3,
        save_dir: str = 'data/raw/', 
        post_process: bool = False):
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
    provided_sources = sum([url is not None, images is not None, html_files is not None])
    if provided_sources != 1:
        raise ValueError("You must provide exactly one of 'url', 'images', or 'html'.")

    identifier = str(uuid.uuid4())

    if url:
        html = fetch_html_from_url(url)
        cleaned_text = parse_html([html])
        file_path = os.path.join(f'{save_dir}parsed_html/', f'{identifier}.txt')
        with open(file_path, "wb" if isinstance(cleaned_text, bytes) else "w") as f:
            f.write(cleaned_text)
    elif images:
        cleaned_text = extract_text_from_images(images, uuid=identifier, model=model)
    elif html_files:
        scraped_text = parse_html(html_files)
        cleaned_text = clean_text(scraped_text)

    recipe = extract_recipe(text=cleaned_text, recipe_temp=recipe_temp, model=model)
    tags_and_notes = interpret_recipe(text=cleaned_text, model=tag_model, temp=tag_temp)
    recipe['tags'] = tags_and_notes['tags']
    recipe['recipe_notes'] = tags_and_notes['recipe_notes']
    print('Recipe processed.')
    if post_process:
        processed_recipe = post_process_recipe(recipe, cleaned_text, temp=process_temp, model=model)
    else:
        processed_recipe = recipe

    processed_recipe['date_added'] = datetime.now().strftime('%Y-%m-%d')

    if url:
        processed_recipe['url'] = url

    processed_recipe['uuid'] = identifier
    processed_recipe['user_notes'] = str()
    processed_recipe['user_rating'] = None
    processed_recipe['cooked_already'] = bool()

    return processed_recipe
