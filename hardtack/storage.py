# database.py

import json
import os
import weaviate
import requests
import streamlit as st

def define_update_params(changes_to_make: str, uuid: str, model: str = 'llama3.1', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    """
    Generate the parameters needed to update a recipe in the database based on user input.

    Args:
        changes_to_make (str): The description of changes to be made to the recipe.
        uuid (str): The UUID of the recipe to update.
        model (str): The language model to use for generating the update parameters.
        query_temp (float): The temperature for query generation.
        server_url (str): The URL of the server for generating the update parameters.

    Returns:
        dict: A dictionary containing the update parameters.
    """
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

    Keep in mind, that if the user wants you to add an imte to a list. You should pass the entire new list (old list + new item) as an argument.
    For example, if the user wants to add the tag "Spicy" and the existing tags are ["American", "Easy"],
    then you should pass the entire new list of ["American", "Easy", "Spicy"]

    Similarly, if you are removing an item(s) from a list, then pass the new list without the unwanted items in it.

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
                    'temperature': query_temp,
                    "num_ctx": 32768
                },
            }
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])
    
    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}

def add_weaviate_record(
        recipe_json: dict,
        collection: str = 'Recipe'
    ):
    """
    Add a new recipe to the Weaviate database.

    Args:
        recipe_json (dict): The recipe data to be added to the database.
        collection (str): The name of the collection in the Weaviate database.

    Returns:
        None
    """
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

def update_weaviate_record(update_params: dict, uuid: str, class_name: str = "Recipe"):
    """
    Update an existing recipe in the Weaviate database.

    Args:
        update_params (dict): The fields to update in the recipe.
        uuid (str): The UUID of the recipe to update.
        class_name (str): The class name in the Weaviate database (default is "Recipe").

    Returns:
        dict: The response from the Weaviate API.
    """
    try:
        headers = {
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')
        }
        client = weaviate.connect_to_local(headers=headers)
        collection = client.collections.get(class_name)

        collection.data.update(
            uuid=update_params['uuid'],
            properties=update_params['update_params']
        )
        client.close()
        print(f"Successfully updated Weaviate record for UUID {uuid}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}

def update_local_json_record(update_params: dict, uuid: str, data_path: str = "data/json"):
    """
    Update an existing recipe stored locally in a JSON file.

    Args:
        update_params (dict): The fields to update in the recipe.
        uuid (str): The UUID of the recipe to update.
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
