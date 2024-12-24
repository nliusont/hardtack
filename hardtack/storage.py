# database.py

import json
import os
import weaviate
from io import BytesIO
from google.cloud import storage
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import requests
import streamlit as st
from openai import OpenAI

def define_update_params(changes_to_make: str, uuid: str, model: str = 'openai', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
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

    **NEVER CREATE NEW KEYS. Only update existing keys in the existing JSON structure. Your keys in update_params must be a verbatim existing key in the recipe JSON and must not be formatted with any markdown. Use your best judgement.

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
                    "stream": False,
                    "options": {
                        "temperature": query_temp,
                        "num_ctx": 32768
                    }
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data['response']
            else:
                return f"Error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the server. Details: {e}"

def add_weaviate_record(
        recipe_json: dict,
        collection: str = 'Recipe',
        db: str = 'remote'
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

def update_weaviate_record(update_params: dict, uuid: str, class_name: str = "Recipe", db: str = 'remote'):
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

def retrieve_file_from_gcs(blob_name, bucket_name='hardtack-bucket'):
    """
    Retrieves a file from GCS and loads it into memory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the file in GCS.

    Returns:
        BytesIO: File content loaded into memory.
    """
    # Initialize the GCS client
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    storage_client = storage.Client()

    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file content into memory
    file_content = BytesIO()
    blob.download_to_file(file_content)
    file_content.seek(0)  # Reset the file pointer to the beginning

    storage_client.close()

    print(f"File {blob_name} retrieved from GCS into memory.")
    return file_content

def save_to_gcs(blob_name, content=str, bucket_name='hardtack-bucket', content_type=None):

    """
    Saves a file (JSON, HTML, or image) to Google Cloud Storage.

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Path to save the file in GCS.
        content (str, dict, or bytes): Content to save. Can be a JSON dict, HTML string, or image bytes.
        content_type (str): MIME type of the content (optional). If not provided, it will be inferred.

    Returns:
        None
    """
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

        # Initialize the GCS client
        storage_client = storage.Client()

        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Handle different content types
        if isinstance(content, dict):
            # JSON content
            prefix = 'recipe/'
            content_type = content_type or "application/json"
        elif isinstance(content, str):
            # Text content
            prefix = 'txt/'
            content_type = content_type or "text/plain" 
        elif isinstance(content, bytes):
            # Image or binary content
            prefix = 'image/'
            content_type = content_type or "application/octet-stream"  # Default to binary
        else:
            raise ValueError("Unsupported content type. Use dict (JSON), str (HTML), or bytes (image).")
        
        
        full_blob_name = f"{prefix}{blob_name}"

        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(full_blob_name)

        # Upload the content
        blob.upload_from_string(
            json.dumps(content) if isinstance(content, dict) else content,
            content_type=content_type
        )

        storage_client.close()

        print(f"File saved to GCS at {bucket_name}/{blob_name} with content type {content_type}.")
    except Exception as e:
        raise Exception(f"Error saving file to GCS: {e}")
    
def update_gcs_json_record(update_params: dict, uuid: str, bucket_name: str = "hardtack-bucket", gcs_path_prefix: str = "recipe"):

    """
    Update an existing JSON record stored in Google Cloud Storage.

    Args:
        update_params (dict): The fields to update in the JSON record.
        uuid (str): The UUID of the JSON record to update.
        bucket_name (str): The name of the GCS bucket where the JSON file is stored.
        gcs_path_prefix (str): The path prefix in the bucket where the JSON files are stored.

    Returns:
        dict: The result of the update, either success message or error details.
    """
    try:

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

        # Initialize the GCS client
        storage_client = storage.Client()

        # Construct the full blob path
        blob_name = f"{gcs_path_prefix}/{uuid}.json"

        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if the file exists in GCS
        if not blob.exists():
            raise FileNotFoundError(f"No file found for UUID: {uuid} at {blob_name}")

        # Download the existing JSON content
        file_content = blob.download_as_text()
        recipe_data = json.loads(file_content)

        # Update the fields specified in update_params
        for key, value in update_params['update_params'].items():
            recipe_data[key] = value

        # Convert the updated JSON to a string and re-upload it to GCS
        blob.upload_from_string(
            json.dumps(recipe_data, indent=4),
            content_type="application/json"
        )

        print(f"Successfully updated GCS JSON record for UUID {uuid}")
        storage_client.close()
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
     
def delete_weaviate_object(uuid=str, collection: str = 'Recipe'):
    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key))

    collection = client.collections.get(collection)
    collection.data.delete_by_id(uuid)

    client.close()

    print(f'Successfully deleted object: {uuid}')