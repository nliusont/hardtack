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

from dotenv import load_dotenv
load_dotenv()



def extract_text_from_images(image_paths):
    """
    processes multiple images and combines the extracted text from each.

    Args:
        image_paths (list): a list of file paths to the images.

    Returns:
        str: the combined extracted text from all images.
    """
    combined_text = ""

    for image_path in image_paths:
        try:            
            # load the image using PIL (pillow-heif will allow HEIC/HEIF to be loaded)
            print(f"Attempting to load image with PIL: {image_path}")
            image = Image.open(image_path)
            image.seek(0) # for iOS live pictures
            image = image.convert('RGB')
            
            # apply OCR to extract text from the image
            custom_config = r'--psm 1'
            extracted_text = pytesseract.image_to_string(image, config=custom_config)
            
            # combine extracted text
            combined_text += extracted_text + "\n\n"  # separate each image's text with a newline
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return combined_text

def extract_recipe_info_ollama(text: str, model: str = 'llama3.2', recipe_temp: float = 0.3, tag_temp: float = 0.75, server_url: str = "http://192.168.0.19:11434") -> dict:
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
    - dish_name - A title for the dish
    - ingredients - stored as a dictionary of ingredients in the form k:[v1, v2] such as <ingredient_name>: [<quantity>, <preparation>]. For example, 'white onion': ['1 cup', 'diced'], 'all purpose flour':['1 cup'], 'walnuts':['2 cups', 'chopped']. If there is no preparation, skip it. The ingredients will be used as search indices, so they should be standardized. If the dish provides both imperial and metric quantities, only choose the imperial units.
    - shopping list - stored as a list of ingredients. Do not include quantities or preparations. For example, 'poblano peppers, roughly chopped, seeds and stems discarded', should be stored as 'poblano peppers'. 
    Or 'loosely packed fresh cilantro leaves and fine stems' should be stored as 'cilantro'. Essentially, it should be the keys of the ingredients dictionary.
    - cooking_steps - stored as list
    - active_time - amount of time actively cooking in minutes
    - total_time - total cook time from start to finish in minutes
    - dish_description - a 20 to 30 word description of the dish that summarizes it's flavor, style, and recipe
    - source_name - NYTimes, Hellofresh, Blue Apron, Serious Eats, etc.
    - author - The name of the person who developed the recipe, if available.

    Return the result as a JSON object.

    Text for extraction: 
    {text}
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
                    'temperature':recipe_temp,
                    "num_ctx": 32768},

            }
        )
        response.raise_for_status()
        result = response.json()
        output = json.loads(result['response'])
        tags_and_notes = extract_tags_and_notes(text=text, model=model, temp=tag_temp, server_url=server_url)
        output['tags'] = tags_and_notes['tags']
        output['recipe_notes'] = tags_and_notes['recipe_notes']

        return output

    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return {}
    
def extract_tags_and_notes(text: str, model: str = 'llama3.2', temp: float = 0.75, server_url: str = "http://192.168.0.19:11434") -> list:
    '''Recipe extract requires low temp, but tag extraction requires high temp. So send a separate call for tags.'''

    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is a recipe for a dish. Please generate tags and notes for the dish. The tags will be used as search indices, so they should be standardized.
    Generate at least ten tags.
    The tags will be used by a user to help them search for recipes to cook for home meals. The tags should be practical and useful for narrowing a list of possible recipes.
    Examples of tags include: Spicy, Mexican, Italian, Savory, Sweet, Warm, Soup, Stir-fry, Bread, Low-carb, Vegetarian, Vegan, Fish, Hearty, Dinner, Lunch, Dessert, Quick, Pressure cooker, Air fryer, Chili, Sandwiches, Comfort food, Easy, Pasta, Salad, Custard, Casserole, etc
    Tags to describe the dish stored as list, e.g., cuisine type, dish type, flavor profile, cook time. The tags should not also be ingredients unless the ingredient is notable and unique. Create at least 10 tags.

    Return the tags as a simple list. For example: ['Spicy', 'Dinner', 'Easy', 'Soup', 'Asian', 'Savory']. Do not return any other text with the list.

    For recipe notes, provide any additional guidance that someone making the recipe should know (e.g. notes on process, tips, instructions, etc). This should be information not included in the cooking steps. It should be information that augments the instructions. DO NOT INCLUDE RECIPE HISTORY.
    Notes can be stored in a simple list.

    Return your output as a JSON. Where the keys are 'tags' and 'recipe_notes' and the values are lists.
    

    Text for extraction: 
    {text}
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

def post_process_recipe(recipe: str, model: str = 'llama3.2', temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    
    prompt = f"""
    You are an expert chef and recipe writer.
    Following this message is a recipe for a dish that was extract from an image or website by an LLM. Your job is to further refine and clean the recipe.
    The purpose of this is to catalogue the recipe as part of a database. Therefore, everything must be standardize and universal.

    Please check and clean the following:
    - dish_name - This should be the name of the food, not the name of the recipe.
    - cooking_steps - This should be a simple list, where each item in the list is a step. There is no need to number or demarcate steps.
    - ingredients - Ensure that the ingredients are stored in one dictionary with each ingredient as a key in form k:[v1, v2] such as <ingredient_name>: [<quantity>, <preparation>]. If the dish provides both imperial and metric quantities, only choose the imperial units.
        - Example: 'all purpose flour':['1 cup'], 'walnut':['2 cups', 'chopped']
    - tags -  Review the list of tags and ensure they make sense and are standardized given you understanding of the recipe and the dish. Add any tags that you think would be valuable to add. Tags should not be ingredients unless the ingredient is rare or unique. Ensure the tag list is purely a list. Ensure that if a dish has meat in it, that it is not tagged vegetarian or vegan. If it has dairy in it, make sure it is not tagged vegan. Aim for roughly ten tags. Err on the side of keeping tags, not deleting them.
    - shopping_list - This is of the ingredients that go in the dish. It should be a simple list, not a dictionary. It is just the name of the ingredients, they do not include preparations or quantities. Essentially, it should be the keys of the ingredients dictionary.
    - recipe_notes - Provide any additional guidance that someone making the recipe should know (e.g. notes on process, tips, instructions, etc). This should be information not included in your cooking steps. DO NOT INCLUDE RECIPE HISTORY.
    The purpose of the shopping list is to provide a standard list of ingredients that can be indexed against. Therefore, review and edit this list so it is standardized.

    Return your cleaned version of the recipe as a JSON in the same structure as it was provided.

    Recipe:
    {recipe}
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
    try:
        # send GET request to the URL
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
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

def process_recipe(*, url: str = None, images: list = None, html_file: str = None, model: str = 'llama3.2', recipe_temp: float = 0.1, tag_temp: float = 0.3, save_dir: str = 'data/raw/'):
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

    recipe = extract_recipe_info_ollama(text=cleaned_text, recipe_temp=recipe_temp, tag_temp=tag_temp, model=model)
    processed_recipe = post_process_recipe(recipe, temp=recipe_temp, model=model)
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
    return processed_recipe

def define_query_params(user_input: str, model: str = 'llama3.2', query_temp: float = 0.3, server_url: str = "http://192.168.0.19:11434"):
    
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
    model: str = 'llama3.2',
    temp: float = 0.6,
    server_url: str = "http://192.168.0.19:11434",
    stream: bool = True
):

    prompt = f"""
        You are an expert chef and recipe writer. You are interacting with a user that is looking for recipes in a database so they can make a dish.
        Following this message is a user query along with a JSON of several recipes that have been returned by a search engine based on the query. The recipes are ordered such that the closest match is first.

        Please review the results and summarize them to the user to help the user decide on which to cook. Please return:
        The name of each dish, a brief description of the dish, a brief description of why the dish matches their query.
        Return your result in concise, friendly, plain language. Do not suggest any recipes that aren't included in the JSON.

        If any of the recipes do not seem to match the search query well, please omit them.

        User input:
        {user_input}

        Search results:
        {results}
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
            if stream:
                # Stream response lines for Streamlit
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_line = json.loads(line)  # Parse the streamed JSON line
                            if "message" in json_line and "content" in json_line["message"]:
                                yield json_line["message"]["content"]
                        except json.JSONDecodeError as e:
                            yield f"Error parsing response line: {line}"
            else:
                # Handle non-streaming case
                data = response.json()
                return data.get("message", {}).get("content", "")
        else:
            yield f"Error: Received status code {response.status_code} from the server."
    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to the server. Details: {e}"

def get_bot_response(message, model: str = 'llama3.2', temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = True):
    try:
        messages = [
            {"role": "system", "content": """
            You are a recipe chat bot. You are an expert home cook and recipe writer.
            You can help the user by answering questions or querying a local database of recipes to assist the user in selecting a dish to make. 
            Respond to the user in a concise and professional manner. If the user wants recommendations but you feel you need more information first, do not hesitate to ask more questions in order to refine their desires and obtain enough information.
            
            If needed, you have access to a single function:
            1. run_query_pipeline(user_desire="<YOUR INPUT>") - Triggers a pipeline that searches for recipes that match a user's request. Only trigger this function if the user asks you for recommendations. The user_desires is a positional input that is a string. It should be a summary of what the user is looking for including flavor profile, cuisine, type of equipment (e.g. pressure cooker), meal type (e.g. lunch), food type (e.g. soup, salad), ingredients, etc. The more descriptive the better.

            When you want to call a function, respond with this exact format and DO NOT RESPOND WITH ANY OTHER TEXT:
            One moment.
            ```json
            {"function_name": "name_of_function", "arguments": {"arg1": "value1", "arg2": "value2"}}
             
            For example, to run function 1, it might look like this:
            One moment.
            ```json
            {"function_name": "run_query_pipeline", "arguments": {"user_desire": "<YOUR INPUT>"}}

            When calling a function, it is crucial that you lead with "One moment." (verbatim) as this is the trigger phrase that tells the system you want a function.

            If the user has not asked for you to search the database or for recommendations, then you can respond normally.
            
             
            Restrictions:
            You can tell the user about your abilities, but do not surface the exact function calls to the user.
            Do not offer up recipes to the user unless they've been returned to you by the database. 
            If you call a function, do not produce any text after your function call json.
            """}
        ]
        for msg_type, msg_text in st.session_state['chat_history']:
            role = "user" if msg_type == "user" else "assistant"
            messages.append({"role": role, "content": msg_text})
        messages.append({"role": "user", "content": message})
        
        response = requests.post(f"{server_url}/api/chat", json={
            "model": model,  # specify the model name
            "messages": messages,
            "stream": stream,  # control streaming behavior
            'options': {
                'temperature': temp,
                "num_ctx": 32768
            }
        }, stream=stream)

        buffer = ""  # Used to accumulate content for function call JSON
        in_function_call = False  # Flag to know if we're streaming a function call
        
        if response.status_code == 200:
            if stream:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_line = json.loads(line)  # convert line to dict
                            if "message" in json_line and "content" in json_line["message"]:
                                message_content = json_line["message"]["content"]
                                
                                # Accumulate the full message if we're in function call mode
                                buffer += message_content
                                
                                if not in_function_call and re.search(r'One moment.', buffer):
                                    in_function_call = True  # We detected the start of a function call
                                    print('in_function_call')
                                    continue
                                
                                if in_function_call:
                                    # Try to extract the function call
                                    if re.search(r'One moment\.\s*```json(.*?)```', buffer, re.DOTALL):
                                        function_call = extract_function_call(buffer)
                                        if function_call:  # If the LLM requests a function call
                                            print(f"Executing function '{function_call['function_name']}' with args: {function_call['arguments']}")
                                            function_result = handle_function_call(function_call)

                                            yield function_result
                                            buffer = ""  # Reset the buffer
                                            in_function_call = False  # Reset the flag
                                else:
                                    yield message_content  # Stream normal content
                        except json.JSONDecodeError as e:
                            yield f"Error parsing response line: {line}"
            else:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                function_call = extract_function_call(content)
                
                if function_call:  # If a function call was detected
                    print(f"Executing function '{function_call['function_name']}' with args: {function_call['arguments']}")
                    function_result = handle_function_call(function_call)
                    yield function_result  # Stream the result
                else:
                    yield content  # Return the content as a normal LLM message
        else:
            yield f"Error: Received status code {response.status_code} from the bot server."
    except requests.exceptions.RequestException as e:
        yield f"Error: Could not connect to the bot server. Details: {e}"

def run_query_pipeline(*, user_desire: str, model: str = 'llama3.2', query_temp: float = 0.9, summary_temp: float = 0.6, server_url: str = "http://192.168.0.19:11434", stream: bool = True):
    query_params = define_query_params(user_input=user_desire, query_temp=query_temp)
    dists, dims = query_vectors(query_params)
    scores = score_query_results(dists, dims)
    results = retrieve_results(scores)
    summary = summarize_results(user_desire, results)
    yield summary

def extract_function_call(message_content: str) -> dict:
    try:
        # Step 1: Extract JSON block from the assistant's message
        match = re.search(r'```json(.*?)```', message_content, re.DOTALL)
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



###### FUNCTION REGISTRY #####
# Create a function registry (dictionary) for the chatbot
FUNCTION_REGISTRY = {
    "run_query_pipeline": run_query_pipeline,
}