# app/utils.py

import time
import random
import json
import re
from hardtack.agent import FUNCTION_REGISTRY

def simulate_stream(text):
    """
    Simulate streaming of text by splitting it into words and yielding them one by one with a small delay.

    Args:
        text (str): The text to be streamed.

    Yields:
        str: One word at a time, simulating the streaming process with delays.
    """
    # Split the text into words
    words = text.split(" ")

    # For each word, yield the word with a random short delay
    for word in words:
        yield word + " "
        # Sleep for a short random interval between 0.01 and 0.05 seconds
        time.sleep(random.uniform(0.01, 0.1))


def extract_function_call(message_content: str) -> dict:
    """
    Extract a function call from a message's content, if present. This function looks for specific JSON patterns
    that denote function calls.

    Args:
        message_content (str): The content of the message that may contain a function call.

    Returns:
        dict: A dictionary representing the function call, or None if no function call is found.
    """
    try:
        # Look for a JSON block that starts with {function_name: ...}
        match = re.search(r'(\{"function_name"\s*:\s*".+?",\s*"arguments"\s*:\s*\{.*?\}\})', message_content, re.DOTALL)
        if match:
            json_block = match.group(1).strip()  # Extract and strip the JSON portion
            
            # Parse the JSON into a Python dictionary
            function_call = json.loads(json_block)
            
            # Ensure the necessary keys are present
            if "function_name" in function_call and "arguments" in function_call:
                return function_call  # Return the parsed function call
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # If no match is found or an error occurs, return None
    return None


def handle_function_call(function_call: dict) -> str:
    """
    Handle the function call by calling the appropriate function from the registry.

    Args:
        function_call (dict): The parsed function call, containing the function name and arguments.

    Returns:
        str: The result of the function call, or an error message if the function call fails.
    """
    try:
        func_name = function_call["function_name"]
        args = function_call["arguments"]
        
        # Check if the function exists in the function registry
        if func_name in FUNCTION_REGISTRY:
            func = FUNCTION_REGISTRY[func_name]
            try:
                result = func(**args)  # Call the function with the provided arguments
                print(f"Function '{func_name}' called successfully.")
                return result
            except Exception as e:
                return f"Error calling function '{func_name}': {e}"
        else:
            return f"Error: Function '{func_name}' is not in the registry."
    except Exception as e:
        return f"Error handling function call: {e}"
