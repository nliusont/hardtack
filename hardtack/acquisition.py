from PIL import Image
import base64
import io
import requests
from openai import OpenAI
from fake_useragent import UserAgent
from hardtack.storage import save_to_gcs
from bs4 import BeautifulSoup
import streamlit as st
import re
import os

def extract_text_from_images(
        image_paths: list, 
        model: str = 'openai', 
        server_url: str = "http://192.168.0.19:11434", 
        temp: float = 0.3,
        uuid: str = ''
    ) -> dict:
    """
    Extract structured recipe information directly from images using a Vision model.

    Args:
        image_paths (list): A list of file paths to the images.
        model (str): The Vision model to use.
        server_url (str): The URL of the server where the LLM is running.
        temp (float): Temperature for the LLM.
        api_key (str): API key for OpenAI models if used.

    Returns:
        dict: Extracted recipe information as a structured JSON object.
    """
    # Step 1: Resize images and encode them in base64
    encoded_images = resize_and_encode_images(image_paths, uuid)
    if not encoded_images:
        print("No images available for extraction.")
        return {}

    # Step 2: Construct the Vision prompt
    prompt = f"""
    You are an expert chef and recipe writer.
    Perform OCR and extract the structured recipe information from the following image.
    Extract the exact text and respect borders and columns so text that is continuous.
    Feel free to omit superfluous text that is filler such as recipe history or personal stories from the author. 
    Do not omit tips or notes on how to make the dish.
    """

    extracted_text = []

    for enc_img in encoded_images:
        if model == 'openai':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            try:
                print('Requesting text extraction from OpenAI..')
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{enc_img}"}}
                            ]
                        }
                    ],
                    temperature=temp
                )

                content = response.choices[0].message.content
                extracted_text.append(content)

            except Exception as e:
                print(f"Error communicating with OpenAI: {e}")
                return {}

        else:
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
                print('Requesting text extraction from Vision model...')
                response = requests.post(f"{server_url}/api/chat", json=payload)
                response.raise_for_status()

                result = response.json()
                output = result.get("message", {}).get("content", "")
                extracted_text.append(output)

            except Exception as e:
                print(f"Error communicating with Vision server: {e}")
                return {}

    return extracted_text


def resize_and_encode_images(image_paths, uuid, max_dimension: int = 1120) -> list:
    """
    Resize images so that the largest dimension is `max_dimension`, save them as PNG in the save_dir,
    and encode them as base64 strings for processing.

    Args:
        image_paths (list): List of image file paths or BytesIO objects.
        save_dir (str): Directory to save the images.
        uuid (str): UUID for naming the saved files.
        max_dimension (int): Maximum dimension for resizing.

    Returns:
        list: List of base64-encoded images.
    """
    encoded_images = []

    for idx, image_input in enumerate(image_paths, start=1):
        try:
            if isinstance(image_input, io.BytesIO):
                # Open image from BytesIO object directly
                image_bytes = io.BytesIO(image_input.getvalue())
                image = Image.open(image_bytes)

            elif isinstance(image_input, str) and os.path.isfile(os.path.expanduser(image_input)):
                # Expand and open the image from a file path
                file_path = os.path.expanduser(image_input)
                image = Image.open(file_path)
                print('DOOR 2')

            else:
                # Unsupported input type
                print(f"Unsupported file input: {image_input}")
                continue

            # Resize image if necessary
            original_size = image.size
            max_side = max(original_size)
            
            if max_side > max_dimension:
                ratio = max_dimension / max_side
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                print(f"Resizing image from {original_size} to {new_size}")
                image = image.resize(new_size, Image.LANCZOS)

            file_name = f"{uuid}-{idx}.png"

            # Convert the image to bytes and encode as base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")  # Save the image as PNG in memory
            image_bytes = buffered.getvalue()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            encoded_images.append(encoded_image)

            # save to GCS
            save_to_gcs(file_name, content=image_bytes, content_type='image/png')

        except Exception as e:
            print(f"Error processing image {image_input}: {e}")
            continue

    if not encoded_images:
        print("No images were successfully processed.")
    
    return encoded_images

def fetch_html_from_url(url):
    """
    Fetch the HTML content from a URL using a random user-agent to simulate a real browser request.

    Args:
        url (str): The URL to fetch.

    Returns:
        response: The HTML response object or None if there was an error.
    """
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
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url} - {e}")
        return None
    
    return response

def parse_html(html_list):
    """
    Parse the HTML content to extract and clean the text.

    Args:
        html (str): The raw HTML content.

    Returns:
        str: Cleaned text extracted from the HTML.
    """
    combined_text = ''

    for htm in html_list:
        if hasattr(htm, 'content'):
            html = htm.content
        else:
            html = htm

        soup = BeautifulSoup(html, 'lxml')

        # Remove unnecessary tags
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        # Extract all text from visible elements
        text = soup.get_text(separator=' ')

        combined_text += text

    return clean_text(combined_text)

def clean_text(text):
    """
    Clean the extracted text by removing extra spaces and blank lines.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[ \t]+', ' ', text)  # Remove extra spaces/tabs
    text = re.sub(r'\n{2,}', '\n\n', text)  # Collapse multiple blank lines
    text = '\n'.join([line.strip() for line in text.split('\n')])

    return text.strip()
