from PIL import Image
import base64
import io
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import re

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

def parse_html(html):
    """
    Parse the HTML content to extract and clean the text.

    Args:
        html (str): The raw HTML content.

    Returns:
        str: Cleaned text extracted from the HTML.
    """
    if hasattr(html, 'content'):
        html = html.content

    soup = BeautifulSoup(html, 'lxml')

    # Remove unnecessary tags
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    # Extract all text from visible elements
    text = soup.get_text(separator=' ')

    return clean_text(text)

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
