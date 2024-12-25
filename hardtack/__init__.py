# app/__init__.py

from .agent import get_bot_response
import os
#from dotenv import load_dotenv

# Load environment variables from the .env file
#load_dotenv()

# load the json key from the environment variable
gcloud_key = os.environ.get("GCLOUD_SERVICE_KEY")

if gcloud_key:
    # write the key to a file (if needed)
    key_path = "/tmp/gcloud-key.json"
    with open(key_path, "w") as key_file:
        key_file.write(gcloud_key)

    # set GOOGLE_APPLICATION_CREDENTIALS to point to this file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


