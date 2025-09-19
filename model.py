import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Configure the API key
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY not found in your .env file.")
        exit()
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"An error occurred during API configuration: {e}")
    exit()

print("--- Querying for available generative models ---")

try:
    # List all available models
    for model in genai.list_models():
        # We only care about models that can be used for chatting/generating text
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)
except Exception as e:
    print(f"\nAn error occurred while trying to list the models: {e}")


print("---------------------------------------------")