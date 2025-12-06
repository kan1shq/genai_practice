import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# 1. Load the API key from .env file
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: API Key not found. Please add the correct API Key to .env file")
else:
    # 2. Configure the library with the API Key
    genai.configure(api_key=api_key)

    # 3. Choose our model ('gemini-2.5-flash' for free tier)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 4. Ask the model something through the API call
    print("Asking Gemini...")
    response = model.generate_content("Explain 'Generative AI' to a 5 year old in 5 sentences")

    # 5. Print the results
    print("-" * 30)
    print(response.text)
    print("-" * 30)
