import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


# --- CONFIGURATION ---
# 1. System instruction: This sets the behaviour BEFORE the user speaks
system_information = """
    You are a specialized data extraction tool.
    You strictly output JSON only. 
    No markdown formatting, no chatting, no intro text.
    Extract the 'customer_name', 'order_id', and 'sentiment' (positive/negative) from the text.
    """

system_information2 = """ You are a helpful, very excited Gen Z support assistant using lots of emojis."""

# 2. Generation config: controls the "physics" of the model
config = genai.types.GenerationConfig(
            temperature=0.1,
            #response_mime_type="application/json"
        )

# 3. Initialize the model with these settings
model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_information,
            generation_config=config
        )

# --- THE INPUT ---
#user_email = """
#Subject: URGENT!!!
#Hi team, I am writing to complain about order #99281. 
#My name is Sarah connor and I am very angry. 
#The package arrive crushed! Fix this immediately please.
#"""

user_input = str(input("Awaiting user prompt: "))


print(f"Analysing email from user: ")

# 4. Generate
response = model.generate_content(user_input)

# 5. Output
print("-" * 30, response.text, "-" * 30, sep="\n")
