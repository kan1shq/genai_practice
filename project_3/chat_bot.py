import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

dot_env_path = Path(__file__).parent.parent/'.env'
load_dotenv(dot_env_path)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-2.5-flash')

# 1. Initialize the chat object (empty history list)
chat = model.start_chat(history=[])

print("Bot: Hello! I am ready to chat. Type 'exit' to quit.")

while True:
    # 2. Get User Input
    user_input = str(input("User: "))
    if user_input.lower() in ['exit', 'quit']:
        break

    # 3. Send the message (already appended to history list)
    response = chat.send_message(user_input)

    # 4. Print Responses
    print(f"Bot: {response.text}")

    # OPTIONAL: Peek at metrics
    print(f"\n[DEBUG] History count: {len(chat.history)} items\n")
