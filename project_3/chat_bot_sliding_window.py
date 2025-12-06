import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# --- CONFIGURATION ---
# real context window sizes are typically 100k+
MAX_TOKEN_LIMIT = 500

chat = model.start_chat(history=[])

def trim_history(chat_session):
    """
    If history is too large, remove the oldest message (FIFO).
    We keep removing index 0 until we are within limits.
    """
    while True:
        # 1. Count tokens in the current history
        # (Gemini seems to have a dedicated method for this)
        curr_history_list = chat_session.history

        if not curr_history_list:
            break

        count_result = model.count_tokens(curr_history_list) # takes in both list and str
        token_count = count_result.total_tokens

        print(f"\n[System] Current Tokens: {token_count} / {MAX_TOKEN_LIMIT}")

        # 2. Break if we are safe
        if token_count < MAX_TOKEN_LIMIT:
            break

        # 3. If not, remove the oldest interaction (User + AI response)
        # We usually remove 2 items to keep the flow logical
        if len(chat_session.history) > 1:
            print("[System] Memory full! Removing oldest interaction...")
            # Pop twice to remove User prompt + AI Response
            chat_session.history.pop(0)
            chat_session.history.pop(0)
        else:
            # Safety valve: if initial prompt itself is too big
            break

print("Senior Bot: I have limited memory. Watch me forget things!")
print(f"    (Max limit set to {MAX_TOKEN_LIMIT} tokens for demo purposes)")

while True:
    user_input = str(input("\nYou: "))
    if user_input.lower() in ['exit', 'quit']:
        break

    # Before sending, we check and trim history
    trim_history(chat)

    try:
        response = chat.send_message(user_input)
        print(f"Bot: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

