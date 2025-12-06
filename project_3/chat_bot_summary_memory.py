import os
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We use two models:
# 1. Chat model (can be pro as well)
# 2. Summarizer (faster models)
chat_model = genai.GenerativeModel('gemini-2.5-flash')
summary_model = genai.GenerativeModel('gemini-2.5-pro')

# --- SETTINGS ---
MAX_TOKEN_LIMIT = 300

# State
chat_history = []
running_summary = ""    # Holds the long-term memory setting

def get_token_count(history, current_summary):
    """
    Helper to count tokens of (Summary + History List)
    """
    # We pretend we are sending [Summary] + [History List] to get accurate count
    # Note: We just treat the summary as a text string for counting
    combined_text = f"Summary: {current_summary}\n"
    history_string = ' '.join(msg['parts'][0] for msg in history)
    combined_text += history_string

    return chat_model.count_tokens(combined_text).total_tokens

def update_summary(messages_batch, old_summary):
    """
    Asks the LLM to merge new events into the existing memory
    """
    print("\n   [System] Compressing oldest memories...")

    batch_text = ""
    for msg in messages_batch:
        role = "User" if msg['role'] == 'user' else 'Bot'
        text = msg['parts'][0]
        batch_text += f"{role}: {text}\n"

    prompt = f"""
    Current summary of conversation: "{old_summary}"

    New conversation Lines to add: 
    {batch_text}

    Goal: Update the summary to include the new event efficiently. Keep it concise.
    Constraint: The new summary MUST be under {MAX_TOKEN_LIMIT // 5} tokens.
    Constraint: Ignore trivial small talk ("hi", "thanks"). Focus on facts.
    """
 
    response = summary_model.generate_content(prompt)
    return response.text.strip()

print(f"Smart Bot: I keep recent messages raw, but summarize old ones recursively.")
print(f"    (Max Limit: {MAX_TOKEN_LIMIT} tokens)\n")

while True:
    user_input = str(input("You: "))
    if user_input.lower() in ['exit', 'quit']: break

    # 1.Add User input to History (Temporarily)
    chat_history.append({"role": "user", "parts": [user_input]})

    # 2. The Smart Loop: Compress until we fit
    while True:
        batch_to_summarize = []

        # Check size
        current_tokens = get_token_count(chat_history, running_summary)
        print(f"   [Debug] Tokens: {current_tokens} / {MAX_TOKEN_LIMIT}")

        if current_tokens <= MAX_TOKEN_LIMIT:
            break # we fit the token window

        # If we are over limit, must compress the oldest pairs (4 and then 2 for rate limit) 
        if len(chat_history) >= 4:
            # Grab oldest 4 items (2 User/Bot pairs)
            batch_to_summarize.extend(chat_history.pop(0) for _ in range(4))
        elif len(chat_history) >= 2:
            batch_to_summarize.extend(chat_history.pop(0) for _ in range(2))
        else:
            print("   [Warning] Input too long even for summary! Truncating...")
            break

        # Send the WHOLE batch to the new function
        if batch_to_summarize:
            running_summary = update_summary(
                    batch_to_summarize, 
                    running_summary
                    )
            print(f"   [Debug] New Summary: {running_summary[:50]}...")

    # 3. Construct the payload
    # Logic: System Instruction (Summary) + Recent Messages
    # We cheat a little by injecting the summary into the system prompt for this turn
    messages_to_send = chat_history

    # We define a temporary system prompt just for this turn containing the memory
    dynamic_system_prompt = f"""
    You are a helpful assistant.
    CONTEXT FROM PAST CONVERSATION:
    {running_summary}
    """

    # NOTE: model needs to be re-instantiated to maintain the dynamic summary
    chat_model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=dynamic_system_prompt
            )

    # 4. Generate Answer
    # Note: We pass the system_instruction dynamically here
    response = chat_model.generate_content(
            messages_to_send
            )

    # 5. Add bot answer to history
    print(f"Bot: {response.text}")
    chat_history.append({"role": "model", "parts": [response.text]})

