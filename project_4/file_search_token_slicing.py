import chromadb
from chromadb.utils import embedding_functions
import tiktoken

def split_text_by_tokens(text, chunk_size=50, overlap=10):
    """
    Splits text into chunks based on TOKEN count, not character count.
    """
    # 1. Load a tokenizer (cl100k_base is the standard for modern models.
    # Even though we use Gemini, this tokenizer is a good proxy for english
    encoder = tiktoken.get_encoding("cl100k_base")

    # 2. Encode: Turn string into a list of integers
    token_list = encoder.encode(text)
    print(f"[Stats] Original Text: {len(text)} chars -> {len(token_list)} tokens.")

    chunks = []
    start_token_idx = 0

    # 3. The Sliding window loop
    while start_token_idx < len(token_list):
        end_token_idx = start_token_idx + chunk_size

        # Slice the list of integers
        chunk_tokens = token_list[start_token_idx:end_token_idx]

        # 4. Decode: Turn integers back into a string chunk
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Slide forward
        start_token_idx += (chunk_size - overlap)

    return chunks

# --- TEST IT ---
text_input = "Generative AI is transforming software engineering." * 10

print("--- Token Slizer ---")
chunks = split_text_by_tokens(text_input, chunk_size=20, overlap=5)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: \"{chunk}\"")
