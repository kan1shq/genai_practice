import os
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
FILE_PATH = "my_knowledge.txt"  # <--- Make sure this file exists
CHUNK_SIZE = 500 # How many characters per chunk
OVERLAP = 50    # Overlap to ensure we don't cut sentences in half

# 1. Setup database and embedder
client = chromadb.PersistentClient(path="./my_library_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Reset collection for a clean start
try:
    client.delete_collection(name="my_files")
except:
    pass

collection = client.create_collection(name="my_files", embedding_function=ef)

# 2. Helper: The "Chunker"
def split_text(text, chunk_size, overlap):
    """
    Splits text into sliding windows.
    Simple approach: character slicing.
    (Professional apps use Token slicing, but this works fine for now).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Get the slice
        chunk = text[start:end]
        chunks.append(chunk)

        # Slide the window forward, minus the overlap
        start += (chunk_size - overlap)
    return chunks


# 3. Read & Process File
if os.path.exists(FILE_PATH):
    print(f"Reading {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()


    # Create chunks
    text_chunks = split_text(full_text, CHUNK_SIZE, OVERLAP)
    print(f"Split Document into {len(text_chunks)} chunks.")

    # Generate IDs (chunk_0)
    ids = [f"chunks_{i}" for i in range(len(text_chunks))]

    # 4. Index into Vector DB
    print("Indexing chunks into ChromaDB...")
    collection.add(
            documents=text_chunks,
            ids=ids,
            metadatas=[{"source": FILE_PATH} for _ in text_chunks]
            )
    print("Indexing complete!")

else:
    print(f"Error: Could not find {FILE_PATH}. Please create a knowledge book first.")
    exit()

# 5. Search Loop
print("\nLibrarian: I have read your file. Ask me questions!")
while True:
    query = input("\nQuery (or 'exit'): ")
    if query.lower() in ['exit', 'quit']: break

    results = collection.query(
            query_texts=[query],
            n_results=2
            )

    print("\n--- Relevant Excerpts: ---")
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        score = results['distances'][0][i]
        print(f"[{i+1}] (Score: {score:.4f}): \"...{text}...\"") 
