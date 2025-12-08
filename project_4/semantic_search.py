import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup the vector database
# We use a persistent client so data saves to our hard drive
client = chromadb.PersistentClient(path="./my_library_db")

# 2. Setup the translator/Embedding function
# This turns text into numbers. We use the default, which downloads a small model (auto)
# (all-MiniLM-L6-v2 is the standard)
print("Downloading embedding model...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Create a collection (think of this as a Table in SQL)
# We delete it first if it exists so we start fresh every time we run this script
try:
    client.delete_collection(name="my_knowledge_base")
except:
    pass # Collection doesn't exist, which is fine

collection = client.create_collection(
        name="my_knowledge_base",
        embedding_function=sentence_transformer_ef
        )

# Add Documents (The knowledge)
documents = [
        "The 2024 Olympics were held in Paris, France.",
    "Python is a dynamically typed language known for readability.",
    "To fix a 'Connection Refused' error, check your firewall settings.",
    "The secret ingredient to the family soup recipe is a dash of nutmeg.",
    "Battlefield 2042 had a rocky launch but improved with updates."
]

# We now need IDs for each document (just like a primary key)
ids = [f"doc_{i}" for i in range(len(documents))]

print("Indexing documents... (This might take a second for the model to download")
collection.add(
        documents=documents,
        ids=ids
        )
print(f"DONE! Indexed {len(documents)} documents.\n")

# 5. The Search Loop
print("Librarian: Ask me anything! (I match meaning, not just words)")
while True:
    query = input("\nQuery: ")
    if query.lower in ['exit', 'quit']: break

    # Query the database
    results = collection.query(
            query_texts=[query],
            n_results=2 # Return top 2 matches
            )

    # Print results
    print("\n--- Found relevant info: ---")
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        distance = results['distances'][0][i] # Lower distance is a better match
        print(f"Match {i+1} (Dist: {distance:.4f}): \"{doc_text}\"")
