import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()


# Gemini client
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="company_documents"
)


# User's question
question = "What does Zi Corporation do?"


# Convert question into an embedding
result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = result.embeddings[0].values


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)


print("\n--- Retrieved Chunk ---")
print(results["documents"][0][0])

print("\n--- Metadata ---")
print(results["metadatas"][0][0])

print("\n--- Distance ---")
print(results["distances"][0][0])