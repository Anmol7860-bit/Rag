import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

text = "The company provides 24 annual paid leave days."

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

embedding = result.embeddings[0].values

print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])