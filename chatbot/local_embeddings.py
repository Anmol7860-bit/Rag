from sentence_transformers import SentenceTransformer


# Load the local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Test text
text = "The company provides software technology and educational products."


# Generate embedding
embedding = model.encode(text)


print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])