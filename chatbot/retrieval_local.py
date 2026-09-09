import chromadb
from sentence_transformers import SentenceTransformer


# Load the same embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="company_documents_local"
)


# User's question
question = "What does Zi Corporation do?"


# Convert question into embedding
question_embedding = embedding_model.encode(
    question
).tolist()


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)


print("\n===== RETRIEVED CHUNKS =====")

for i, document in enumerate(results["documents"][0]):

    print(f"\n--- Result {i + 1} ---")
    print(document)

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nDistance:")
    print(results["distances"][0][i])