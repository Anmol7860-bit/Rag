import os

import chromadb
from dotenv import load_dotenv
from google import genai

from document_loader import load_text_file
from chunker import chunk_text


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


# Load document
text = load_text_file("data/company.txt")

# Create chunks
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")


# Process chunks in batches
batch_size = 100

for start in range(0, len(chunks), batch_size):

    batch_chunks = chunks[start:start + batch_size]

    print(
        f"Processing chunks {start} to "
        f"{start + len(batch_chunks) - 1}"
    )

    # Generate embeddings for the entire batch
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch_chunks
    )

    embeddings = [
        embedding.values
        for embedding in result.embeddings
    ]

    # Create IDs
    ids = [
        f"chunk_{i}"
        for i in range(start, start + len(batch_chunks))
    ]

    # Metadata
    metadatas = [
        {
            "source": "company.txt",
            "chunk_index": i
        }
        for i in range(start, start + len(batch_chunks))
    ]

    # Store batch in ChromaDB
    collection.upsert(
        ids=ids,
        documents=batch_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


print("\nAll chunks successfully stored!")
print("Total records in ChromaDB:", collection.count())