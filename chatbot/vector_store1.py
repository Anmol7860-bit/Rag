import chromadb
from sentence_transformers import SentenceTransformer

from document_loader import load_text_file
from chunker import chunk_text


# Load local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="company_documents_local"
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

    # Generate local embeddings
    embeddings = embedding_model.encode(
        batch_chunks
    ).tolist()

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

    # Store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=batch_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


print("\nAll chunks successfully stored!")

print(
    "Total records in ChromaDB:",
    collection.count()
)