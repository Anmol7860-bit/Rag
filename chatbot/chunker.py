from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_text_file


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":
    text = load_text_file("data/company.txt")

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)