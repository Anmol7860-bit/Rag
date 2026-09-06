from pathlib import Path


def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

if __name__ == "__main__":
    text = load_text_file("data/company.txt")

    print(text)