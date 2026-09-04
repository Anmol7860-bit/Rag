import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

conversation = []

while True:
    user_message = input("You: ")

    if user_message.lower() == "exit":
        break

    conversation.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=conversation
    )

    print("AI:", response.text)

    conversation.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })











