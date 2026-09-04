import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Temporary in-memory conversation storage
conversations = {}


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


@app.get("/")
def home():
    return {"message": "Chatbot API is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    # Create a new conversation if it doesn't exist
    if request.conversation_id not in conversations:
        conversations[request.conversation_id] = []

    conversation = conversations[request.conversation_id]

    # Add user's message
    conversation.append({
        "role": "user",
        "parts": [{"text": request.message}]
    })

    # Send conversation history to Gemini
    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=conversation
    )

    # Save Gemini's response
    conversation.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    return {
        "conversation_id": request.conversation_id,
        "response": response.text
    }