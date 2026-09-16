import os

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider
from .request import ModelRequest

load_dotenv()


class OpenRouterProvider(LLMProvider):

    def __init__(self, model="nvidia/nemotron-3.5-lightning:free"):

        self.model = model

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    def generate(self, request: ModelRequest) -> str:

        model = self.model

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        )

        return response.choices[0].message.content