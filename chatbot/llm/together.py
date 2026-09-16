import os

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider
from .request import ModelRequest


load_dotenv()


class TogetherProvider(LLMProvider):

    def __init__(self, model="openai/gpt-oss-20b"):

        self.model = model

        self.client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.ai/v1"
        )

    def generate(self, request: ModelRequest) -> str:

        model = request.model or self.model

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