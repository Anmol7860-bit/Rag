import os

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider
from .request import ModelRequest


load_dotenv()


class OpenAIProvider(LLMProvider):

    def __init__(
        self,
        model="gpt-5.1"
    ):

        self.model = model

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate(
        self,
        request: ModelRequest
    ) -> str:

        model = request.model or self.model

        response = self.client.responses.create(
            model=model,
            input=request.prompt
        )

        return response.output_text
    