import os

from dotenv import load_dotenv
from google import genai

from .base import LLMProvider
from .request import ModelRequest


load_dotenv()


class GeminiProvider(LLMProvider):

    def __init__(
        self,
        model="gemini-3.7-flash"
    ):

        self.model = model

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def generate(
        self,
        request: ModelRequest
    ) -> str:

        model = request.model or self.model

        response = self.client.models.generate_content(
            model=model,
            contents=request.prompt
        )

        return response.text