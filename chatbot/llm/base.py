from abc import ABC, abstractmethod

from .request import ModelRequest


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, request: ModelRequest) -> str:
        pass