from abc import ABC, abstractmethod
from typing import List  # Add this import


class LLMGeneratorBase(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, **kwargs) -> List[str]:  # Change list[str] to List[str]
        pass
