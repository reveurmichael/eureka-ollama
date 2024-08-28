from llm_generator_base import LLMGeneratorBase
from typing import List  # Add this import
import ollama
import time


class OllamaGenerator(LLMGeneratorBase):
    def __init__(self, model: str):
        super().__init__(model)

    def generate(self, prompt, k, logger, temperature) -> List[str]:  # Change list[str] to List[str]

        options = ollama.Options(
            temperature=temperature
        )

        responses = []
        for i in range(k):
            init_t = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False,
                options=options
            )
            end_t = time.time()
            responses.append(response['message']['content'])
            if logger is not None:
                logger.info(f"Generation {i + 1}/{k} completed in {end_t - init_t:.2f} seconds")
                logger.info(f"Response: {response['message']['content']}")
                logger.info("+---------------------------------+")

        return responses
