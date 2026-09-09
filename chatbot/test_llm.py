from llm.gemini import GeminiProvider
from llm.openai import OpenAIProvider


prompt = "Explain what a vector database is in one sentence."


print("\n===== GEMINI =====")

gemini = GeminiProvider()

gemini_response = gemini.generate(prompt)

print(gemini_response)


print("\n===== OPENAI =====")

openai = OpenAIProvider()

openai_response = openai.generate(prompt)

print(openai_response)