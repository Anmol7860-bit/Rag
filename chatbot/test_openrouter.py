from llm.openrouter import OpenRouterProvider
from llm.request import ModelRequest


provider = OpenRouterProvider()

request = ModelRequest(
    prompt="Explain Retrieval-Augmented Generation in two sentences."
)

response = provider.generate(request)

print("\n===== OPENROUTER RESPONSE =====")
print(response)