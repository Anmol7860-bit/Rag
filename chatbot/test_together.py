from llm.together import TogetherProvider
from llm.request import ModelRequest


provider = TogetherProvider()

request = ModelRequest(
    prompt="Explain Retrieval-Augmented Generation in two sentences."
)

response = provider.generate(request)

print("\n===== TOGETHER AI RESPONSE =====")
print(response)