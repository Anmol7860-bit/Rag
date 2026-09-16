from llm.router import ModelRouter
from llm.request import ModelRequest


router = ModelRouter()

request = ModelRequest(
    prompt="Explain RAG in two sentences.",
    model="nvidia-nemotron"
)

response = router.generate(request)

print("\n===== ROUTER RESPONSE =====")
print(response)