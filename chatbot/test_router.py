from llm.router import ModelRouter
from llm.request import ModelRequest


router = ModelRouter(
    primary="gemini",
    fallback="openai"
)


request = ModelRequest(
    prompt="Explain RAG in one sentence.",
    task="reasoning",
    priority="balanced"
)


response = router.generate(request)

print("\nAI:", response)