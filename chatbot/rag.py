import chromadb
from sentence_transformers import SentenceTransformer

from llm.router import ModelRouter
from llm.request import ModelRequest

# =========================
# Model selection
# =========================

print("\n==============================")
print("       RAG MODEL SELECTION")
print("==============================")

print("1. Gemini 2.5 Flash")
print("2. Gemini 3.7 Flash")
print("3. GPT-4o Mini")
print("4. GPT-5 Mini")
print("5. GPT-5.1")
print("6. NVIDIA Nemotron")
print("7. Automatic Router")
print("8. All Models with Evals")

choice = input("\nSelect model: ")

model_map = {
    "1": "gemini-2.5-flash",
    "2": "gemini-3.7-flash",
    "3": "gpt-4o-mini",
    "4": "gpt-5-mini",
    "5": "gpt-5.1",
    "6": "nvidia-nemotron",
    "7": None,
    "8": "benchmark"
}

selected_model = model_map.get(choice)

if choice not in model_map:
    print("Invalid selection. Using automatic routing.")
    selected_model = None

if selected_model == "benchmark":
    print("\nAll-model evaluation mode enabled.")

elif selected_model:
    print(f"\nManual model selected: {selected_model}")

else:
    print("\nAutomatic model routing enabled.")


# =========================
# 1. Embedding model
# =========================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# 2. ChromaDB
# =========================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="company_documents_local"
)


# =========================
# 3. Model Router
# =========================

router = ModelRouter()


# =========================
# 4. Conversation memory
# =========================

conversation_history = []


# =========================
# 5. Rewrite question
# =========================

def rewrite_question(question, conversation_history):

    if not conversation_history:
        return question

    history = "\n".join(
        [
            f"User: {turn['user']}\nAI: {turn['ai']}"
            for turn in conversation_history
        ]
    )

    rewrite_prompt = f"""
Convert the user's latest question into a standalone question
that can be understood without the conversation history.

Conversation:
{history}

Latest question:
{question}

Return ONLY the rewritten question.
"""

    request = ModelRequest(
        prompt=rewrite_prompt,
        task="query_rewriting",
        priority="cost"
    )

    response = router.generate(request)

    return response.strip()


# =========================
# 6. Chat loop
# =========================

import time


def run_all_models(question, context, history):
    models = [
        "gemini-2.5-flash",
        "gemini-3.7-flash",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-5.1",
        "nvidia-nemotron"
    ]

    results = []

    prompt = f"""
You are a helpful conversational RAG assistant.

Use the provided document context to answer the user's question.

Previous conversation:
{history}

Retrieved document context:
{context}

Current question:
{question}

Rules:
- Use the retrieved context when answering questions about the documents.
- Use conversation history to understand references such as "it", "they", or "that company".
- Do not make up information.
- If the information is not available in the retrieved context, say:
  "I don't have enough information in the provided documents."

Answer:
"""

    for model in models:

        print("\n" + "=" * 60)
        print(f"Running model: {model}")
        print("=" * 60)

        request = ModelRequest(
            prompt=prompt,
            task="rag_generation",
            priority="balanced",
            model=model
        )

        start_time = time.perf_counter()

        try:
            answer = router.generate(request)

            end_time = time.perf_counter()

            latency = end_time - start_time

            results.append({
                "model": model,
                "answer": answer,
                "latency": latency,
                "status": "success"
            })

            print(f"\nLatency: {latency:.2f} seconds")
            print("\nAnswer:")
            print(answer)

        except Exception as error:

            end_time = time.perf_counter()

            latency = end_time - start_time

            results.append({
                "model": model,
                "answer": None,
                "latency": latency,
                "status": f"failed: {type(error).__name__}"
            })

            print(f"\nModel failed: {type(error).__name__}")

    return results

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break


    # =========================
    # 7. Rewrite question
    # =========================

    search_question = rewrite_question(
        question,
        conversation_history
    )

    print("\nSearch query:", search_question)


    # =========================
    # 8. Embed question
    # =========================

    question_embedding = embedding_model.encode(
        search_question
    ).tolist()


    # =========================
    # 9. Retrieve documents
    # =========================

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    retrieved_chunks = results["documents"][0]


    # =========================
    # 10. Build RAG context
    # =========================

    context = "\n\n".join(
        retrieved_chunks
    )


    # =========================
    # 11. Build conversation history
    # =========================

    history = "\n".join(
        [
            f"User: {turn['user']}\nAI: {turn['ai']}"
            for turn in conversation_history
        ]
    )


    # =========================
    # 12. Build RAG prompt
    # =========================

    prompt = f"""
You are a helpful conversational RAG assistant.

Use the provided document context to answer the user's question.

Previous conversation:
{history}

Retrieved document context:
{context}

Current question:
{question}

Rules:
- Use the retrieved context when answering questions about the documents.
- Use conversation history to understand references such as "it", "they", or "that company".
- Do not make up information.
- If the information is not available in the retrieved context, say:
  "I don't have enough information in the provided documents."

Answer:
"""


    # =========================
    # 13. Generate answer
    # =========================

    if selected_model == "benchmark":

        benchmark_results = run_all_models(
            question=question,
            context=context,
            history=history
        )

        # =========================
        # 14. Display comparison
        # =========================

        print("\n\n")
        print("=" * 70)
        print("                    MODEL COMPARISON")
        print("=" * 70)

        for result in benchmark_results:

            print("\n" + "-" * 70)
            print(f"MODEL: {result['model']}")
            print(f"STATUS: {result['status']}")
            print(f"LATENCY: {result['latency']:.2f} seconds")

            if result["answer"] is not None:
                print("\nANSWER:")
                print(result["answer"])

        # Do NOT save benchmark answers
        # into conversation history.

    else:

        request = ModelRequest(
            prompt=prompt,
            task="rag_generation",
            priority="balanced",
            model=selected_model
        )

        answer = router.generate(request)

        # =========================
        # 14. Save conversation
        # =========================

        conversation_history.append({
            "user": question,
            "ai": answer
        })

        # =========================
        # 15. Display answer
        # =========================

        print("\nAI:", answer)




