from .gemini import GeminiProvider
from .openai import OpenAIProvider


class ModelRouter:

    def __init__(self):

        # =====================================
        # Available models
        # =====================================

        self.models = {

            # Gemini
            "gemini-2.5-flash":
                GeminiProvider("gemini-2.5-flash"),

            "gemini-3.7-flash":
                GeminiProvider("gemini-3.7-flash"),

            # OpenAI
            "gpt-4o-mini":
                OpenAIProvider("gpt-4o-mini"),

            "gpt-5-mini":
                OpenAIProvider("gpt-5-mini"),

            "gpt-5.1":
                OpenAIProvider("gpt-5.1"),
        }


        # =====================================
        # Default automatic routing
        # =====================================

        self.primary_model = "gemini-3.7-flash"

        self.fallback_model = "gemini-2.5-flash"


    # =====================================
    # Automatic model selection
    # =====================================

    def select_model(self, request):

        # Vision request
        if request.requires_vision:

            return "gemini-3.7-flash"


        # Reasoning request
        if request.task == "reasoning":

            return "gpt-5.1"


        # Cost optimization
        if request.priority == "cost":

            return "gemini-2.5-flash"


        # Speed optimization
        if request.priority == "speed":

            return "gemini-2.5-flash"


        # Default
        return self.primary_model


    # =====================================
    # Generate
    # =====================================

    def generate(self, request):

        # ---------------------------------
        # Manual model selection
        # ---------------------------------

        if request.model is not None:

            selected_model = request.model

        # ---------------------------------
        # Automatic model selection
        # ---------------------------------

        else:

            selected_model = self.select_model(request)


        # ---------------------------------
        # Validate model
        # ---------------------------------

        if selected_model not in self.models:

            raise ValueError(
                f"Unknown model: {selected_model}"
            )


        print(
            f"\nSelected model: {selected_model}"
        )


        provider = self.models[selected_model]


        # ---------------------------------
        # Try selected model
        # ---------------------------------

        try:

            return provider.generate(request)


        # ---------------------------------
        # Fallback
        # ---------------------------------

        except Exception as error:

            print(
                f"\nSelected model failed: "
                f"{type(error).__name__}"
            )


            # Don't fallback to itself
            if selected_model == self.fallback_model:

                raise error


            print(
                f"Falling back to: "
                f"{self.fallback_model}"
            )


            fallback_provider = self.models[
                self.fallback_model
            ]


            # Create a new request for fallback
            fallback_request = ModelRequest(
                prompt=request.prompt,
                task=request.task,
                requires_vision=request.requires_vision,
                priority=request.priority,
                model=self.fallback_model
            )


            return fallback_provider.generate(
                fallback_request
            )
    