from dataclasses import dataclass


@dataclass
class ModelRequest:

    prompt: str

    task: str = "general"

    requires_vision: bool = False

    priority: str = "balanced"

    # If specified, use this exact model.
    # If None, router decides automatically.
    model: str | None = None