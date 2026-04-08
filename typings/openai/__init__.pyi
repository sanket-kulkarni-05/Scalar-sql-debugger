from typing import Any

class OpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    class chat:
        class completions:
            @staticmethod
            def create(*args: Any, **kwargs: Any) -> Any: ...
