"""AI models through OpenRouter: one key, a model per task, and the Model lab to compare them.

    openrouter.py  the HTTP client (chat, function calling, speech to text, model list, usage and cost)
    tasks.py       the AI tasks and the model and temperature each one uses
    forms.py       the inputs voice and AI can fill, and how each is checked
    service.py     AIService: transcribe, fill fields, refine, and the chat model, per task
    evals.py       test cases and scores for comparing models (the Model lab)

Nothing is imported here on purpose, so `pacesense.assistant` and `pacesense.llm` can import each other's
modules without a cycle.
"""
