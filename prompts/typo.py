from .base import SimpleAugmentation


class Typo(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "typo_llm"

    @property
    def prompt_template(self) -> str:
        return """Add realistic typos and formatting variations to the text:
- 2-4 typos (keyboard adjacency errors, missing letters, doubled letters)
- Capitalization variations
- Common misspellings

CRITICAL: Do NOT change the actual words or meaning, only add surface-level errors.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
