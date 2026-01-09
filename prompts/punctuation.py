from .base import SimpleAugmentation


class Punctuation(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "punctuation_llm"

    @property
    def prompt_template(self) -> str:
        return """Add realistic punctuation Easy Data transformations that modifies punctuation in text to improve model robustness to punctuation variations.

Operations:
- Add/remove terminal punctuation (period, exclamation, question mark)
- Insert/remove commas
- Add/remove ellipses
- Swap punctuation types
- Add/remove spaces around punctuation
- Double/single quote variations

CRITICAL: Do NOT change the actual words or meaning, only add surface-level errors.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
