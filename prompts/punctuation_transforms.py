"""
Custom TextAttack transformations for punctuation manipulation.
"""

import random
import re
from typing import List

from textattack.shared import AttackedText
from textattack.transformations import Transformation


class PunctuationTransformation(Transformation):
    """
    Transformation that modifies punctuation in text to improve model robustness
    to punctuation variations.

    Operations:
    - Add/remove terminal punctuation (period, exclamation, question mark)
    - Insert/remove commas
    - Add/remove ellipses
    - Swap punctuation types
    - Add/remove spaces around punctuation
    - Double/single quote variations
    """

    def __init__(
        self,
        add_prob: float = 0.3,
        remove_prob: float = 0.3,
        swap_prob: float = 0.2,
        ellipsis_prob: float = 0.1,
        space_prob: float = 0.1,
    ):
        """
        Args:
            add_prob: Probability of adding punctuation
            remove_prob: Probability of removing punctuation
            swap_prob: Probability of swapping punctuation types
            ellipsis_prob: Probability of adding/converting to ellipsis
            space_prob: Probability of modifying spaces around punctuation
        """
        self.add_prob = add_prob
        self.remove_prob = remove_prob
        self.swap_prob = swap_prob
        self.ellipsis_prob = ellipsis_prob
        self.space_prob = space_prob

        # Punctuation sets
        self.terminal_punct = [".", "!", "?"]
        self.internal_punct = [",", ";", ":", "-"]
        self.swap_map = {
            ".": ["!", "?", "...", ""],
            "!": [".", "?", "..."],
            "?": [".", "!", "?!"],
            ",": [";", " -", ""],
            ";": [",", ".", ":"],
            ":": [";", " -", ","],
            "...": [".", "..", "…."],
            '"': ["'", '"', '"'],
            # "'": ['"', ''', '`'],
        }

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: List[int]
    ) -> List[AttackedText]:
        """Generate transformed texts with punctuation modifications."""

        transformed_texts = []
        original_text = current_text.text

        # Strategy 1: Modify terminal punctuation
        terminal_modified = self._modify_terminal_punctuation(original_text)
        if terminal_modified != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(terminal_modified))

        # Strategy 2: Modify internal punctuation (commas, etc.)
        internal_modified = self._modify_internal_punctuation(original_text)
        if internal_modified != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(internal_modified))

        # Strategy 3: Add random punctuation
        punct_added = self._add_random_punctuation(original_text)
        if punct_added != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(punct_added))

        # Strategy 4: Remove punctuation
        punct_removed = self._remove_random_punctuation(original_text)
        if punct_removed != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(punct_removed))

        # Strategy 5: Ellipsis variations
        ellipsis_modified = self._modify_ellipsis(original_text)
        if ellipsis_modified != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(ellipsis_modified))

        # Strategy 6: Space around punctuation
        space_modified = self._modify_punctuation_spacing(original_text)
        if space_modified != original_text:
            transformed_texts.append(current_text.generate_new_attacked_text(space_modified))

        return transformed_texts

    def _modify_terminal_punctuation(self, text: str) -> str:
        """Modify end-of-sentence punctuation."""
        text = text.rstrip()

        if not text:
            return text

        last_char = text[-1]

        if last_char in self.terminal_punct:
            if random.random() < self.remove_prob:
                # Remove terminal punctuation
                return text[:-1]
            elif random.random() < self.swap_prob:
                # Swap to different terminal punctuation
                options = [p for p in self.swap_map.get(last_char, []) if p != last_char]
                if options:
                    return text[:-1] + random.choice(options)
        else:
            if random.random() < self.add_prob:
                # Add terminal punctuation
                return text + random.choice(self.terminal_punct)

        return text

    def _modify_internal_punctuation(self, text: str) -> str:
        """Modify internal punctuation (commas, semicolons, etc.)."""
        result = text

        for punct in self.internal_punct:
            if punct in result:
                if random.random() < self.swap_prob:
                    options = self.swap_map.get(punct, [])
                    if options:
                        # Replace one random occurrence
                        positions = [m.start() for m in re.finditer(re.escape(punct), result)]
                        if positions:
                            pos = random.choice(positions)
                            replacement = random.choice(options)
                            result = result[:pos] + replacement + result[pos + len(punct) :]
                            break

        return result

    def _add_random_punctuation(self, text: str) -> str:
        """Add punctuation at random positions."""
        if random.random() > self.add_prob:
            return text

        words = text.split()
        if len(words) < 3:
            return text

        # Add comma after a random word (not first or last)
        insert_pos = random.randint(1, len(words) - 2)
        punct_to_add = random.choice([",", ";", " -"])

        words[insert_pos] = words[insert_pos].rstrip(".,;:") + punct_to_add

        return " ".join(words)

    def _remove_random_punctuation(self, text: str) -> str:
        """Remove random punctuation from text."""
        if random.random() > self.remove_prob:
            return text

        # Find all punctuation positions (excluding terminal)
        punct_pattern = r"[,;:\-]"
        matches = list(re.finditer(punct_pattern, text[:-1] if text else text))

        if matches:
            match = random.choice(matches)
            pos = match.start()
            # Remove punctuation and normalize spacing
            result = text[:pos] + text[pos + 1 :]
            result = re.sub(r"  +", " ", result)  # Normalize double spaces
            return result

        return text

    def _modify_ellipsis(self, text: str) -> str:
        """Add, remove, or modify ellipses."""
        if random.random() > self.ellipsis_prob:
            return text

        # Convert period to ellipsis at end
        if text.endswith(".") and not text.endswith(".."):
            return text[:-1] + "..."

        # Convert ellipsis to period
        if text.endswith("..."):
            return text[:-3] + "."

        # Convert multiple periods to proper ellipsis
        text = re.sub(r"\.{2,}", "...", text)

        # Add ellipsis in middle (for dramatic pause effect)
        words = text.split()
        if len(words) > 4 and random.random() < 0.3:
            pos = random.randint(2, len(words) - 2)
            words[pos] = words[pos] + "..."
            return " ".join(words)

        return text

    def _modify_punctuation_spacing(self, text: str) -> str:
        """Modify spacing around punctuation."""
        if random.random() > self.space_prob:
            return text

        modifications = [
            # Add space before punctuation (common error)
            (r"([a-zA-Z])([,.])", r"\1 \2"),
            # Remove space after punctuation
            (r"([,.]) +", r"\1"),
            # Add extra space after punctuation
            (r"([,.]) ", r"\1  "),
            # Remove space before punctuation (fix error)
            (r" ([,.])", r"\1"),
        ]

        pattern, replacement = random.choice(modifications)
        result = re.sub(pattern, replacement, text, count=1)

        return result

    @property
    def deterministic(self) -> bool:
        return False


class SentenceTerminationSwap(Transformation):
    """
    Simple transformation focused specifically on sentence termination variations.

    Handles the common case where models are sensitive to:
    - Period vs no period
    - Period vs exclamation vs question mark
    - Single period vs ellipsis
    """

    def __init__(self):
        self.variations = {
            # Current ending -> possible alternatives
            ".": ["", "!", "?", "...", ".."],
            "!": [".", "", "?", "!!"],
            "?": [".", "", "!", "??", "?!"],
            "...": [".", "..", "", "…."],
            "..": [".", "...", ""],
            "": [".", "!", "?"],  # No punctuation
        }

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: List[int]
    ) -> List[AttackedText]:
        """Generate all termination variations."""

        transformed_texts = []
        original_text = current_text.text.rstrip()

        if not original_text:
            return transformed_texts

        # Determine current ending
        current_ending = ""
        for ending in ["...", "..", "?!", "??", "!!", ".", "!", "?"]:
            if original_text.endswith(ending):
                current_ending = ending
                break

        # Get base text without ending
        base_text = original_text
        if current_ending:
            base_text = original_text[: -len(current_ending)]

        # Generate variations
        alternatives = self.variations.get(current_ending, [".", ""])

        for alt in alternatives:
            if alt != current_ending:
                new_text = base_text + alt
                transformed_texts.append(current_text.generate_new_attacked_text(new_text))

        return transformed_texts

    @property
    def deterministic(self) -> bool:
        return False


class CommaInsertion(Transformation):
    """
    Transformation that adds or removes commas to test comma sensitivity.
    """

    def __init__(self, insertion_prob: float = 0.5):
        self.insertion_prob = insertion_prob
        # Common positions where commas are optional or often misused
        self.comma_after_words = [
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "meanwhile",
            "nevertheless",
            "consequently",
            "additionally",
            "finally",
            "well",
            "yes",
            "no",
            "oh",
            "ah",
            "so",
            "now",
            "then",
            "first",
            "second",
            "third",
            "lastly",
            "also",
            "indeed",
        ]

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: List[int]
    ) -> List[AttackedText]:
        """Generate comma variations."""

        transformed_texts = []
        text = current_text.text

        # Strategy 1: Add comma after common transitional words
        for word in self.comma_after_words:
            pattern = rf"\b({word})\b(?!,)"
            if re.search(pattern, text, re.IGNORECASE):
                modified = re.sub(pattern, r"\1,", text, count=1, flags=re.IGNORECASE)
                if modified != text:
                    transformed_texts.append(current_text.generate_new_attacked_text(modified))
                    break

        # Strategy 2: Remove existing comma after transitional words
        for word in self.comma_after_words:
            pattern = rf"\b({word}),\s*"
            if re.search(pattern, text, re.IGNORECASE):
                modified = re.sub(pattern, r"\1 ", text, count=1, flags=re.IGNORECASE)
                if modified != text:
                    transformed_texts.append(current_text.generate_new_attacked_text(modified))
                    break

        # Strategy 3: Add Oxford comma or remove it
        # Pattern: "x, y and z" <-> "x, y, and z"
        oxford_add = re.sub(r",\s+(\w+)\s+and\s+", r", \1, and ", text, count=1)
        if oxford_add != text:
            transformed_texts.append(current_text.generate_new_attacked_text(oxford_add))

        oxford_remove = re.sub(r",\s+(\w+),\s+and\s+", r", \1 and ", text, count=1)
        if oxford_remove != text:
            transformed_texts.append(current_text.generate_new_attacked_text(oxford_remove))

        # Strategy 4: Add comma before "and" or "but" in compound sentence
        compound_pattern = r"(\w)\s+(and|but|or)\s+(\w)"
        if re.search(compound_pattern, text) and random.random() < self.insertion_prob:
            modified = re.sub(compound_pattern, r"\1, \2 \3", text, count=1)
            if modified != text:
                transformed_texts.append(current_text.generate_new_attacked_text(modified))

        return transformed_texts

    @property
    def deterministic(self) -> bool:
        return False
