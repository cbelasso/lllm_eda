"""
Fast native Python text augmentations.
These bypass TextAttack's framework for simple operations that don't need it.
"""

import random
import re
import string
from typing import Any, Dict, List, Optional

# QWERTY keyboard adjacency map
QWERTY_NEIGHBORS = {
    "q": "wa",
    "w": "qeas",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
    "1": "2q",
    "2": "13qw",
    "3": "24we",
    "4": "35er",
    "5": "46rt",
    "6": "57ty",
    "7": "68yu",
    "8": "79ui",
    "9": "80io",
    "0": "9p",
}


class FastNoiseInjection:
    """Fast character-level noise injection."""

    def __init__(
        self,
        qwerty_prob: float = 0.4,
        delete_prob: float = 0.3,
        swap_prob: float = 0.2,
        insert_prob: float = 0.1,
        pct_chars_to_modify: float = 0.05,
    ):
        self.qwerty_prob = qwerty_prob
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob
        self.insert_prob = insert_prob
        self.pct_chars_to_modify = pct_chars_to_modify

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions of text."""
        if not text or not text.strip():
            return [text] * n

        results = []
        for _ in range(n):
            results.append(self._apply_noise(text))
        return results

    def _apply_noise(self, text: str) -> str:
        """Apply random noise to text."""
        chars = list(text)
        num_modifications = max(1, int(len(chars) * self.pct_chars_to_modify))

        # Get positions of alphabetic characters (skip spaces, punctuation)
        modifiable_positions = [i for i, c in enumerate(chars) if c.isalpha()]

        if not modifiable_positions:
            return text

        # Select random positions to modify
        positions_to_modify = random.sample(
            modifiable_positions, min(num_modifications, len(modifiable_positions))
        )

        for pos in sorted(positions_to_modify, reverse=True):
            rand = random.random()

            if rand < self.qwerty_prob:
                # QWERTY substitution
                chars[pos] = self._qwerty_swap(chars[pos])
            elif rand < self.qwerty_prob + self.delete_prob:
                # Delete character
                chars.pop(pos)
            elif rand < self.qwerty_prob + self.delete_prob + self.swap_prob:
                # Swap with neighbor
                if pos < len(chars) - 1:
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            else:
                # Insert random character
                chars.insert(pos, random.choice(string.ascii_lowercase))

        return "".join(chars)

    def _qwerty_swap(self, char: str) -> str:
        """Swap character with QWERTY neighbor."""
        lower = char.lower()
        if lower in QWERTY_NEIGHBORS and QWERTY_NEIGHBORS[lower]:
            replacement = random.choice(QWERTY_NEIGHBORS[lower])
            return replacement.upper() if char.isupper() else replacement
        return char


class FastTypo:
    """Fast typo simulation."""

    def __init__(self, pct_words_to_modify: float = 0.1):
        self.pct_words_to_modify = pct_words_to_modify

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions of text."""
        if not text or not text.strip():
            return [text] * n

        results = []
        for _ in range(n):
            results.append(self._apply_typos(text))
        return results

    def _apply_typos(self, text: str) -> str:
        """Apply typos to text."""
        words = text.split(" ")
        num_to_modify = max(1, int(len(words) * self.pct_words_to_modify))

        # Get indices of words with enough characters
        modifiable_indices = [i for i, w in enumerate(words) if len(w) > 2]

        if not modifiable_indices:
            return text

        indices_to_modify = random.sample(
            modifiable_indices, min(num_to_modify, len(modifiable_indices))
        )

        for idx in indices_to_modify:
            words[idx] = self._add_typo_to_word(words[idx])

        return " ".join(words)

    def _add_typo_to_word(self, word: str) -> str:
        """Add a single typo to a word."""
        if len(word) < 2:
            return word

        typo_type = random.choice(["swap", "delete", "double", "qwerty"])

        # Find alphabetic positions
        alpha_positions = [i for i, c in enumerate(word) if c.isalpha()]
        if not alpha_positions:
            return word

        pos = random.choice(alpha_positions)
        chars = list(word)

        if typo_type == "swap" and pos < len(chars) - 1:
            # Swap adjacent characters
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo_type == "delete":
            # Delete character
            chars.pop(pos)
        elif typo_type == "double":
            # Double a character
            chars.insert(pos, chars[pos])
        elif typo_type == "qwerty":
            # QWERTY neighbor
            lower = chars[pos].lower()
            if lower in QWERTY_NEIGHBORS and QWERTY_NEIGHBORS[lower]:
                replacement = random.choice(QWERTY_NEIGHBORS[lower])
                chars[pos] = replacement.upper() if chars[pos].isupper() else replacement

        return "".join(chars)


class FastPunctuation:
    """Fast punctuation manipulation."""

    def __init__(
        self,
        terminal_modify_prob: float = 0.5,
        comma_modify_prob: float = 0.3,
    ):
        self.terminal_modify_prob = terminal_modify_prob
        self.comma_modify_prob = comma_modify_prob
        self.terminal_options = [".", "!", "?", "...", ""]

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions of text."""
        if not text or not text.strip():
            return [text] * n

        results = []
        for _ in range(n):
            results.append(self._modify_punctuation(text))
        return results

    def _modify_punctuation(self, text: str) -> str:
        """Modify punctuation in text."""
        text = text.rstrip()

        # Modify terminal punctuation
        if random.random() < self.terminal_modify_prob:
            text = self._modify_terminal(text)

        # Modify commas
        if random.random() < self.comma_modify_prob:
            text = self._modify_commas(text)

        return text

    def _modify_terminal(self, text: str) -> str:
        """Modify sentence-ending punctuation."""
        # Remove existing terminal punctuation
        while text and text[-1] in ".!?":
            text = text[:-1]

        # Add new terminal punctuation
        new_terminal = random.choice(self.terminal_options)
        return text + new_terminal

    def _modify_commas(self, text: str) -> str:
        """Add or remove commas."""
        if "," in text and random.random() < 0.5:
            # Remove a random comma
            comma_positions = [i for i, c in enumerate(text) if c == ","]
            if comma_positions:
                pos = random.choice(comma_positions)
                text = text[:pos] + text[pos + 1 :]
        else:
            # Add a comma after a random word
            words = text.split(" ")
            if len(words) > 3:
                # Insert comma after a word (not first or last)
                insert_idx = random.randint(1, len(words) - 2)
                words[insert_idx] = words[insert_idx].rstrip(",") + ","
                text = " ".join(words)

        return text


class FastTerminalPunctuation:
    """Fast terminal punctuation swapping."""

    def __init__(self):
        self.options = [".", "!", "?", "...", "..", ""]

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions with different endings."""
        if not text or not text.strip():
            return [text] * n

        # Strip existing terminal punctuation
        base_text = text.rstrip()
        while base_text and base_text[-1] in ".!?":
            base_text = base_text[:-1]

        results = []
        used_endings = set()

        for _ in range(n):
            # Pick a random ending we haven't used yet if possible
            available = [e for e in self.options if e not in used_endings]
            if not available:
                available = self.options

            ending = random.choice(available)
            used_endings.add(ending)
            results.append(base_text + ending)

        return results


class FastCommaVariation:
    """Fast comma insertion/removal."""

    def __init__(self):
        self.transitional_words = {
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
            "so",
            "now",
            "then",
            "also",
            "indeed",
        }

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions with comma variations."""
        if not text or not text.strip():
            return [text] * n

        results = []
        for _ in range(n):
            results.append(self._modify_commas(text))
        return results

    def _modify_commas(self, text: str) -> str:
        """Modify commas in text."""
        modification = random.choice(
            [
                "add_after_transitional",
                "remove_after_transitional",
                "toggle_oxford",
                "add_before_conjunction",
                "random_add",
                "random_remove",
            ]
        )

        if modification == "add_after_transitional":
            return self._add_comma_after_transitional(text)
        elif modification == "remove_after_transitional":
            return self._remove_comma_after_transitional(text)
        elif modification == "toggle_oxford":
            return self._toggle_oxford_comma(text)
        elif modification == "add_before_conjunction":
            return self._add_comma_before_conjunction(text)
        elif modification == "random_add":
            return self._random_add_comma(text)
        else:
            return self._random_remove_comma(text)

    def _add_comma_after_transitional(self, text: str) -> str:
        words = text.split(" ")
        for i, word in enumerate(words):
            clean_word = word.lower().strip(".,;:")
            if clean_word in self.transitional_words and not word.endswith(","):
                words[i] = word + ","
                break
        return " ".join(words)

    def _remove_comma_after_transitional(self, text: str) -> str:
        words = text.split(" ")
        for i, word in enumerate(words):
            clean_word = word.lower().rstrip(".,;:")
            if clean_word in self.transitional_words and word.endswith(","):
                words[i] = word.rstrip(",")
                break
        return " ".join(words)

    def _toggle_oxford_comma(self, text: str) -> str:
        # Try to add oxford comma
        pattern1 = r"(\w+),\s+(\w+)\s+and\s+"
        if re.search(pattern1, text):
            return re.sub(pattern1, r"\1, \2, and ", text, count=1)

        # Try to remove oxford comma
        pattern2 = r"(\w+),\s+(\w+),\s+and\s+"
        if re.search(pattern2, text):
            return re.sub(pattern2, r"\1, \2 and ", text, count=1)

        return text

    def _add_comma_before_conjunction(self, text: str) -> str:
        for conj in [" and ", " but ", " or "]:
            if conj in text:
                pos = text.find(conj)
                if pos > 0 and text[pos - 1] != ",":
                    return text[:pos] + "," + text[pos:]
        return text

    def _random_add_comma(self, text: str) -> str:
        words = text.split(" ")
        if len(words) > 3:
            idx = random.randint(1, len(words) - 2)
            if not words[idx].endswith(","):
                words[idx] = words[idx] + ","
        return " ".join(words)

    def _random_remove_comma(self, text: str) -> str:
        if "," in text:
            positions = [i for i, c in enumerate(text) if c == ","]
            if positions:
                pos = random.choice(positions)
                return text[:pos] + text[pos + 1 :]
        return text


# Registry mapping
FAST_AUGMENTATION_REGISTRY = {
    "noise_injection": FastNoiseInjection,
    "typo": FastTypo,
    "punctuation": FastPunctuation,
    "terminal_punct": FastTerminalPunctuation,
    "comma": FastCommaVariation,
}


def get_fast_augmentation(name: str, params: Optional[Dict[str, Any]] = None):
    """Get a fast augmentation instance."""
    if name not in FAST_AUGMENTATION_REGISTRY:
        return None

    cls = FAST_AUGMENTATION_REGISTRY[name]

    # Map config params to constructor args
    if params and name == "noise_injection":
        return cls(
            pct_chars_to_modify=params.get("pct_words_to_swap", 0.05),
        )
    elif params and name == "typo":
        return cls(
            pct_words_to_modify=params.get("pct_words_to_swap", 0.1),
        )

    return cls()


def is_fast_augmentation_available(name: str) -> bool:
    """Check if fast augmentation is available for given name."""
    return name in FAST_AUGMENTATION_REGISTRY
