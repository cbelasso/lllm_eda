from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional

from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
    WordSwapHomoglyphSwap,
    WordSwapInflections,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapWordNet,
)

from .punctuation_transforms import (
    CommaInsertion,
    PunctuationTransformation,
    SentenceTerminationSwap,
)


class TextAttackAugmentation(ABC):
    """Base class for TextAttack-based augmentations."""

    def __init__(self):
        self._cached_augmenter = None
        self._cached_params_hash = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this augmentation type."""
        pass

    @property
    def backend(self) -> str:
        return "textattack"

    @abstractmethod
    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        """Build the TextAttack Augmenter instance."""
        pass

    def _get_params_hash(self, params: Optional[Dict[str, Any]]) -> int:
        """Create a hash of params for caching."""
        if params is None:
            return hash(None)
        # Convert to sorted tuple of items for consistent hashing
        return hash(tuple(sorted((k, str(v)) for k, v in params.items())))

    def get_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        """Get or create cached augmenter."""
        params_hash = self._get_params_hash(params)

        if self._cached_augmenter is None or self._cached_params_hash != params_hash:
            self._cached_augmenter = self.build_augmenter(params)
            self._cached_params_hash = params_hash

        return self._cached_augmenter

    def augment(
        self, text: str, params: Optional[Dict[str, Any]] = None, n: int = 1
    ) -> List[str]:
        """
        Apply augmentation to text.

        Args:
            text: Input text to augment
            params: Optional parameters for the augmentation
            n: Number of augmented versions to generate

        Returns:
            List of augmented texts
        """
        if not text or not text.strip():
            return [text] * n

        augmenter = self.get_augmenter(params)
        results = []

        for _ in range(n):
            try:
                augmented = augmenter.augment(text)
                if augmented:
                    results.append(augmented[0])
                else:
                    results.append(text)
            except Exception:
                results.append(text)

        return results

    def augment_batch(
        self, texts: List[str], params: Optional[Dict[str, Any]] = None, n: int = 1
    ) -> List[List[str]]:
        """
        Apply augmentation to a batch of texts.

        Args:
            texts: List of input texts
            params: Optional parameters
            n: Number of augmented versions per text

        Returns:
            List of lists of augmented texts
        """
        return [self.augment(text, params, n) for text in texts]


# =============================================================================
# TRANSFORMATION REGISTRY
# =============================================================================

TRANSFORMATION_REGISTRY = {
    # Character-level transformations
    "qwerty": WordSwapQWERTY,
    "char_delete": WordSwapRandomCharacterDeletion,
    "char_insert": WordSwapRandomCharacterInsertion,
    "char_substitute": WordSwapRandomCharacterSubstitution,
    "char_swap": WordSwapNeighboringCharacterSwap,
    "homoglyph": WordSwapHomoglyphSwap,
    # Word-level transformations
    "wordnet": WordSwapWordNet,
    "inflection": WordSwapInflections,
    "contract": WordSwapContract,
    "extend": WordSwapExtend,
    "change_name": WordSwapChangeName,
    "change_number": WordSwapChangeNumber,
    "change_location": WordSwapChangeLocation,
    # Punctuation transformations (custom)
    "punctuation": PunctuationTransformation,
    "terminal_punct": SentenceTerminationSwap,
    "comma": CommaInsertion,
}

CONSTRAINT_PRESETS = {
    "default": [
        RepeatModification(),
        StopwordModification(),
    ],
    "strict": [
        RepeatModification(),
        StopwordModification(),
        MinWordLength(min_length=4),
    ],
    "minimal": [
        RepeatModification(),
    ],
    "none": [],
}


def build_transformation(name: str, **kwargs) -> Any:
    """Build a transformation by name with optional kwargs."""
    if name not in TRANSFORMATION_REGISTRY:
        raise ValueError(
            f"Unknown transformation: {name}. Available: {list(TRANSFORMATION_REGISTRY.keys())}"
        )
    return TRANSFORMATION_REGISTRY[name](**kwargs)


def build_composite_transformation(names: List[str]) -> CompositeTransformation:
    """Build a composite transformation from a list of names."""
    transformations = [build_transformation(name) for name in names]
    return CompositeTransformation(transformations)


# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================


class TextAttackNoiseInjection(TextAttackAugmentation):
    """Character-level noise injection using TextAttack."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "noise_injection"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        # Default transformations for noise
        transformation_names = params.get(
            "transformations", ["qwerty", "char_delete", "char_swap"]
        )
        pct_words_to_swap = params.get("pct_words_to_swap", 0.15)
        transformations_per_example = params.get("transformations_per_example", 3)
        constraint_preset = params.get("constraints", "minimal")

        transformation = build_composite_transformation(transformation_names)
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["minimal"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackSynonymReplace(TextAttackAugmentation):
    """Synonym replacement using WordNet."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "synonym_replace"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        pct_words_to_swap = params.get("pct_words_to_swap", 0.2)
        transformations_per_example = params.get("transformations_per_example", 3)
        constraint_preset = params.get("constraints", "default")

        transformation = WordSwapWordNet()
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["default"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackHomoglyph(TextAttackAugmentation):
    """Homoglyph character substitution for adversarial robustness."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "homoglyph"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        pct_words_to_swap = params.get("pct_words_to_swap", 0.1)
        transformations_per_example = params.get("transformations_per_example", 2)
        constraint_preset = params.get("constraints", "minimal")

        transformation = WordSwapHomoglyphSwap()
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["minimal"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackTypo(TextAttackAugmentation):
    """Realistic keyboard typo simulation."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "typo"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        pct_words_to_swap = params.get("pct_words_to_swap", 0.1)
        transformations_per_example = params.get("transformations_per_example", 2)
        constraint_preset = params.get("constraints", "minimal")

        # Mix of QWERTY errors and character swaps
        transformation = CompositeTransformation(
            [
                WordSwapQWERTY(),
                WordSwapNeighboringCharacterSwap(),
            ]
        )
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["minimal"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackMorphological(TextAttackAugmentation):
    """Morphological variations (inflections, contractions)."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "morphological"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        pct_words_to_swap = params.get("pct_words_to_swap", 0.15)
        transformations_per_example = params.get("transformations_per_example", 2)
        constraint_preset = params.get("constraints", "default")

        transformation = CompositeTransformation(
            [
                WordSwapInflections(),
                WordSwapContract(),
                WordSwapExtend(),
            ]
        )
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["default"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackEntitySwap(TextAttackAugmentation):
    """Entity swapping (names, numbers, locations)."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "entity_swap"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        pct_words_to_swap = params.get("pct_words_to_swap", 0.3)
        transformations_per_example = params.get("transformations_per_example", 2)
        constraint_preset = params.get("constraints", "minimal")

        # Select which entity types to swap
        swap_types = params.get("entity_types", ["name", "number", "location"])

        transformations = []
        if "name" in swap_types:
            transformations.append(WordSwapChangeName())
        if "number" in swap_types:
            transformations.append(WordSwapChangeNumber())
        if "location" in swap_types:
            transformations.append(WordSwapChangeLocation())

        if not transformations:
            transformations = [WordSwapChangeName()]

        transformation = CompositeTransformation(transformations)
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["minimal"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackComposite(TextAttackAugmentation):
    """Fully configurable composite augmentation."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "composite"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        transformation_names = params.get("transformations", ["qwerty", "wordnet"])
        pct_words_to_swap = params.get("pct_words_to_swap", 0.15)
        transformations_per_example = params.get("transformations_per_example", 3)
        constraint_preset = params.get("constraints", "default")

        transformation = build_composite_transformation(transformation_names)
        constraints = CONSTRAINT_PRESETS.get(constraint_preset, CONSTRAINT_PRESETS["default"])

        return Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example,
        )


class TextAttackPunctuation(TextAttackAugmentation):
    """Full punctuation manipulation (add, remove, swap, spacing)."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "punctuation"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}

        add_prob = params.get("add_prob", 0.3)
        remove_prob = params.get("remove_prob", 0.3)
        swap_prob = params.get("swap_prob", 0.2)
        ellipsis_prob = params.get("ellipsis_prob", 0.1)
        space_prob = params.get("space_prob", 0.1)
        transformations_per_example = params.get("transformations_per_example", 2)

        transformation = PunctuationTransformation(
            add_prob=add_prob,
            remove_prob=remove_prob,
            swap_prob=swap_prob,
            ellipsis_prob=ellipsis_prob,
            space_prob=space_prob,
        )

        return Augmenter(
            transformation=transformation,
            constraints=[],  # No word-level constraints for punctuation
            pct_words_to_swap=1.0,  # Punctuation transform handles its own logic
            transformations_per_example=transformations_per_example,
        )


class TextAttackTerminalPunctuation(TextAttackAugmentation):
    """Focused on sentence-ending punctuation variations."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "terminal_punct"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}
        transformations_per_example = params.get("transformations_per_example", 3)

        transformation = SentenceTerminationSwap()

        return Augmenter(
            transformation=transformation,
            constraints=[],
            pct_words_to_swap=1.0,
            transformations_per_example=transformations_per_example,
        )


class TextAttackCommaVariation(TextAttackAugmentation):
    """Comma insertion and removal variations."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "comma"

    def build_augmenter(self, params: Optional[Dict[str, Any]] = None) -> Augmenter:
        params = params or {}
        insertion_prob = params.get("insertion_prob", 0.5)
        transformations_per_example = params.get("transformations_per_example", 2)

        transformation = CommaInsertion(insertion_prob=insertion_prob)

        return Augmenter(
            transformation=transformation,
            constraints=[],
            pct_words_to_swap=1.0,
            transformations_per_example=transformations_per_example,
        )


# =============================================================================
# REGISTRY
# =============================================================================

TEXTATTACK_REGISTRY: Dict[str, TextAttackAugmentation] = {
    "noise_injection": TextAttackNoiseInjection(),
    "synonym_replace": TextAttackSynonymReplace(),
    "homoglyph": TextAttackHomoglyph(),
    "typo": TextAttackTypo(),
    "morphological": TextAttackMorphological(),
    "entity_swap": TextAttackEntitySwap(),
    "composite": TextAttackComposite(),
    # Punctuation augmentations
    "punctuation": TextAttackPunctuation(),
    "terminal_punct": TextAttackTerminalPunctuation(),
    "comma": TextAttackCommaVariation(),
}


def get_textattack_augmentation(name: str) -> TextAttackAugmentation:
    """Get a TextAttack augmentation by name."""
    if name not in TEXTATTACK_REGISTRY:
        raise ValueError(
            f"Unknown TextAttack augmentation: {name}. "
            f"Available: {list(TEXTATTACK_REGISTRY.keys())}"
        )
    return TEXTATTACK_REGISTRY[name]


def is_textattack_available(name: str) -> bool:
    """Check if a TextAttack augmentation is available for the given name."""
    return name in TEXTATTACK_REGISTRY
