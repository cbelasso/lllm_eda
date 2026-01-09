from typing import Union

from .base import BaseAugmentation, ReferenceBasedAugmentation, SimpleAugmentation
from .compression import Compression
from .elaboration import Elaboration
from .formality_shift import FormalityShift
from .noise_injection import NoiseInjection
from .paraphrase import Paraphrase
from .punctuation import Punctuation
from .punctuation_transforms import (
    CommaInsertion,
    PunctuationTransformation,
    SentenceTerminationSwap,
)
from .sentence_restructure import SentenceRestructure
from .style_rewrite import StyleRewrite
from .synonym_replace import SynonymReplace
from .textattack_base import (
    CONSTRAINT_PRESETS,
    TEXTATTACK_REGISTRY,
    TRANSFORMATION_REGISTRY,
    TextAttackAugmentation,
    TextAttackCommaVariation,
    TextAttackComposite,
    TextAttackEntitySwap,
    TextAttackHomoglyph,
    TextAttackMorphological,
    TextAttackNoiseInjection,
    TextAttackPunctuation,
    TextAttackSynonymReplace,
    TextAttackTerminalPunctuation,
    TextAttackTypo,
    build_composite_transformation,
    build_transformation,
    get_textattack_augmentation,
    is_textattack_available,
)
from .typo import Typo

# Registry of LLM-based augmentations
LLM_AUGMENTATION_REGISTRY = {
    "style_rewrite": StyleRewrite(),
    "synonym_replace_llm": SynonymReplace(),
    "sentence_restructure": SentenceRestructure(),
    "noise_injection_llm": NoiseInjection(),
    "paraphrase": Paraphrase(),
    "formality_shift": FormalityShift(),
    "elaboration": Elaboration(),
    "compression": Compression(),
    "punctuation_llm": Punctuation(),
    "typo_llm": Typo(),
}

# Combined registry for backward compatibility
AUGMENTATION_REGISTRY = LLM_AUGMENTATION_REGISTRY.copy()

# Default backend mapping - which backend to use when not specified
DEFAULT_BACKEND_MAPPING = {
    "style_rewrite": "llm",
    "paraphrase": "llm",
    "sentence_restructure": "llm",
    "formality_shift": "llm",
    "elaboration": "llm",
    "compression": "llm",
    "punctuation_llm": "llm",
    "typo_llm": "llm",
    "noise_injection_llm": "llm",
    "synonym_replace_llm": "llm",
    # These can be done deterministically
    "noise_injection": "textattack",
    "synonym_replace": "textattack",
    # TextAttack-only operations
    "homoglyph": "textattack",
    "typo": "textattack",
    "morphological": "textattack",
    "entity_swap": "textattack",
    "composite": "textattack",
    # Punctuation operations
    "punctuation": "textattack",
    "terminal_punct": "textattack",
    "comma": "textattack",
}


def get_default_backend(name: str) -> str:
    """Get the default backend for an augmentation type."""
    return DEFAULT_BACKEND_MAPPING.get(name, "llm")


def get_augmentation(
    name: str, backend: str = None
) -> Union[BaseAugmentation, TextAttackAugmentation]:
    """
    Get an augmentation by name, optionally specifying backend.

    Args:
        name: Name of the augmentation
        backend: "llm", "textattack", or None (use default)

    Returns:
        Augmentation instance (either LLM-based or TextAttack-based)
    """
    if backend is None:
        backend = get_default_backend(name)

    if backend == "textattack":
        if name in TEXTATTACK_REGISTRY:
            return TEXTATTACK_REGISTRY[name]
        else:
            raise ValueError(
                f"TextAttack backend not available for '{name}'. "
                f"Available: {list(TEXTATTACK_REGISTRY.keys())}"
            )
    elif backend == "llm":
        if name in LLM_AUGMENTATION_REGISTRY:
            return LLM_AUGMENTATION_REGISTRY[name]
        else:
            raise ValueError(
                f"LLM backend not available for '{name}'. "
                f"Available: {list(LLM_AUGMENTATION_REGISTRY.keys())}"
            )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'llm' or 'textattack'.")


def is_llm_augmentation(name: str, backend: str = None) -> bool:
    """Check if an augmentation should use the LLM backend."""
    if backend is not None:
        return backend == "llm"
    return get_default_backend(name) == "llm"


def is_textattack_augmentation(name: str, backend: str = None) -> bool:
    """Check if an augmentation should use the TextAttack backend."""
    if backend is not None:
        return backend == "textattack"
    return get_default_backend(name) == "textattack"


def list_available_augmentations() -> dict:
    """List all available augmentations grouped by backend."""
    return {
        "llm": list(LLM_AUGMENTATION_REGISTRY.keys()),
        "textattack": list(TEXTATTACK_REGISTRY.keys()),
        "default_backends": DEFAULT_BACKEND_MAPPING.copy(),
    }


__all__ = [
    # Base classes
    "BaseAugmentation",
    "ReferenceBasedAugmentation",
    "SimpleAugmentation",
    "TextAttackAugmentation",
    # LLM augmentations
    "StyleRewrite",
    "SynonymReplace",
    "SentenceRestructure",
    "NoiseInjection",
    "Paraphrase",
    "FormalityShift",
    "Elaboration",
    "Compression",
    # TextAttack augmentations
    "TextAttackNoiseInjection",
    "TextAttackSynonymReplace",
    "TextAttackHomoglyph",
    "TextAttackTypo",
    "TextAttackMorphological",
    "TextAttackEntitySwap",
    "TextAttackComposite",
    "TextAttackPunctuation",
    "TextAttackTerminalPunctuation",
    "TextAttackCommaVariation",
    # Punctuation transformations
    "PunctuationTransformation",
    "SentenceTerminationSwap",
    "CommaInsertion",
    # Registries
    "AUGMENTATION_REGISTRY",
    "LLM_AUGMENTATION_REGISTRY",
    "TEXTATTACK_REGISTRY",
    "TRANSFORMATION_REGISTRY",
    "CONSTRAINT_PRESETS",
    "DEFAULT_BACKEND_MAPPING",
    # Functions
    "get_augmentation",
    "get_default_backend",
    "get_textattack_augmentation",
    "is_textattack_available",
    "is_llm_augmentation",
    "is_textattack_augmentation",
    "list_available_augmentations",
    "build_transformation",
    "build_composite_transformation",
]
