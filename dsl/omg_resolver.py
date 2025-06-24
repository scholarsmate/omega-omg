"""
Entity resolver implementation for the OMG DSL.

This module provides compatibility imports for the reorganized resolver components.
The actual implementation has been split into focused modules in the resolver/ subdirectory
for better maintainability and reduced complexity.

For new code, import directly from dsl.resolver instead of this module.
"""

# Import all classes from the new modular structure
from dsl.resolver import (
    EntityResolver,
    HorizontalCanonicalizer,
    MetadataEnricher,
    OverlapResolver,
    RawMatchProcessor,
    ResolvedMatch,
    TokenizationFlags,
    Tokenizer,
    VerticalChildResolver,
)

# Maintain backward compatibility by exposing all classes at module level
__all__ = [
    "EntityResolver",
    "HorizontalCanonicalizer",
    "MetadataEnricher",
    "OverlapResolver",
    "RawMatchProcessor",
    "ResolvedMatch",
    "TokenizationFlags",
    "Tokenizer",
    "VerticalChildResolver",
]
