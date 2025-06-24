"""
OMG Resolver package - modular entity resolution components.

This package provides the core entity resolution functionality for the OMG DSL,
broken down into focused, maintainable modules:

- core: Core data structures (ResolvedMatch, TokenizationFlags)
- tokenizer: Text tokenization and token bag processing
- raw_processor: Raw match loading and validation
- overlap_resolver: Overlap resolution algorithm
- horizontal: Horizontal canonicalization (parent rule matching)
- vertical: Vertical child resolution (child-to-parent matching)
- metadata: Metadata enrichment (sentence/paragraph boundaries)
- entity_resolver: Main resolver orchestrating the full pipeline
"""

from .core import ResolvedMatch, TokenizationFlags
from .entity_resolver import EntityResolver
from .horizontal import HorizontalCanonicalizer
from .metadata import MetadataEnricher
from .overlap_resolver import OverlapResolver
from .raw_processor import RawMatchProcessor
from .tokenizer import Tokenizer
from .vertical import VerticalChildResolver

__all__ = [
    "ResolvedMatch",
    "TokenizationFlags",
    "Tokenizer",
    "RawMatchProcessor",
    "OverlapResolver",
    "HorizontalCanonicalizer",
    "VerticalChildResolver",
    "MetadataEnricher",
    "EntityResolver",
]
