"""
Core data structures for the OMG resolver.

Contains the fundamental data classes and flags used throughout the resolver system.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ResolvedMatch:
    """A match enriched with resolver information."""

    offset: int
    length: int
    rule: str
    match: str
    reference: Optional[int] = None
    sentence_end: Optional[int] = None
    paragraph_end: Optional[int] = None


class TokenizationFlags:
    """Flags that control how text is tokenized and normalized."""

    def __init__(self, ignore_case: bool = False, ignore_punctuation: bool = False):
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation


def extract_resolver_config(rule_config: Any) -> Dict[str, Any]:
    """
    Extract configuration dictionary from ResolverConfig object or dict.

    Args:
        rule_config: Either a ResolverConfig AST object or a dictionary

    Returns:
        Configuration dictionary with normalized structure
    """
    if hasattr(rule_config, "method"):  # It's a ResolverConfig AST object
        return {
            "method": rule_config.method,
            "flags": list(rule_config.flags),
            "args": list(rule_config.args),
            "optional_tokens": list(rule_config.optional_tokens),
        }
    # It's already a dictionary
    return rule_config
