# mypy: disable-error-code=annotation-unchecked
"""
Text tokenization functionality for the OMG resolver.

Handles tokenization of text with support for optional tokens and normalization flags.
Implements Step 2 of the Plan of Attack: tokenization with optional-token stripping.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Set

from .core import TokenizationFlags

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Handles tokenization of text with support for optional tokens and normalization flags.

    Implements Step 2 of the Plan of Attack: tokenization with optional-token stripping.
    """

    def __init__(self):
        self._optional_tokens_cache: Dict[str, Set[str]] = {}
        self._token_bag_cache: Dict[str, str] = {}  # Cache for tokenized results

    def load_optional_tokens(
        self, file_path: str, dsl_file_path: Optional[str] = None
    ) -> Set[str]:
        """
        Load optional tokens from a file, with caching.

        Args:
            file_path: Path to the optional tokens file
            dsl_file_path: Path to DSL file for resolving relative paths

        Returns:
            Set of optional tokens (normalized to lowercase)
        """
        # Cache key uses string for compatibility
        abs_file = os.path.abspath(file_path)
        abs_dsl = os.path.abspath(dsl_file_path) if dsl_file_path else ""
        cache_key = f"{abs_file}|{abs_dsl}"
        if cache_key in self._optional_tokens_cache:
            return self._optional_tokens_cache[cache_key]

        # Resolve relative paths
        if not os.path.isabs(file_path) and dsl_file_path:
            file_path = os.path.join(os.path.dirname(dsl_file_path), file_path)

        optional_tokens = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    token = line.strip()
                    if token and not token.startswith(
                        "#"
                    ):  # Skip comments and empty lines
                        optional_tokens.add(token.lower())  # Normalize to lowercase

            logger.debug(
                "Loaded %s optional tokens from %s", len(optional_tokens), file_path
            )

        except OSError as e:
            logger.warning("Failed to load optional tokens from %s: %s", file_path, e)

        self._optional_tokens_cache[cache_key] = optional_tokens
        return optional_tokens

    def tokenize(
        self,
        text: str,
        flags: Optional[TokenizationFlags] = None,
        optional_tokens: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Tokenize text with normalization flags and optional token filtering. Use cache for repeated calls.

        Args:
            text: Input text to tokenize
            flags: Tokenization flags for normalization
            optional_tokens: Set of tokens to filter out (optional)

        Returns:
            List of tokens after normalization and filtering
        """
        if not text:
            return []

        # Create a robust cache key for this tokenization request
        cache_key = f"{text}|{getattr(flags, 'ignore_case', False)}|{getattr(flags, 'ignore_punctuation', False)}"
        # Check cache first
        if cache_key in self._token_bag_cache:
            cached_result = self._token_bag_cache[cache_key]
            tokens = cached_result.split() if cached_result else []
        else:
            # Apply normalization flags
            normalized_text = text

            if flags and getattr(flags, "ignore_case", False):
                normalized_text = normalized_text.lower()

            if flags and getattr(flags, "ignore_punctuation", False):
                # Remove punctuation but preserve spaces
                normalized_text = re.sub(r"[^\w\s]", "", normalized_text)

            # Extract word tokens using \w+ pattern
            tokens = re.findall(r"\w+", normalized_text)

            # Cache the result
            self._token_bag_cache[cache_key] = " ".join(tokens)

        # Optional token filtering (case-sensitive or insensitive based on flags)
        if optional_tokens:
            if flags and getattr(flags, "ignore_case", False):
                # Lowercase all tokens and filter
                tokens = [
                    t
                    for t in tokens
                    if t.lower() not in {ot.lower() for ot in optional_tokens}
                ]
            else:
                tokens = [t for t in tokens if t not in optional_tokens]

        return tokens

    def get_token_bag_key(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens into a sorted, space-joined string for grouping.

        Args:
            tokens: List of tokens

        Returns:
            Sorted, space-joined string representation
        """
        return " ".join(sorted(tokens))  # type: ignore[annotation-unchecked]
