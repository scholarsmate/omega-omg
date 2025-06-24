"""
Horizontal canonicalization implementation for the OMG resolver.

This module handles horizontal canonicalization for parent rule matches,
implementing Step 4 of the Plan of Attack: canonicalize parent matches
by grouping them by token bags and assigning references using configurable
matching algorithms (exact, fuzzy).
"""

import logging
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Set

from .core import ResolvedMatch, TokenizationFlags, extract_resolver_config
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class HorizontalCanonicalizer:
    """
    Handles horizontal canonicalization for parent rule matches.

    Implements Step 4 of the Plan of Attack: canonicalize parent matches
    by grouping them by token bags and assigning references using configurable
    matching algorithms (exact, fuzzy).
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize with a tokenizer instance.
        Args:
            tokenizer: Tokenizer instance for processing match text
        """
        self.tokenizer = tokenizer
        # Cache for resolved match references
        self._match_content_cache: Dict[str, Optional[int]] = {}
        # Custom matching algorithms
        self._custom_algorithms: Dict[str, Callable] = {}

    def canonicalize_matches(
        self,
        matches: List[ResolvedMatch],
        resolver_config: Optional[Dict] = None,
        dsl_file_path: Optional[str] = None,
    ) -> List[ResolvedMatch]:
        """
        Canonicalize parent rule matches by grouping by token bags.

        Args:
            matches: List of ResolvedMatch objects
            resolver_config: Configuration for resolver methods and optional tokens
            dsl_file_path: Path to DSL file for resolving relative paths

        Returns:
            List of ResolvedMatch objects with canonical references assigned
        """
        if not matches:
            return matches

        # Separate parent and child rule matches
        parent_matches = [m for m in matches if "." not in m.rule]
        child_matches = [m for m in matches if "." in m.rule]

        logger.info("Canonicalizing %s parent rule matches", len(parent_matches))

        # Group parent matches by rule
        matches_by_rule: Dict[str, List[ResolvedMatch]] = {}
        for match in parent_matches:
            if match.rule not in matches_by_rule:
                matches_by_rule[match.rule] = []
            matches_by_rule[match.rule].append(match)

        canonicalized_matches: List[ResolvedMatch] = []
        # Process each parent rule separately
        for rule, rule_matches in matches_by_rule.items():
            logger.debug(
                "Processing rule '%s' with %s matches", rule, len(rule_matches)
            )

            # Get resolver configuration for this rule
            rule_config = None
            if resolver_config and rule in resolver_config:
                rule_config = resolver_config[rule]

            # Canonicalize matches for this rule
            rule_canonicalized = self._canonicalize_rule_matches(
                rule_matches, rule_config, dsl_file_path
            )
            canonicalized_matches.extend(rule_canonicalized)

        # Add child matches back (unchanged at this step)
        canonicalized_matches.extend(child_matches)

        # Sort by offset to maintain order
        canonicalized_matches.sort(key=lambda m: m.offset)

        logger.info(
            "Horizontal canonicalization complete: %s matches",
            len(canonicalized_matches),
        )
        return canonicalized_matches

    def _canonicalize_rule_matches(
        self,
        matches: List[ResolvedMatch],
        rule_config: Optional[Any] = None,
        dsl_file_path: Optional[str] = None,
    ) -> List[ResolvedMatch]:
        """
        Canonicalize matches for a single rule using configured matching algorithm.

        Args:
            matches: List of matches for the same rule
            rule_config: Configuration for this rule (method, flags, threshold)
            dsl_file_path: Path to DSL file for resolving relative paths
        Returns:
            List of matches with canonical references assigned
        """
        if len(matches) <= 1:
            # No canonicalization needed for single match
            return matches

        # Extract configuration parameters
        method = "exact"  # Default matching method
        threshold = 0.85  # Default threshold for fuzzy matching
        flags = TokenizationFlags(ignore_case=True, ignore_punctuation=False)
        optional_tokens = set()

        if rule_config:
            config_dict = extract_resolver_config(rule_config)

            # Extract method and threshold if available
            if "method" in config_dict:
                method = config_dict["method"]
            if "threshold" in config_dict:
                threshold = config_dict["threshold"]

            # Extract flags if available
            if "ignore_case" in config_dict:
                flags.ignore_case = config_dict["ignore_case"]
            if "ignore_punctuation" in config_dict:
                flags.ignore_punctuation = config_dict["ignore_punctuation"]

            # Load optional tokens if available
            if "optional_tokens" in config_dict and config_dict["optional_tokens"]:
                # Take the first optional tokens file
                optional_tokens_file = config_dict["optional_tokens"][0]
                optional_tokens = self.tokenizer.load_optional_tokens(
                    optional_tokens_file, dsl_file_path
                )  # Apply the configured matching algorithm
        if method == "exact":
            logger.debug("Canonicalizing rule matches using method '%s'", method)
            return self._canonicalize_exact(matches, flags, optional_tokens)
        if method == "fuzzy":
            logger.debug(
                "Canonicalizing rule matches using method '%s' with threshold %s",
                method,
                threshold,
            )
            return self._canonicalize_fuzzy(matches, flags, optional_tokens, threshold)
        if hasattr(self, "_custom_algorithms") and method in self._custom_algorithms:
            # Apply custom algorithm
            kwargs = (
                {"threshold": threshold}
                if rule_config and "threshold" in rule_config
                else {}
            )
            return self._apply_custom_algorithm(
                method, matches, flags, optional_tokens, **kwargs
            )
        # Fallback to exact matching
        logger.warning("Unknown matching method '%s', falling back to exact", method)
        return self._canonicalize_exact(matches, flags, optional_tokens)

    def _canonicalize_exact(
        self,
        matches: List[ResolvedMatch],
        flags: TokenizationFlags,
        optional_tokens: Set[str],
    ) -> List[ResolvedMatch]:
        """
        Canonicalize matches using exact token bag matching.

        Args:
            matches: List of matches to canonicalize
            flags: Tokenization flags
            optional_tokens: Set of optional tokens to ignore

        Returns:
            List of matches with canonical references assigned
        """
        # Group matches by exact token bags
        groups_by_token_bag: Dict[str, List[ResolvedMatch]] = {}

        for match in matches:
            # Check cache first for this match content
            cache_key = self._get_match_cache_key(
                match.match, match.rule, flags, optional_tokens, "exact"
            )

            # If we've seen this exact content before, reuse the result
            if cache_key in self._match_content_cache:
                cached_reference = self._match_content_cache[cache_key]
                if cached_reference is not None:
                    # Create a copy of the match with the cached reference
                    match.reference = cached_reference
                    logger.debug(
                        "Cache hit: Match '%s' -> references %s",
                        match.match,
                        cached_reference,
                    )
                # If cached_reference is None, this match should be canonical
                # Continue with normal processing to group it

            # Tokenize the match text (do NOT pass optional_tokens; filter after)
            tokens = self.tokenizer.tokenize(match.match, flags)
            # Remove optional tokens if present
            if optional_tokens:
                tokens = [t for t in tokens if t not in optional_tokens]
            token_bag_key = self.tokenizer.get_token_bag_key(tokens)

            if token_bag_key not in groups_by_token_bag:
                groups_by_token_bag[token_bag_key] = []
            groups_by_token_bag[token_bag_key].append(match)

            logger.debug(
                "Match '%s' -> tokens %s -> key '%s'",
                match.match,
                tokens,
                token_bag_key,
            )

        logger.debug(
            "Exact matching: grouped into %s token bag groups", len(groups_by_token_bag)
        )

        # Canonicalize each group
        canonicalized_matches = []

        for token_bag_key, group_matches in groups_by_token_bag.items():
            if len(group_matches) == 1:
                # Single match - no canonicalization needed, but update cache
                single_match = group_matches[0]
                cache_key = self._get_match_cache_key(
                    single_match.match,
                    single_match.rule,
                    flags,
                    optional_tokens,
                    "exact",
                )
                # Single matches are canonical (reference = None)
                self._match_content_cache[cache_key] = None
                canonicalized_matches.extend(group_matches)
            else:
                # Multiple matches - canonicalize
                logger.debug(
                    "Canonicalizing exact group with key '%s': %s matches",
                    token_bag_key,
                    len(group_matches),
                )
                canonicalized_group = self._canonicalize_group_with_cache(
                    group_matches, flags, optional_tokens, "exact"
                )
                canonicalized_matches.extend(canonicalized_group)

        return canonicalized_matches

    def _canonicalize_fuzzy(
        self,
        matches: List[ResolvedMatch],
        flags: TokenizationFlags,
        optional_tokens: Set[str],
        threshold: float,
    ) -> List[ResolvedMatch]:
        """
        Canonicalize matches using fuzzy token bag matching with edit distance.

        Args:
            matches: List of matches to canonicalize
            flags: Tokenization flags
            optional_tokens: Set of optional tokens to ignore
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            List of matches with canonical references assigned
        """
        # pylint: disable=too-many-locals,too-many-branches
        # Create a canonical token map for fuzzy matching
        token_map: Dict[str, ResolvedMatch] = {}

        def get_canonical_key(match: ResolvedMatch) -> str:
            """Get the canonical token bag key for a match."""
            tokens = self.tokenizer.tokenize(match.match, flags)
            if optional_tokens:
                tokens = [t for t in tokens if t not in optional_tokens]
            return self.tokenizer.get_token_bag_key(tokens)

        # Sort matches by length (desc), then offset (asc) to prioritize canonical selection
        matches_sorted = sorted(matches, key=lambda m: (-m.length, m.offset))

        # First pass: exact matches (check cache first)
        for match in matches_sorted:
            # Check cache first for this match content
            cache_key = self._get_match_cache_key(
                match.match, match.rule, flags, optional_tokens, "fuzzy", threshold
            )

            # If we've seen this exact content before, reuse the result
            if cache_key in self._match_content_cache:
                cached_reference = self._match_content_cache[cache_key]
                if cached_reference is not None:
                    # Create a copy of the match with the cached reference
                    match.reference = cached_reference
                    logger.debug(
                        "Fuzzy cache hit: Match '%s' -> references %s",
                        match.match,
                        cached_reference,
                    )
                    continue
                # If cached_reference is None, this match should be canonical
                # Continue with normal processing

            key = get_canonical_key(match)
            if key in token_map:
                # Exact match found - assign reference
                match.reference = token_map[key].offset
                logger.debug(
                    "Exact match: '%s' -> references %s",
                    match.match,
                    token_map[key].offset,
                )
            else:
                # New canonical match
                token_map[key] = match
                logger.debug("New canonical: '%s' with key '%s'", match.match, key)

        # Second pass: fuzzy matches for unresolved matches
        canonical_keys = list(token_map.keys())

        for match in matches_sorted:
            if match.reference is not None:
                continue  # Already resolved

            match_key = get_canonical_key(match)
            best_similarity = 0.0
            best_canonical = None

            # Compare against all canonical keys
            for canonical_key in canonical_keys:
                similarity = SequenceMatcher(None, match_key, canonical_key).ratio()
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_canonical = token_map[canonical_key]

            if best_canonical:
                # Fuzzy match found
                match.reference = best_canonical.offset
                logger.debug(
                    "Fuzzy match: '%s' -> references %s (similarity: %.3f)",
                    match.match,
                    best_canonical.offset,
                    best_similarity,
                )
            else:  # No match found - this becomes a new canonical
                token_map[match_key] = match
                canonical_keys.append(match_key)
                logger.debug(
                    "New fuzzy canonical: '%s' with key '%s'", match.match, match_key
                )

        # Update cache for all matches processed in this batch
        for match in matches_sorted:
            cache_key = self._get_match_cache_key(
                match.match, match.rule, flags, optional_tokens, "fuzzy", threshold
            )
            if match.reference is None:
                # Canonical match
                self._match_content_cache[cache_key] = None
            else:
                # Non-canonical match
                self._match_content_cache[cache_key] = match.reference

            logger.debug(
                "Cached fuzzy match '%s' -> %s",
                match.match,
                self._match_content_cache[cache_key],
            )

        logger.debug(
            "Fuzzy matching: created %s canonical groups with threshold %s",
            len(token_map),
            threshold,
        )
        return matches_sorted

    def add_matching_algorithm(self, name: str, algorithm_func: Callable):
        """
        Add a new matching algorithm for future extensibility.

        Args:
            name: Name of the algorithm (e.g., "metaphone", "regex")
            algorithm_func: Function that takes (matches, flags, optional_tokens, **kwargs)
              and returns matches
        """
        if not hasattr(self, "_custom_algorithms"):
            self._custom_algorithms = {}
        self._custom_algorithms[name] = algorithm_func
        logger.info("Registered custom matching algorithm: %s", name)

    def _apply_custom_algorithm(
        self,
        algorithm_name: str,
        matches: List[ResolvedMatch],
        flags: TokenizationFlags,
        optional_tokens: Set[str],
        **kwargs,
    ) -> List[ResolvedMatch]:
        """
        Apply a custom matching algorithm.

        Args:
            algorithm_name: Name of the custom algorithm
            matches: List of matches to canonicalize
            flags: Tokenization flags
            optional_tokens: Set of optional tokens to ignore
            **kwargs: Additional parameters for the algorithm

        Returns:
            List of matches with canonical references assigned
        """
        if (
            not hasattr(self, "_custom_algorithms")
            or algorithm_name not in self._custom_algorithms
        ):
            logger.error("Unknown custom algorithm: %s", algorithm_name)
            return self._canonicalize_exact(matches, flags, optional_tokens)

        algorithm_func = self._custom_algorithms[algorithm_name]
        return algorithm_func(matches, flags, optional_tokens, **kwargs)

    def _canonicalize_group(self, matches: List[ResolvedMatch]) -> List[ResolvedMatch]:
        """
        Canonicalize a group of matches with the same token bag.

        Sort by length (desc), then offset (asc).
        The first becomes canonical, others reference it.

        Args:
            matches: List of matches with same token bag

        Returns:
            List of matches with canonical references assigned
        """
        # Early exit for single match - no canonicalization needed
        if len(matches) == 1:
            return matches

        # Sort by length (desc), then offset (asc)
        sorted_matches = sorted(matches, key=lambda m: (-m.length, m.offset))

        # First match becomes canonical
        canonical_match = sorted_matches[0]
        canonical_offset = canonical_match.offset

        logger.debug(
            "Canonical match: '%s' at offset %s",
            canonical_match.match,
            canonical_offset,
        )

        result = []

        for match in sorted_matches:
            # Create a copy to avoid modifying original
            canonicalized_match = ResolvedMatch(
                offset=match.offset,
                length=match.length,
                rule=match.rule,
                match=match.match,
                reference=match.reference,  # Preserve existing reference if any
                sentence_end=match.sentence_end,
                paragraph_end=match.paragraph_end,
            )

            if match is canonical_match:
                # Canonical match has no reference (or keeps existing reference)
                pass
            else:
                # Non-canonical match references the canonical one
                canonicalized_match.reference = canonical_offset
                logger.debug(
                    "Match '%s' at %s -> references %s",
                    match.match,
                    match.offset,
                    canonical_offset,
                )

            result.append(canonicalized_match)

        return result

    def _canonicalize_group_with_cache(
        self,
        matches: List[ResolvedMatch],
        flags: TokenizationFlags,
        optional_tokens: Set[str],
        method: str,
        threshold: float = 0.85,
    ) -> List[ResolvedMatch]:
        """
        Canonicalize a group of matches and update the content cache.

        Args:
            matches: List of matches with same token bag
            flags: Tokenization flags
            optional_tokens: Set of optional tokens
            method: Matching method for cache key generation
            threshold: Threshold for cache key generation

        Returns:
            List of matches with canonical references assigned
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments# Call the existing canonicalization logic
        result = self._canonicalize_group(matches)

        # Cache the resolution outcome for each match
        for match in result:
            cache_key = self._get_match_cache_key(
                match.match, match.rule, flags, optional_tokens, method, threshold
            )
            if match.reference is None:
                # Canonical match
                self._match_content_cache[cache_key] = None
            else:  # Non-canonical match
                self._match_content_cache[cache_key] = match.reference
            logger.debug(
                "Cached match '%s' -> %s",
                match.match,
                self._match_content_cache[cache_key],
            )
        return result

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _get_match_cache_key(
        self,
        match_text: str,
        rule: str,
        flags: TokenizationFlags,
        optional_tokens: Set[str],
        method: str,
        threshold: float = 0.85,
    ) -> str:
        """
        Generate a cache key for match content processing.

        Args:
            match_text: The text content of the match
            rule: Rule name for the match
            flags: Tokenization flags
            optional_tokens: Set of optional tokens
            method: Matching method (exact, fuzzy)
            threshold: Threshold for fuzzy matching

        Returns:
            Cache key string for this match processing configuration
        """
        optional_key = str(sorted(optional_tokens)) if optional_tokens else "none"
        return (
            f"{match_text}|{rule}|{flags.ignore_case}|{flags.ignore_punctuation}|{optional_key}|"
            f"{method}|{threshold}"
        )
