"""
Vertical child resolution implementation for the OMG resolver.

This module handles vertical child resolution for child rule matches,
implementing Step 5 of the Plan of Attack: resolve child matches to canonical parent matches
using subset or fuzzy token matching.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

from .core import ResolvedMatch, TokenizationFlags, extract_resolver_config
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class VerticalChildResolver:
    """
    Handles vertical child resolution for child rule matches.

    Implements Step 5 of the Plan of Attack: resolve child matches to canonical parent matches
    using subset or fuzzy token matching.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize with tokenizer for consistent token processing.

        Args:
            tokenizer: Tokenizer instance for text processing
        """
        self.tokenizer = tokenizer
        # Cache for child-to-parent match resolution: child_cache_key -> parent_offset or None
        self._child_resolution_cache: Dict[str, Optional[int]] = {}

    def resolve_child_matches(
        self,
        matches: List[ResolvedMatch],
        resolver_config: Optional[Dict] = None,
        dsl_file_path: Optional[str] = None,
    ) -> List[ResolvedMatch]:
        """
        Resolve child rule matches to canonical parent matches.

        Args:
            matches: List of ResolvedMatch objects (both parents and children)
            resolver_config: Configuration for resolver methods and optional tokens
            dsl_file_path: Path to DSL file for resolving relative paths

        Returns:
            List of ResolvedMatch objects with child references resolved or children discarded
        """
        # pylint: disable=too-many-locals,too-many-branches
        if not matches:
            return matches

        # Separate parent and child matches
        parent_matches = [m for m in matches if "." not in m.rule]
        child_matches = [m for m in matches if "." in m.rule]

        if not child_matches:
            # No child matches to resolve
            return matches

        logger.info(
            "Resolving %s child rule matches against %s parent matches",
            len(child_matches),
            len(parent_matches),
        )

        # Group canonical parents by rule name for efficient lookup
        canonical_parents_by_rule: Dict[str, List[ResolvedMatch]] = {}
        for parent in parent_matches:
            if parent.reference is None:  # Only canonical parents (no reference)
                if parent.rule not in canonical_parents_by_rule:
                    canonical_parents_by_rule[parent.rule] = []
                canonical_parents_by_rule[parent.rule].append(parent)

        logger.debug(
            "Found canonical parents for rules: %s",
            list(canonical_parents_by_rule.keys()),
        )

        # Group child matches by their parent rule
        child_matches_by_parent_rule: Dict[str, List[ResolvedMatch]] = {}
        for child in child_matches:
            parent_rule = child.rule.split(".")[0]  # Extract parent rule name
            if parent_rule not in child_matches_by_parent_rule:
                child_matches_by_parent_rule[parent_rule] = []
            child_matches_by_parent_rule[parent_rule].append(child)

        resolved_matches = []
        discarded_count = 0

        # Process each child rule group
        for parent_rule, children in child_matches_by_parent_rule.items():
            logger.debug(
                "Processing %s child matches for parent rule '%s'",
                len(children),
                parent_rule,
            )

            # Get canonical parents for this rule
            canonical_parents = canonical_parents_by_rule.get(parent_rule, [])
            if not canonical_parents:
                # Check if this is likely individual rule evaluation (no parent rules in input)
                has_parent_rules = any("." not in match.rule for match in matches)
                if has_parent_rules:
                    # This is full pipeline resolution - warn about missing parents
                    logger.warning(
                        "No canonical parents found for child rule '%s.*' "
                        "- discarding %s child matches",
                        parent_rule,
                        len(children),
                    )
                else:
                    # This is likely individual rule evaluation - log at debug level instead
                    logger.debug(
                        "No canonical parents found for child rule '%s.*' "
                        "(individual rule evaluation) - discarding %s child matches",
                        parent_rule,
                        len(children),
                    )
                discarded_count += len(children)
                continue

            # Resolve each child match
            for child in children:
                resolved_child = self._resolve_single_child(
                    child, canonical_parents, resolver_config, dsl_file_path
                )
                if resolved_child:
                    resolved_matches.append(resolved_child)
                else:
                    discarded_count += 1
                    logger.debug(
                        "Discarded child match '%s' at %s - no matching parent found",
                        child.match,
                        child.offset,
                    )

        # Combine all matches: parents + successfully resolved children
        final_matches = parent_matches + resolved_matches

        # Sort by offset to maintain order (only if needed - many operations preserve order)
        if resolved_matches:  # Only sort if we added child matches
            final_matches.sort(key=lambda m: m.offset)

        logger.info(
            "Child resolution complete: %s children resolved, %s discarded",
            len(resolved_matches),
            discarded_count,
        )
        return final_matches

    def _resolve_single_child(
        self,
        child_match: ResolvedMatch,
        canonical_parents: List[ResolvedMatch],
        resolver_config: Optional[Dict] = None,
        dsl_file_path: Optional[str] = None,
    ) -> Optional[ResolvedMatch]:
        """
        Resolve a single child match to the first matching canonical parent.
        Args:
            child_match: The child match to resolve
            canonical_parents: List of canonical parent matches to check against
            resolver_config: Configuration for resolver methods and optional tokens
            dsl_file_path: Path to DSL file for resolving relative paths

        Returns:
            ResolvedMatch with reference set, or None if no match found
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        # Extract configuration for the child rule
        method = "exact"  # Default matching method
        threshold = 0.85  # Default threshold for fuzzy matching
        flags = TokenizationFlags(ignore_case=True, ignore_punctuation=False)
        optional_tokens = set()

        if resolver_config and child_match.rule in resolver_config:
            rule_config = resolver_config[child_match.rule]
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
                optional_tokens_file = config_dict["optional_tokens"][0]
                optional_tokens = self.tokenizer.load_optional_tokens(
                    optional_tokens_file, dsl_file_path
                )

        # Check cache first for this child-to-parent resolution
        parent_rule = child_match.rule.split(".")[0]
        canonical_parent_texts = [parent.match for parent in canonical_parents]
        cache_key = self._get_child_cache_key(
            child_match.match,
            child_match.rule,
            parent_rule,
            flags,
            optional_tokens,
            method,
            threshold,
            canonical_parent_texts,
        )

        if cache_key in self._child_resolution_cache:
            cached_parent_offset = self._child_resolution_cache[cache_key]
            if cached_parent_offset is not None:
                # Found cached parent reference - create resolved child
                resolved_child = ResolvedMatch(
                    offset=child_match.offset,
                    length=child_match.length,
                    rule=child_match.rule,
                    match=child_match.match,
                    reference=cached_parent_offset,
                    sentence_end=child_match.sentence_end,
                    paragraph_end=child_match.paragraph_end,
                )
                logger.debug(
                    "Child cache hit: '%s' -> references %s",
                    child_match.match,
                    cached_parent_offset,
                )
                return resolved_child
            # Cached result was no match found
            logger.debug("Child cache hit: '%s' -> no parent match", child_match.match)
            return None

        # Tokenize the child match (do NOT pass optional_tokens; filter after)
        child_tokens = self.tokenizer.tokenize(child_match.match, flags)
        if optional_tokens:
            child_tokens = [t for t in child_tokens if t not in optional_tokens]
        child_token_set = set(child_tokens)

        logger.debug(
            "Resolving child '%s' (tokens: %s) using method '%s'",
            child_match.match,
            child_tokens,
            method,
        )

        # Check each canonical parent for a match
        for parent in canonical_parents:
            # Tokenize the parent match with same configuration
            parent_tokens = self.tokenizer.tokenize(parent.match, flags)
            if optional_tokens:
                parent_tokens = [t for t in parent_tokens if t not in optional_tokens]
            parent_token_set = set(parent_tokens)

            # Apply matching algorithm
            is_match = False
            if method == "exact":
                # Early exit: if parent has fewer tokens than child, it can't be a superset
                if len(parent_token_set) < len(child_token_set):
                    logger.debug(
                        "  Skipping parent '%s' - fewer tokens (%s) than child (%s)",
                        parent.match,
                        len(parent_token_set),
                        len(child_token_set),
                    )
                    continue

                # Exact: child tokens must be subset of parent tokens
                is_match = child_token_set.issubset(parent_token_set)
                logger.debug(
                    "  Exact match vs parent '%s' (tokens: %s): %s",
                    parent.match,
                    parent_tokens,
                    is_match,
                )
            elif method == "fuzzy":
                # Fuzzy: edit distance similarity must meet threshold
                child_joined = " ".join(sorted(child_tokens))
                parent_joined = " ".join(sorted(parent_tokens))

                sm = SequenceMatcher(None, child_joined, parent_joined)
                similarity = sm.ratio()
                is_match = similarity >= threshold
                logger.debug(
                    "  Fuzzy match vs parent '%s' (similarity: %.3f >= %s): %s",
                    parent.match,
                    similarity,
                    threshold,
                    is_match,
                )

            if is_match:
                # Found a match - create resolved child with reference
                resolved_child = ResolvedMatch(
                    offset=child_match.offset,
                    length=child_match.length,
                    rule=child_match.rule,
                    match=child_match.match,
                    reference=parent.offset,  # Reference the parent's offset
                    sentence_end=child_match.sentence_end,
                    paragraph_end=child_match.paragraph_end,
                )
                logger.debug(
                    "  Child '%s' resolved to parent '%s' at offset %s",
                    child_match.match,
                    parent.match,
                    parent.offset,
                )

                # Update cache with successful match
                self._child_resolution_cache[cache_key] = parent.offset
                logger.debug(
                    "Cached child resolution: '%s' -> %s",
                    child_match.match,
                    parent.offset,
                )

                return resolved_child

        # No matching parent found
        logger.debug("  No matching parent found for child '%s'", child_match.match)

        # Update cache with no match found
        self._child_resolution_cache[cache_key] = None
        logger.debug("Cached child resolution: '%s' -> no match", child_match.match)

        return None

    def _get_child_cache_key(
        self,
        child_match_text: str,
        child_rule: str,
        parent_rule: str,
        flags: TokenizationFlags,
        optional_tokens: Set[str],
        method: str,
        threshold: float = 0.85,
        canonical_parent_texts: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a cache key for child-to-parent match resolution.

        Args:
            child_match_text: The text content of the child match
            child_rule: Rule name for the child match
            parent_rule: Rule name for the parent match
            flags: Tokenization flags
            optional_tokens: Set of optional tokens
            method: Matching method (exact, fuzzy)
            threshold: Threshold for fuzzy matching
            canonical_parent_texts: List of canonical parent texts for context

        Returns:
            Cache key string for this child resolution configuration
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        optional_key = str(sorted(optional_tokens)) if optional_tokens else "none"
        parent_context = (
            str(sorted(canonical_parent_texts)) if canonical_parent_texts else "none"
        )
        return (
            f"{child_match_text}|{child_rule}|{parent_rule}|{flags.ignore_case}|"
            f"{flags.ignore_punctuation}|{optional_key}|{method}|{threshold}|{parent_context}"
        )
