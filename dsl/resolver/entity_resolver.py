"""
Main entity resolver implementation for the OMG DSL.

This module provides the main EntityResolver class that orchestrates the complete
entity resolution pipeline, following the algorithm specified in RESOLUTION.md.

Implements Steps 1-6 of the Plan of Attack:
1. Raw match loading and validation
2. Tokenization module with optional-tokens support
3. Overlap resolution
4. Horizontal Canonicalization (parent rule resolution)
5. Vertical Child Resolution (child-to-parent matching)
6. Metadata enrichment (sentence and paragraph boundaries)
"""

import logging
from typing import Callable, Dict, List, Optional, Set

from ..omg_ast import ResolverConfig
from .core import ResolvedMatch, TokenizationFlags, extract_resolver_config
from .horizontal import HorizontalCanonicalizer
from .metadata import MetadataEnricher
from .overlap_resolver import OverlapResolver
from .raw_processor import RawMatchProcessor
from .tokenizer import Tokenizer
from .vertical import VerticalChildResolver

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class EntityResolver:
    """
    Entity resolver implementation following the RESOLUTION.md algorithm.

    This version implements Steps 1-6 of the Plan of Attack:
    1. Raw match loading and validation
    2. Tokenization module with optional-tokens support
    3. Overlap resolution
    4. Horizontal Canonicalization (parent rule resolution)
    5. Vertical Child Resolution (child-to-parent matching)
    6. Metadata enrichment (sentence and paragraph boundaries)
    """

    def __init__(self, haystack: bytes):
        """
        Initialize the resolver with the haystack text.

        Args:
            haystack: The input text as bytes
        """
        self.haystack = haystack
        self.haystack_str = haystack.decode("utf-8", errors="replace")
        self.haystack_len = len(haystack)

        # Initialize components
        self.tokenizer = Tokenizer()
        self.match_processor = RawMatchProcessor()
        self.overlap_resolver = OverlapResolver()
        self.horizontal_canonicalizer = HorizontalCanonicalizer(self.tokenizer)
        self.vertical_child_resolver = VerticalChildResolver(self.tokenizer)
        self.metadata_enricher = MetadataEnricher(self.haystack_str)

    def _resolve_default_resolver(
        self, resolver_config: Optional[Dict], default_resolver
    ) -> Optional[Dict]:
        """Resolve 'default resolver' references to actual method from default_resolver."""
        if not resolver_config or not default_resolver:
            return resolver_config

        resolved_config = {}
        for rule_name, config in resolver_config.items():
            # Handle both dict format and ResolverConfig namedtuple
            if hasattr(config, "method"):
                # ResolverConfig namedtuple
                if config.method == "default resolver":
                    # Replace with actual default resolver configuration
                    resolved_config[rule_name] = ResolverConfig(
                        method=default_resolver.method,  # Get the actual method (e.g., 'exact')
                        flags=default_resolver.flags,
                        args=default_resolver.args,
                        optional_tokens=config.optional_tokens,  # Keep rule-specific tokens
                    )
                    logger.debug(
                        "Resolved '%s': 'default resolver' -> '%s'",
                        rule_name,
                        default_resolver.method,
                    )
                else:
                    # Keep as-is
                    resolved_config[rule_name] = config
            elif (
                isinstance(config, dict) and config.get("method") == "default resolver"
            ):
                # Dict format (legacy)
                resolved_config[rule_name] = ResolverConfig(
                    method=default_resolver.method,
                    flags=tuple(default_resolver.flags),
                    args=tuple(default_resolver.args),
                    optional_tokens=tuple(config.get("optional_tokens", [])),
                )
                logger.debug(
                    "Resolved '%s': 'default resolver' -> '%s'",
                    rule_name,
                    default_resolver.method,
                )
            elif isinstance(config, dict):
                # Convert any other dicts to ResolverConfig
                resolved_config[rule_name] = ResolverConfig(
                    method=config.get("method", "exact"),
                    flags=tuple(config.get("flags", [])),
                    args=tuple(config.get("args", [])),
                    optional_tokens=tuple(config.get("optional_tokens", [])),
                )
            else:
                # Keep as-is
                resolved_config[rule_name] = config

        return resolved_config

    def resolve_matches(
        self,
        matches: List[Dict],
        resolver_config: Optional[Dict] = None,
        dsl_file_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        default_resolver=None,
    ) -> List[ResolvedMatch]:
        """
        Resolve a list of raw matches using the specified resolver configuration.

        This version implements Steps 1-6 of the algorithm:
        1. Load and validate raw matches
        2. Resolve overlapping matches (before tokenization to save work)
        3. Tokenization (demonstrated with test cases)
        4. Horizontal canonicalization (parent rule resolution)
        5. Vertical child resolution (child-to-parent matching)
        6. Metadata enrichment (sentence and paragraph boundaries)

        Args:
            matches: List of raw match dictionaries
            resolver_config: Configuration for the resolver method and options
            dsl_file_path: Path to the DSL file for resolving relative paths
            progress_callback: Optional callback function(stage_name, current, total)
            default_resolver: Default resolver configuration to use for 'default resolver' method

        Returns:
            List of ResolvedMatch objects with overlaps resolved, canonical references assigned,
            child matches resolved to parents or discarded, and metadata enriched
        """  # Convert resolver config from AST format if needed
        if resolver_config:
            converted_config = {}
            for rule_name, config in resolver_config.items():
                converted_config[rule_name] = extract_resolver_config(config)
            resolver_config = converted_config
        # Resolve 'default resolver' references if we have a default_resolver
        resolver_config = self._resolve_default_resolver(
            resolver_config, default_resolver
        )

        if progress_callback:
            progress_callback("converting", 0, len(matches))

        # Step 1: Load and validate raw matches
        logger.info("Processing %s raw matches", len(matches))
        validated_matches = self.match_processor.load_and_validate_matches(matches)

        if progress_callback:
            progress_callback("converting", len(validated_matches), len(matches))

        # Convert to ResolvedMatch objects
        resolved_matches = self.match_processor.convert_to_resolved_matches(
            validated_matches
        )

        if progress_callback:
            progress_callback("resolving_overlaps", 0, len(resolved_matches))

        # Step 3: Resolve overlapping matches (done before tokenization to save work)
        logger.info("Resolving overlaps for %s matches", len(resolved_matches))
        non_overlapping_matches = self.overlap_resolver.resolve_overlaps(
            resolved_matches
        )

        if progress_callback:
            progress_callback(
                "resolving_overlaps",
                len(non_overlapping_matches),
                len(resolved_matches),
            )

        # Step 2: Demonstrate tokenization (for now, just log examples)
        # Note: We do this after overlap resolution to avoid wasting tokenization work
        self._demonstrate_tokenization(
            non_overlapping_matches[:5], resolver_config, dsl_file_path
        )

        if progress_callback:
            progress_callback("canonicalizing_parents", 0, len(non_overlapping_matches))

        # Step 4: Horizontal canonicalization (parent rule resolution)
        logger.info("Canonicalizing %s matches", len(non_overlapping_matches))
        canonicalized_matches = self.horizontal_canonicalizer.canonicalize_matches(
            non_overlapping_matches, resolver_config, dsl_file_path
        )

        if progress_callback:
            progress_callback(
                "canonicalizing_parents",
                len(canonicalized_matches),
                len(non_overlapping_matches),
            )
            progress_callback("resolving_children", 0, len(canonicalized_matches))

        # Step 5: Vertical child resolution (child-to-parent matching)
        logger.info(
            "Resolving child matches for %s matches", len(canonicalized_matches)
        )
        final_matches = self.vertical_child_resolver.resolve_child_matches(
            canonicalized_matches, resolver_config, dsl_file_path
        )
        if progress_callback:
            progress_callback(
                "resolving_children", len(final_matches), len(canonicalized_matches)
            )
            progress_callback("enriching_metadata", 0, len(final_matches))

        # Step 6: Metadata enrichment (sentence and paragraph boundaries)
        logger.info("Enriching metadata for %s matches", len(final_matches))
        enriched_matches = self.metadata_enricher.enrich_matches(final_matches)

        if progress_callback:
            progress_callback(
                "enriching_metadata", len(enriched_matches), len(final_matches)
            )
            progress_callback("complete", len(enriched_matches), len(enriched_matches))

        logger.info(
            "Completed processing: %s matches with full algorithm (Steps 1-6)",
            len(enriched_matches),
        )
        return enriched_matches

    def _demonstrate_tokenization(
        self,
        sample_matches: List[ResolvedMatch],
        resolver_config: Optional[Dict],
        dsl_file_path: Optional[str],
    ):
        """
        Demonstrate tokenization on a sample of matches.

        This shows Step 2 functionality until we implement full resolution.
        """
        if not sample_matches:
            return

        logger.info("Demonstrating tokenization on sample matches:")

        # Create some example tokenization flags
        default_flags = TokenizationFlags(ignore_case=True, ignore_punctuation=False)

        for match in sample_matches:
            # Tokenize without optional tokens
            tokens = self.tokenizer.tokenize(match.match, default_flags)
            token_bag_key = self.tokenizer.get_token_bag_key(tokens)

            logger.info(
                "Match: '%s' -> Tokens: %s -> Key: '%s'",
                match.match,
                tokens,
                token_bag_key,
            )

            # If we have resolver config, try to load optional tokens
            if resolver_config and match.rule in resolver_config:
                rule_config = resolver_config[match.rule]

                # Convert ResolverConfig object to dict if needed
                if hasattr(rule_config, "method"):  # It's a ResolverConfig object
                    rule_config_dict = {
                        "method": rule_config.method,
                        "flags": list(rule_config.flags),
                        "args": list(rule_config.args),
                        "optional_tokens": list(rule_config.optional_tokens),
                    }
                else:
                    rule_config_dict = rule_config

                # Check if rule_config has optional_tokens configuration
                if (
                    "optional_tokens" in rule_config_dict
                    and rule_config_dict["optional_tokens"]
                ):
                    # Take the first optional tokens file (there could be multiple)
                    optional_tokens_file = rule_config_dict["optional_tokens"][0]
                    optional_tokens = self.tokenizer.load_optional_tokens(
                        optional_tokens_file, dsl_file_path
                    )

                    # Tokenize and filter out optional tokens
                    tokens_with_opt = self.tokenizer.tokenize(
                        match.match, default_flags
                    )
                    filtered_tokens = [
                        t for t in tokens_with_opt if t not in optional_tokens
                    ]
                    filtered_key = self.tokenizer.get_token_bag_key(filtered_tokens)
                    logger.info(
                        "  Filtered: %s -> Key: '%s'", filtered_tokens, filtered_key
                    )

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
            threshold: Threshold for fuzzy matching        Returns:
            Cache key string for this match processing configuration
        """
        optional_key = str(sorted(optional_tokens)) if optional_tokens else "none"
        return (
            f"{match_text}|{rule}|{flags.ignore_case}|{flags.ignore_punctuation}|"
            f"{optional_key}|{method}|{threshold}"
        )

        # Remove unreachable and incorrect assignments to self.resolver_config at the end of the file
