"""
Overlap resolution for the OMG resolver.

Handles overlap resolution for matches using priority rules defined in RESOLUTION.md.
Implements Step 3 of the Plan of Attack: resolve overlapping matches.
"""

import logging
from typing import List

from .core import ResolvedMatch

logger = logging.getLogger(__name__)


class OverlapResolver:
    """
    Handles overlap resolution for matches.

    Implements Step 3 of the Plan of Attack: resolve overlapping matches
    using the priority rules defined in RESOLUTION.md.
    """

    @staticmethod
    def resolve_overlaps(matches: List[ResolvedMatch]) -> List[ResolvedMatch]:
        """
        Resolve overlapping matches using the priority rules.

        Priority rules (highest to lowest):
        1. Match length (longer wins)
        2. Offset (earlier wins)
        3. Rule name length (shorter wins)
        4. Alphabetical rule name

        Args:
            matches: List of ResolvedMatch objects

        Returns:
            List of non-overlapping ResolvedMatch objects
        """
        if not matches:
            return []

        # Step 1: Sort matches by offset for efficient overlap detection
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.offset, -m.length, len(m.rule), m.rule),
        )

        # Step 2: One-pass scan for overlaps
        result = []
        last_end = -1
        for match in sorted_matches:
            match_start = match.offset
            match_end = match.offset + match.length
            if match_start >= last_end:
                result.append(match)
                last_end = match_end
            # else: skip overlapping match (lower priority)

        logger.info("Overlap resolution: %s -> %s matches", len(matches), len(result))
        return result

    @staticmethod
    def _resolve_tie(match1: ResolvedMatch, match2: ResolvedMatch) -> ResolvedMatch:
        """
        Apply tie-breaker rules to determine which match wins.

        Args:
            match1: First match
            match2: Second match

        Returns:
            The winning match
        """
        # Rule 1: Match length (longer wins)
        if match1.length != match2.length:
            return match1 if match1.length > match2.length else match2

        # Rule 2: Offset (earlier wins)
        if match1.offset != match2.offset:
            return match1 if match1.offset < match2.offset else match2

        # Rule 3: Rule name length (shorter wins)
        if len(match1.rule) != len(match2.rule):
            return match1 if len(match1.rule) < len(match2.rule) else match2

        # Rule 4: Alphabetical rule name (lexicographically smaller wins)
        return match1 if match1.rule < match2.rule else match2
