"""
Raw match processing and validation for the OMG resolver.

Handles loading and validation of raw match dictionaries.
Implements Step 1 of the Plan of Attack: verify raw match structure.
"""

import logging
from typing import Dict, List

from .core import ResolvedMatch

logger = logging.getLogger(__name__)


class RawMatchProcessor:
    """
    Handles loading and validation of raw match dictionaries.

    Implements Step 1 of the Plan of Attack: verify raw match structure.
    """

    REQUIRED_FIELDS = {"offset", "length", "rule", "match"}

    @staticmethod
    def validate_raw_match(match_dict: Dict) -> bool:
        """
        Validate that a raw match dictionary has all required fields.

        Args:
            match_dict: Dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(match_dict, dict):
            return False

        # Check required fields
        missing_fields = RawMatchProcessor.REQUIRED_FIELDS - set(match_dict.keys())
        if missing_fields:
            logger.warning("Match missing required fields: %s", missing_fields)
            return False

        # Validate field types
        try:
            offset = match_dict["offset"]
            length = match_dict["length"]
            rule = match_dict["rule"]
            match_text = match_dict["match"]

            if not isinstance(offset, int) or offset < 0:
                logger.warning("Invalid offset: %s", offset)
                return False

            if not isinstance(length, int) or length <= 0:
                logger.warning("Invalid length: %s", length)
                return False

            if not isinstance(rule, str) or not rule:
                logger.warning("Invalid rule: %s", rule)
                return False

            if not isinstance(match_text, str):
                logger.warning("Invalid match text type: %s", type(match_text))
                return False

        except (KeyError, TypeError) as e:
            logger.warning("Error validating match: %s", e)
            return False

        return True

    @staticmethod
    def load_and_validate_matches(matches: List[Dict]) -> List[Dict]:
        """
        Load and validate a list of raw match dictionaries.

        Args:
            matches: List of raw match dictionaries

        Returns:
            List of validated matches (invalid ones are filtered out)
        """
        validated_matches = []

        for i, match_dict in enumerate(matches):
            if RawMatchProcessor.validate_raw_match(match_dict):
                validated_matches.append(match_dict)
            else:
                logger.warning("Skipping invalid match at index %s: %s", i, match_dict)

        logger.info(
            "Validated %s out of %s raw matches", len(validated_matches), len(matches)
        )
        return validated_matches

    @staticmethod
    def convert_to_resolved_matches(matches: List[Dict]) -> List[ResolvedMatch]:
        """
        Convert validated raw match dictionaries to ResolvedMatch objects.

        Args:
            matches: List of validated raw match dictionaries

        Returns:
            List of ResolvedMatch objects
        """
        resolved_matches = []

        for match_dict in matches:
            match_val = match_dict["match"]
            if isinstance(match_val, bytes):
                match_val = match_val.decode("utf-8", errors="replace")
            resolved_match = ResolvedMatch(
                offset=match_dict["offset"],
                length=match_dict["length"],
                rule=match_dict["rule"],
                match=match_val,
            )
            resolved_matches.append(resolved_match)

        return resolved_matches
