"""
OMG Evaluator: Core evaluation engine for OMG DSL rules.

This module provides the RuleEvaluator class that processes OMG AST structures
and evaluates them against haystack text using pyomgmatch library integration.
"""

import os
import re
import sys
import tempfile
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from omega_match.omega_match import Compiler, Matcher

from .omg_ast import (
    Alt,
    Concat,
    Escape,
    Dot,
    LineEnd,
    LineStart,
    CharClass,
    CharRange,
    ListMatch,
    ListMatchResult,
    Literal,
    MatchResult,
    NamedCapture,
    Quantified,
    Root,
    RuleDef,
    RuleMatch,
)
from .omg_resolver import EntityResolver

# Pre-compiled byte constants for performance
NEWLINE_BYTES = {b"\n", b"\r"}
WHITESPACE_BYTES = {
    ord(" "),
    ord("\t"),
    ord("\n"),
    ord("\r"),
}  # Optimized whitespace check

# Pre-compiled regex patterns for escape sequences (performance optimization)
ESCAPE_PATTERNS = {
    "d": re.compile(rb"\d"),  # Matches any digit [0-9]
    "D": re.compile(rb"\D"),  # Matches any non-digit
    "s": re.compile(rb"\s"),  # Matches any whitespace
    "S": re.compile(rb"\S"),  # Matches any non-whitespace
    "w": re.compile(rb"\w"),  # Matches any word character [a-zA-Z0-9_]
    "W": re.compile(rb"\W"),  # Matches any non-word character
    ".": re.compile(rb"\."),  # Matches literal dot
    "]": re.compile(rb"\]"),  # Matches literal ]
    "[": re.compile(rb"\["),  # Matches literal [
    "-": re.compile(rb"\-"),  # Matches literal -
    "\\": re.compile(rb"\\\\"),  # Matches literal backslash
}


# Module-level helper functions


@lru_cache(maxsize=128)
def unwrap_single(node: Any) -> Any:
    """
    Simplify AST nodes: collapse single-alt nodes recursively.
    """
    if isinstance(node, Alt) and len(node.options) == 1:
        return unwrap_single(node.options[0])
    return node


def deep_flatten_matches(matches: Sequence[MatchResult]) -> List[MatchResult]:
    """
    Recursively flatten all _constituent_matches.
    """
    flat: List[MatchResult] = []
    for m in matches:
        sub = getattr(m, "_constituent_matches", None) or getattr(
            m, "sub_matches", None
        )
        if sub:
            flat.extend(deep_flatten_matches(sub))
        else:
            flat.append(m)
    return flat


class RuleEvaluator:
    """
    Core evaluation engine for OMG DSL rules.

    Processes AST structures and evaluates them against haystack text using
    pattern matching with pyomgmatch library integration. Manages import
    compilation, match result processing, and rule evaluation workflow.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, ast_root: Root, haystack: bytes):
        self.ast_root = ast_root
        self.rule_map = {rule.name: rule for rule in ast_root.rules}
        self.haystack = haystack
        self.haystack_len = len(haystack)
        self.haystack_id = id(haystack)  # For caching
        self.matches: Dict[str, List[MatchResult]] = {}
        # Performance caches
        self._pattern_cache: Dict[Tuple[int, int], List[MatchResult]] = {}
        self._sorted_matches_cache: Dict[str, List[MatchResult]] = {}
        # Cache for node type lookups (performance optimization)
        self._node_type_cache: Dict[int, type] = {}
        # Cache for prefix length calculations (performance optimization)
        self._prefix_length_cache: Dict[int, Tuple[int, int]] = {}
        # Offset-indexed cache for ListMatch lookups
        self._offset_indexed_cache: Dict[str, Dict[int, List[MatchResult]]] = {}
        # Initialize resolver
        self.resolver = EntityResolver(haystack)
        self._compile_and_match_imports()

    def _compile_and_match_imports(self):
        compiled_path = None
        for imp in self.ast_root.imports:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    compiled_path = tmp.name

                # Resolve import path relative to DSL file location
                import_path = imp.path
                if self.ast_root.dsl_file_path and not os.path.isabs(import_path):
                    # If we have a DSL file path and the import path is relative,
                    # resolve it relative to the DSL file's directory
                    dsl_dir = os.path.dirname(self.ast_root.dsl_file_path)
                    import_path = os.path.join(dsl_dir, import_path)

                # Check if the patterns file exists before trying to compile
                if not os.path.isfile(import_path):
                    raise RuntimeError(f"Import file not found: {import_path}")

                # Use import flags to configure matching options
                flags = set(imp.flags or [])
                ci = "ignore-case" in flags
                ip = "ignore-punctuation" in flags
                ew = "elide-whitespace" in flags
                wb = "word-boundary" in flags
                wp = "word-prefix" in flags
                ws = "word-suffix" in flags
                ls = "line-start" in flags
                le = "line-end" in flags

                # Compile patterns from file
                Compiler.compile_from_filename(
                    compiled_file=compiled_path,
                    patterns_file=import_path,
                    case_insensitive=ci,
                    ignore_punctuation=ip,
                    elide_whitespace=ew,
                )

                # Run matcher on haystack
                with Matcher(
                    compiled_path,
                    case_insensitive=ci,
                    ignore_punctuation=ip,
                    elide_whitespace=ew,
                ) as matcher:
                    external_results = matcher.match(
                        self.haystack,
                        no_overlap=True,   # No overlapping matches for OMG
                        longest_only=True, # Longest match only for OMG
                        word_boundary=wb,
                        word_prefix=wp,
                        word_suffix=ws,
                        line_start=ls,
                        line_end=le,
                    )
                    # Convert external MatchResult objects to our internal ones
                    internal_results = [
                        MatchResult.from_external_match(r) for r in external_results
                    ]
                    self.matches[imp.alias] = internal_results
                    # Pre-sort matches for faster lookups
                    self._sorted_matches_cache[imp.alias] = sorted(
                        internal_results, key=lambda m: m.offset
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to process import {imp.path}: {e}") from e
            finally:  # Clean up temp file
                if compiled_path and os.path.exists(compiled_path):
                    try:
                        os.unlink(compiled_path)
                    except OSError:
                        pass  # Ignore cleanup errors

        # Build offset-indexed cache after imports
        self._build_offset_indexed_cache()

    def _build_offset_indexed_cache(self):
        """Build offset-indexed cache for fast ListMatch lookups by offset."""
        for alias, matches in self.matches.items():
            offset_map = {}
            for match in matches:
                offset = match.offset
                if offset not in offset_map:
                    offset_map[offset] = []
                offset_map[offset].append(match)
            self._offset_indexed_cache[alias] = offset_map

    # pylint: disable=too-many-locals,too-many-branches
    def _match_single_quantified_list_match(
        self, quant_node: Quantified, required_start_offset: int
    ) -> List[ListMatchResult]:
        """
        Matches a single Quantified(ListMatch) pattern part, e.g., [[word]]+.
        Optimized version with better performance for large haystacks.
        """
        # Phase 2 Optimization: Use type-based check instead of isinstance
        if type(quant_node.expr) is not ListMatch:
            return []

        list_match_node = quant_node.expr
        alias = list_match_node.name

        # Use pre-sorted cache if available
        if alias in self._sorted_matches_cache:
            alias_matches = self._sorted_matches_cache[alias]
        else:
            alias_matches = sorted(self.matches.get(alias, []), key=lambda m: m.offset)
            self._sorted_matches_cache[alias] = alias_matches

        # Binary search for first match at or after required_start_offset
        left, right = 0, len(alias_matches)
        start_idx = -1

        while left < right:
            mid = (left + right) // 2
            if alias_matches[mid].offset < required_start_offset:
                left = mid + 1
            else:
                right = mid

        if (
            left < len(alias_matches)
            and alias_matches[left].offset == required_start_offset
        ):
            start_idx = left
        else:
            return []

        # Greedily extend from this starting point
        temp_chain_end_offset = required_start_offset
        temp_sub_matches: List[ListMatchResult] = []
        max_matches = quant_node.quant.max or float("inf")

        for m_idx in range(
            start_idx, min(start_idx + int(max_matches), len(alias_matches))
        ):
            match_candidate = alias_matches[m_idx]

            if match_candidate.offset == temp_chain_end_offset:
                # Strictly adjacent or first match
                lm_res = ListMatchResult.from_match_result(
                    match_candidate, list_match_node.name
                )
                temp_sub_matches.append(lm_res)
                temp_chain_end_offset = match_candidate.offset + match_candidate.length
            elif match_candidate.offset > temp_chain_end_offset:
                # Gap, this greedy chain ends
                break

        if len(temp_sub_matches) < quant_node.quant.min:
            return []

        # Construct the final ListMatchResult for the best sequence
        combined_offset = temp_sub_matches[0].offset
        combined_length = sum(item.length for item in temp_sub_matches)
        combined_end = combined_offset + combined_length

        # Optimized line boundary checks
        line_start_ok = (
            not quant_node.expect_line_start
            or combined_offset == 0
            or self.haystack[combined_offset - 1 : combined_offset] in NEWLINE_BYTES
        )
        line_end_ok = (
            not quant_node.expect_line_end
            or combined_end == self.haystack_len
            or self.haystack[combined_end : combined_end + 1] in NEWLINE_BYTES
        )

        if line_start_ok and line_end_ok:
            combined_match_bytes = self.haystack[combined_offset:combined_end]
            final_lm_result = ListMatchResult(
                offset=combined_offset,
                match=combined_match_bytes,
                alias=list_match_node.name,
                sub_matches=tuple(temp_sub_matches),
            )
            return [final_lm_result]

        return []

    # pylint: disable=too-many-locals
    def _match_general_quantified(
        self, quant_node: Quantified, initial_offset: int
    ) -> List[MatchResult]:
        final_results: List[MatchResult] = []

        min_q = quant_node.quant.min
        max_q = quant_node.quant.max
        haystack = self.haystack
        haystack_len = self.haystack_len
        expect_line_start = quant_node.expect_line_start
        expect_line_end = quant_node.expect_line_end

        # Fast path for exact quantifier {n}
        if min_q == max_q and min_q > 0:
            all_paths: List[List[Tuple[int, List[MatchResult]]]] = [
                [(initial_offset, [])]
            ]
            for i in range(min_q):
                new_paths = []
                for offset, path in all_paths[-1]:
                    matches = self._match_pattern_part(quant_node.expr, offset)
                    for match in matches:
                        if match.offset == offset and match.length > 0:
                            new_paths.append(
                                (match.offset + match.length, path + [match])
                            )
                if not new_paths:
                    break
                all_paths.append(new_paths)
            else:
                # Only emit if all n matches succeeded
                for offset, path in all_paths[-1]:
                    flat_matches = deep_flatten_matches(tuple(path))
                    if flat_matches:
                        combined_offset = flat_matches[0].offset
                        combined_end = flat_matches[-1].offset + flat_matches[-1].length
                        line_start_ok = (
                            not expect_line_start
                            or combined_offset == 0
                            or haystack[combined_offset - 1 : combined_offset]
                            in NEWLINE_BYTES
                        )
                        line_end_ok = (
                            not expect_line_end
                            or combined_end == haystack_len
                            or haystack[combined_end : combined_end + 1]
                            in NEWLINE_BYTES
                        )
                        if line_start_ok and line_end_ok:
                            seq_match_bytes = haystack[combined_offset:combined_end]
                            res = MatchResult(
                                offset=combined_offset, match=seq_match_bytes
                            )
                            res._ast_node = quant_node  # pylint: disable=protected-access
                            res._constituent_matches = flat_matches  # pylint: disable=protected-access
                            final_results.append(res)
            # If min==0, also emit zero-length match (handled below)
            return final_results

        # Always emit a zero-length match if min == 0 (even if no submatches found)
        if min_q == 0:
            zero_length_match = MatchResult(offset=initial_offset, match=b"")
            zero_length_match._ast_node = quant_node  # pylint: disable=protected-access
            zero_length_match._constituent_matches = []  # pylint: disable=protected-access
            final_results.append(zero_length_match)

        q_paths: List[Tuple[int, int, Tuple[MatchResult, ...]]] = [
            (0, initial_offset, ())
        ]
        processed_sequences = set()
        head = 0

        while head < len(q_paths):
            num_matched_expr, current_expr_end_offset, current_expr_tuple = q_paths[
                head
            ]
            head += 1

            if current_expr_tuple:
                seq_key = tuple((m.offset, m.length) for m in current_expr_tuple)
                if seq_key in processed_sequences:
                    continue
                processed_sequences.add(seq_key)

            if num_matched_expr >= min_q:
                if current_expr_tuple:
                    flat_matches = deep_flatten_matches(current_expr_tuple)
                    if flat_matches:
                        combined_offset = flat_matches[0].offset
                        combined_end = flat_matches[-1].offset + flat_matches[-1].length
                        line_start_ok = (
                            not expect_line_start
                            or combined_offset == 0
                            or haystack[combined_offset - 1 : combined_offset]
                            in NEWLINE_BYTES
                        )
                        line_end_ok = (
                            not expect_line_end
                            or combined_end == haystack_len
                            or haystack[combined_end : combined_end + 1]
                            in NEWLINE_BYTES
                        )
                        if line_start_ok and line_end_ok:
                            seq_match_bytes = haystack[combined_offset:combined_end]
                            res = MatchResult(
                                offset=combined_offset, match=seq_match_bytes
                            )
                            res._ast_node = quant_node  # pylint: disable=protected-access
                            res._constituent_matches = flat_matches  # pylint: disable=protected-access
                            final_results.append(res)

            if max_q is None or num_matched_expr < max_q:
                matches_for_one_more_expr = self._match_pattern_part(
                    quant_node.expr, current_expr_end_offset
                )
                if not matches_for_one_more_expr:
                    continue
                for m_expr in matches_for_one_more_expr:
                    if m_expr.length == 0:
                        continue
                    if min_q == 0:
                        if m_expr.offset > current_expr_end_offset:
                            gap_length = m_expr.offset - current_expr_end_offset
                            if gap_length > 10:
                                continue
                            gap_bytes = haystack[
                                current_expr_end_offset : m_expr.offset
                            ]
                            if not all(ch in WHITESPACE_BYTES for ch in gap_bytes):
                                continue
                    else:
                        if m_expr.offset != current_expr_end_offset:
                            continue
                    new_expr_tuple = current_expr_tuple + (m_expr,)
                    q_paths.append(
                        (
                            num_matched_expr + 1,
                            m_expr.offset + m_expr.length,
                            new_expr_tuple,
                        )
                    )

        return final_results

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,protected-access
    def _match_pattern_part(self, part_ast_node: Any, offset: int) -> List[MatchResult]:
        # Fast path: use object ID directly as primary cache key to avoid hashing overhead
        node_to_match = unwrap_single(part_ast_node)
        node_type = type(node_to_match)
        node_id = id(node_to_match)
        cache_key = (node_id, offset)
        cached_result = self._pattern_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        results: List[MatchResult] = []

        # Fast path for common cases using type comparison
        if node_type is Concat:
            concat_results = self._match_concat(node_to_match, offset)
            for cres in concat_results:
                cres._ast_node = part_ast_node
                cres._constituent_matches = getattr(cres, "_constituent_matches", [])
                results.append(cres)
            self._pattern_cache[cache_key] = results
            return results

        if node_type is Alt:
            alt_results = self._match_alt(node_to_match, offset)
            for ares in alt_results:
                ares._ast_node = part_ast_node
                ares._constituent_matches = getattr(ares, "_constituent_matches", [])
                results.append(ares)
            self._pattern_cache[cache_key] = results
            return results

        # Handle bytes literals - fastest path
        if node_type is bytes:
            if self.haystack.startswith(node_to_match, offset):
                res = MatchResult(offset=offset, match=node_to_match)
                res._ast_node = part_ast_node
                res._constituent_matches = []
                results.append(res)

        elif node_type is Literal:
            literal_bytes = node_to_match.value.encode("utf-8")
            if self.haystack.startswith(literal_bytes, offset):
                res = MatchResult(offset=offset, match=literal_bytes)
                res._ast_node = part_ast_node
                res._constituent_matches = []
                results.append(res)

        elif node_type is Escape:
            escape_results = self._match_escape(node_to_match, offset)
            if escape_results:
                for res in escape_results:
                    res._ast_node = part_ast_node
                    res._constituent_matches = getattr(res, "_constituent_matches", [])
                    results.append(res)

        elif node_type is LineStart:
            if offset == 0 or self.haystack[offset - 1 : offset] in NEWLINE_BYTES:
                res = MatchResult(offset=offset, match=b"")
                res._ast_node = part_ast_node
                res._constituent_matches = []
                results.append(res)

        elif node_type is LineEnd:
            if (
                offset == self.haystack_len
                or self.haystack[offset : offset + 1] in NEWLINE_BYTES
            ):
                res = MatchResult(offset=offset, match=b"")
                res._ast_node = part_ast_node
                res._constituent_matches = []
                results.append(res)

        elif node_type is NamedCapture:
            inner_expr_matches = self._match_pattern_part(node_to_match.expr, offset)
            if inner_expr_matches:
                for m_inner in inner_expr_matches:
                    m_inner._ast_node = node_to_match
                    results.append(m_inner)

        elif node_type is Dot:
            # Dot matches any single non-newline byte
            if offset < self.haystack_len:
                nxt = self.haystack[offset : offset + 1]
                if nxt not in NEWLINE_BYTES:
                    res = MatchResult(offset=offset, match=nxt)
                    res._ast_node = part_ast_node
                    res._constituent_matches = []
                    results.append(res)

        elif node_type is CharClass:
            cc_results = self._match_charclass(node_to_match, offset)
            if cc_results:
                for res in cc_results:
                    res._ast_node = part_ast_node
                    res._constituent_matches = []
                    results.append(res)

        elif node_type is ListMatch:
            alias = node_to_match.name
            alias_offset_map = self._offset_indexed_cache.get(alias)
            if alias_offset_map:
                target_matches = alias_offset_map.get(offset, [])
                if target_matches:
                    for m in target_matches:
                        lm_res = ListMatchResult.from_match_result(m, alias)
                        lm_res._ast_node = part_ast_node
                        lm_res._constituent_matches = []
                        results.append(lm_res)

        elif node_type is Quantified:
            # Quantified(ListMatch) needs special handling
            # Phase 2 Optimization: Use type-based check instead of isinstance
            if isinstance(node_to_match.expr, ListMatch):
                lm_results = self._match_single_quantified_list_match(
                    node_to_match, offset
                )
                if lm_results:
                    for lm_res in lm_results:
                        lm_res._ast_node = part_ast_node
                        lm_res._constituent_matches = (
                            list(lm_res.sub_matches)
                            if hasattr(lm_res, "sub_matches")
                            else []
                        )
                        results.append(lm_res)
            else:
                quant_results = self._match_general_quantified(node_to_match, offset)
                if quant_results:
                    for qres in quant_results:
                        qres._ast_node = part_ast_node
                        qres._constituent_matches = getattr(
                            qres, "_constituent_matches", []
                        )
                        results.append(qres)

        self._pattern_cache[cache_key] = results
        return results

    def _match_charclass(self, node: CharClass, offset: int) -> List[MatchResult]:
        """Match a character class node at the given offset. Returns a 1-byte match if any part matches."""
        results: List[MatchResult] = []
        if offset >= self.haystack_len:
            return results
        b = self.haystack[offset : offset + 1]
        b_int = self.haystack[offset]

        def matches_escape(val: str) -> bool:
            # val is like 'd','D','s','S','w','W','.',']','[','-','\\' (without the leading slash)
            # Use existing ESCAPE_PATTERNS where applicable (single-byte match)
            if val in ESCAPE_PATTERNS:
                pat = ESCAPE_PATTERNS[val]
                return bool(pat.match(self.haystack, offset, offset + 1))
            # Unsupported escapes in char class (e.g., \b, \B) - treat as non-match
            return False

        def matches_char(c: str) -> bool:
            return b_int == ord(c)

        def matches_range(r: CharRange) -> bool:
            start = ord(r.start)
            end = ord(r.end)
            if start <= b_int <= end:
                return True
            return False

        # If any part matches, the class matches
        for part in node.parts:
            if isinstance(part, CharRange):
                if matches_range(part):
                    results.append(MatchResult(offset=offset, match=b))
                    break
            elif isinstance(part, Escape):
                val = part.value
                if val.startswith("\\"):
                    val = val[1:]
                if matches_escape(val):
                    results.append(MatchResult(offset=offset, match=b))
                    break
            else:
                # Raw character inside the class
                if isinstance(part, str) and len(part) == 1 and matches_char(part):
                    results.append(MatchResult(offset=offset, match=b))
                    break

        return results

    def _match_escape(self, part_ast_node: Any, offset: int) -> List[MatchResult]:
        r"""
        Match single Escape AST node at given offset, supporting \d,\D,\s,\S,\w,\W,
        and literal escapes. Optimized version using pre-compiled regex patterns
        for maximum performance.
        """
        results: List[MatchResult] = []
        val = part_ast_node.value
        if val.startswith("\\"):  # Strip leading backslash
            val = val[1:]

        if offset < self.haystack_len:
            # Use pre-compiled regex patterns for common escape sequences
            if val in ESCAPE_PATTERNS:
                pattern = ESCAPE_PATTERNS[val]
                match = pattern.match(self.haystack, offset, offset + 1)
                if match:
                    res = MatchResult(offset=offset, match=match.group())
                    res._ast_node = part_ast_node
                    res._constituent_matches = []
                    results.append(res)
            else:
                # Fallback to character-by-character matching for uncommon escapes
                b = self.haystack[offset]  # Get single byte directly
                try:
                    chr(b)  # Just check if it's a valid character
                except (ValueError, OverflowError):
                    pass  # Invalid character, will not match
                match = False  # type: ignore

                # Handle any remaining literal escaped characters not in ESCAPE_PATTERNS
                if len(val) == 1:
                    if b == ord(val):  # Direct byte comparison
                        match = True  # type: ignore

                if match:
                    res = MatchResult(
                        offset=offset, match=self.haystack[offset : offset + 1]
                    )
                    res._ast_node = part_ast_node
                    res._constituent_matches = []
                    results.append(res)
        return results

    # pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks,protected-access
    def _match_concat(
        self, pattern: Concat, initial_concat_offset: int
    ) -> List[MatchResult]:
        if not pattern.parts:
            empty_match = MatchResult(offset=initial_concat_offset, match=b"")
            empty_match._constituent_matches = []
            empty_match._ast_node = pattern
            return [empty_match]

        active_paths: List[Tuple[int, Tuple[MatchResult, ...]]] = [
            (initial_concat_offset, ())
        ]

        for pattern_part_ast in pattern.parts:
            if not active_paths:
                break
            next_active_paths = []
            for current_expected_start_offset, path_chain_so_far in active_paths:
                matches_for_this_part_at_expected_offset = self._match_pattern_part(
                    pattern_part_ast, current_expected_start_offset
                )
                if not matches_for_this_part_at_expected_offset:
                    continue
                for m in matches_for_this_part_at_expected_offset:
                    if m.offset == current_expected_start_offset:
                        new_path_chain = path_chain_so_far + (m,)
                        next_part_expected_start_offset = m.offset + m.length
                        next_active_paths.append(
                            (next_part_expected_start_offset, new_path_chain)
                        )
            active_paths = next_active_paths

        if not active_paths:
            return []
        final_results_for_concat: List[MatchResult] = [
            MatchResult(
                offset=final_path_chain[0].offset,
                match=self.haystack[
                    final_path_chain[0].offset : final_chain_end_offset
                ],
            )
            for final_chain_end_offset, final_path_chain in active_paths
            if len(final_path_chain) == len(pattern.parts)
        ]
        for i, (final_chain_end_offset, final_path_chain) in enumerate(active_paths):
            if len(final_path_chain) == len(pattern.parts):
                final_results_for_concat[i]._ast_node = pattern
                final_results_for_concat[i]._constituent_matches = list(
                    final_path_chain
                )
        return final_results_for_concat

    def _match_alt(self, pattern: Alt, current_offset: int) -> List[MatchResult]:
        all_results = []
        for option_ast_node in pattern.options:
            # An option in Alt could be anything, not just Concat, e.g. Alt( [[L1]] | [[L2]] )
            # So, we use _match_pattern_part for the option.
            # _match_pattern_part will call _match_concat if option_ast_node is/unwraps to Concat.
            option_matches = self._match_pattern_part(option_ast_node, current_offset)

            all_results.extend(option_matches)

        return all_results

    @lru_cache(maxsize=512)
    def _contains_listmatch(self, node: Any) -> bool:
        """Return True if the AST subtree has at least one ListMatch."""
        if isinstance(node, ListMatch):
            return True
        if hasattr(node, "options"):  # Alt
            return any(self._contains_listmatch(o) for o in node.options)
        if hasattr(node, "parts"):  # Concat
            return any(self._contains_listmatch(p) for p in node.parts)
        if hasattr(node, "expr"):  # Quantified or NamedCapture
            return self._contains_listmatch(node.expr)
        return False

    @lru_cache(maxsize=512)
    def _contains_unbounded_quantifier(self, node: Any) -> bool:
        """Return True if the AST subtree has any unbounded quantifier (* or +)."""
        if isinstance(node, Quantified):
            # max=None indicates * or +
            if node.quant.max is None:
                return True
            return self._contains_unbounded_quantifier(node.expr)
        if hasattr(node, "options"):  # Alt
            return any(self._contains_unbounded_quantifier(o) for o in node.options)
        if hasattr(node, "parts"):  # Concat
            return any(self._contains_unbounded_quantifier(p) for p in node.parts)
        if hasattr(node, "pattern"):  # NamedCapture
            return self._contains_unbounded_quantifier(node.pattern)
        return False

    # pylint: disable=too-many-locals,too-many-branches
    def evaluate_rule(
        self,
        rule_def: RuleDef,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[RuleMatch]:
        """
        Evaluate a rule definition against the haystack and return all matches.

        Args:
            rule_def: The rule definition to evaluate
            progress_callback: Optional callback for progress reporting

        Returns:
            List of RuleMatch objects representing successful matches
        """
        # Enforce rule contains at least one ListMatch anchor
        if not self._contains_listmatch(rule_def.pattern):
            raise ValueError(
                "Every rule must contain at least one ListMatch to anchor matching."
            )
        # Disallow any unbounded quantifier (* or +)
        if self._contains_unbounded_quantifier(rule_def.pattern):
            raise ValueError(
                "Unbounded quantifiers (* or +) are not supported; please use bounded {m,n}."
            )

        # Optimize: Try to extract ListMatch patterns and their anchor points
        # Instead of scanning every offset, focus on locations where ListMatch anchors occur
        anchor_offsets = self._find_listmatch_anchors(rule_def.pattern)

        # Fallback to full scan if anchor optimization fails or returns too many candidates
        if (
            not anchor_offsets or len(anchor_offsets) > self.haystack_len * 0.1
        ):  # More than 10% of positions
            anchor_offsets = list(range(self.haystack_len + 1))

        total_offsets = len(anchor_offsets)
        rule_matches: List[RuleMatch] = []

        # Batch progress callbacks to reduce overhead
        progress_interval = max(
            1, total_offsets // 100
        )  # Update progress at most 100 times

        for idx, i in enumerate(anchor_offsets):
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx, total_offsets)

            current_match_options_for_rule = self._match_pattern_part(
                rule_def.pattern, i
            )
            for overall_match_for_option in current_match_options_for_rule:
                if overall_match_for_option.offset == i:
                    captures: Dict[str, List[MatchResult]] = {}
                    path_chain = getattr(
                        overall_match_for_option, "_constituent_matches", None
                    )
                    ast_node_that_matched = getattr(
                        overall_match_for_option, "_ast_node", None
                    )

                    if ast_node_that_matched:
                        self._collect_named_captures(
                            ast_node_that_matched,
                            overall_match_for_option,
                            captures,
                            path_chain,
                        )
                    else:
                        self._collect_named_captures(
                            rule_def.pattern,
                            overall_match_for_option,
                            captures,
                            path_chain,
                        )
                    rule_matches.append(
                        RuleMatch(
                            name=rule_def.name,
                            offset=overall_match_for_option.offset,
                            match=overall_match_for_option.match,
                            named_captures=captures,
                        )
                    )

        # Final progress callback
        if progress_callback:
            progress_callback(total_offsets, total_offsets)

        # Optimized disambiguation: longest, leftmost, non-overlapping matches only
        if not rule_matches:
            return []

        # Use a more efficient sorting approach
        rule_matches.sort(key=lambda m: (m.offset, -m.length))

        # Optimized selection using single pass
        selected = []
        last_end = 0
        i = 0

        while i < len(rule_matches):
            match = rule_matches[i]
            if match.offset >= last_end:
                # Find longest match at this offset by looking ahead
                current_offset = match.offset
                longest = match
                j = i + 1

                # Look for longer matches at the same offset
                while (
                    j < len(rule_matches) and rule_matches[j].offset == current_offset
                ):
                    if rule_matches[j].length > longest.length:
                        longest = rule_matches[j]
                    j += 1

                selected.append(longest)
                last_end = longest.offset + longest.length
                i = j  # Skip all matches at this offset
            else:
                i += 1

        # Apply resolver enrichment if configured
        enriched_matches = self._apply_resolver(selected, rule_def)

        return enriched_matches

    def get_import_match_counts(self) -> Dict[str, int]:
        """
        Get the number of compiled matches for each import alias.

        Returns:
            Dictionary mapping import aliases to their match counts
        """
        return {alias: len(matches) for alias, matches in self.matches.items()}

    def _find_listmatch_anchors(self, pattern) -> List[int]:
        """Find potential anchor points where ListMatch patterns could start matching.
        Uses adaptive sampling strategies based on pattern prefix characteristics."""
        # Extract all ListMatch nodes from the pattern along with their position info
        listmatch_info: List[Tuple[ListMatch, int, int]] = []
        self._extract_listmatches_with_context(pattern, listmatch_info)

        if not listmatch_info:
            return []

        # For each ListMatch, find potential pattern starting positions
        all_anchor_positions = set()

        for listmatch, prefix_min_len, prefix_max_len in listmatch_info:
            if (
                hasattr(listmatch, "name")
                and listmatch.name in self._sorted_matches_cache
            ):
                # Use pre-computed matches for this ListMatch
                listmatch_positions = [
                    match.offset for match in self._sorted_matches_cache[listmatch.name]
                ]

                # OPTIMIZATION: Use fast path for simple patterns (ListMatch at start)
                if prefix_min_len == 0 and prefix_max_len == 0:
                    # ListMatch is at pattern start - use simple approach
                    all_anchor_positions.update(listmatch_positions)
                    continue

                # Use adaptive sampling logic for all complex patterns
                for match_entry in self._sorted_matches_cache[listmatch.name]:
                    listmatch_pos = match_entry.offset

                    # GENERAL OPTIMIZATION: Adaptive sampling based on range characteristics
                    start_range_begin = max(0, listmatch_pos - prefix_max_len)
                    start_range_end = max(0, listmatch_pos - prefix_min_len + 1)
                    range_size = start_range_end - start_range_begin

                    # Calculate range uncertainty (how variable the prefix length is)
                    range_uncertainty = prefix_max_len - prefix_min_len

                    # Use adaptive strategy based on range size and uncertainty
                    if range_size <= 8:
                        # Small ranges - enumerate all positions for accuracy
                        for pos in range(start_range_begin, start_range_end):
                            all_anchor_positions.add(pos)
                    elif range_size <= 20:
                        # Medium ranges - be more conservative with sampling
                        # Only use step size 2 if uncertainty is very high
                        step_size = 2 if range_uncertainty > 12 else 1
                        for pos in range(start_range_begin, start_range_end, step_size):
                            all_anchor_positions.add(pos)
                    elif range_size <= 50:
                        # Large ranges - use conservative sampling
                        # Higher uncertainty means more sampling needed
                        if range_uncertainty > 25:
                            # Very high uncertainty - use denser sampling
                            step_size = max(2, range_size // 20)
                        else:
                            # Moderate uncertainty - balanced sampling
                            step_size = max(2, range_size // 15)
                        for pos in range(start_range_begin, start_range_end, step_size):
                            all_anchor_positions.add(pos)
                    else:
                        # Very large ranges - aggressive sampling with fallback
                        # Cap the maximum number of positions we'll try
                        max_positions = min(25, range_size // 4)
                        if max_positions < 8:
                            max_positions = 8  # Always try at least several positions

                        step_size = max(1, range_size // max_positions)
                        for pos in range(start_range_begin, start_range_end, step_size):
                            all_anchor_positions.add(pos)

        return sorted(all_anchor_positions)

    def _extract_listmatches(self, node, result):
        """Recursively extract all ListMatch nodes from AST."""
        if isinstance(node, ListMatch):
            result.append(node)
        elif hasattr(node, "options"):  # Alt
            for option in node.options:
                self._extract_listmatches(option, result)
        elif hasattr(node, "parts"):  # Concat
            for part in node.parts:
                self._extract_listmatches(part, result)
        elif hasattr(node, "expr"):  # Quantified or NamedCapture
            self._extract_listmatches(node.expr, result)
        elif hasattr(node, "pattern"):  # NamedCapture
            self._extract_listmatches(node.pattern, result)

    def _extract_listmatches_with_context(self, node, result, path=None):
        """Recursively extract ListMatch nodes with their position context."""
        if path is None:
            path = []

        if isinstance(node, ListMatch):
            # Calculate prefix length from the current path
            min_len, max_len = self._calculate_prefix_length(path)
            result.append((node, min_len, max_len))
        elif hasattr(node, "options"):  # Alt
            for option in node.options:
                self._extract_listmatches_with_context(option, result, path)
        elif hasattr(node, "parts"):  # Concat
            for i, part in enumerate(node.parts):
                # For concat, add all previous parts to the path when processing this part
                prefix_path = path + list(node.parts[:i])
                self._extract_listmatches_with_context(part, result, prefix_path)
        elif hasattr(node, "expr"):  # Quantified or NamedCapture
            self._extract_listmatches_with_context(node.expr, result, path)
        elif hasattr(node, "pattern"):  # NamedCapture (alternative attribute name)
            self._extract_listmatches_with_context(node.pattern, result, path)

    def _calculate_prefix_length(self, prefix_parts):
        """Calculate min/max length of prefix parts before a ListMatch."""
        # Use caching to avoid recomputing the same prefix patterns
        cache_key = id(prefix_parts)
        if cache_key in self._prefix_length_cache:
            return self._prefix_length_cache[cache_key]

        min_length = 0
        max_length = 0

        for part in prefix_parts:
            if isinstance(part, Literal):
                length = len(part.value.encode("utf-8"))
                min_length += length
                max_length += length
            elif isinstance(part, Escape):
                # Single character escapes like \d, \s, etc.
                min_length += 1
                max_length += 1
            elif isinstance(part, Quantified):
                inner_min, inner_max = self._calculate_single_part_length(part.expr)
                min_length += inner_min * part.quant.min
                max_length += inner_max * (
                    part.quant.max if part.quant.max is not None else 10
                )
            else:
                # Conservative estimates for other node types
                part_min, part_max = self._calculate_single_part_length(part)
                min_length += part_min
                max_length += part_max

        # Cache the result
        result = (min_length, max_length)
        self._prefix_length_cache[cache_key] = result
        return result

    def _calculate_single_part_length(self, part):
        """Calculate min/max length for a single pattern part."""
        if isinstance(part, Literal):
            length = len(part.value.encode("utf-8"))
            return length, length
        elif isinstance(part, Escape):
            return 1, 1
        elif isinstance(part, (Dot, CharClass)):
            return 1, 1
        elif isinstance(part, ListMatch):
            # Assume reasonable bounds for list matches
            return 1, 20
        else:
            # Conservative default
            return 0, 5

    # pylint: disable=too-many-locals,too-many-branches
    def _collect_named_captures(
        self, pattern_node, overall_match, captures_dict, path_chain=None
    ):
        # overall_match._ast_node should ideally be pattern_node or the specific
        # sub-node that matched. path_chain is used when overall_match is for a
        # composite structure (like Concat, Quantified) and path_chain represents
        # the ordered sequence of matches for its constituents.

        effective_ast_node = unwrap_single(pattern_node)

        # Use path_chain if provided directly, otherwise try to get from
        # overall_match._constituent_matches
        effective_path_chain = (
            path_chain
            if path_chain is not None
            else getattr(overall_match, "_constituent_matches", None)
        )

        # Fast path for NamedCapture - most common case
        if isinstance(effective_ast_node, NamedCapture):
            # The overall_match is for the NamedCapture node itself.
            captures_dict.setdefault(effective_ast_node.name, []).append(overall_match)

            # Recurse on the inner expression.
            constituents_of_named_capture_match = getattr(
                overall_match, "_constituent_matches", None
            )

            if (
                constituents_of_named_capture_match
                and len(constituents_of_named_capture_match) == 1
            ):
                match_for_inner_expr = constituents_of_named_capture_match[0]
                self._collect_named_captures(
                    unwrap_single(effective_ast_node.expr),
                    match_for_inner_expr,
                    captures_dict,
                    getattr(match_for_inner_expr, "_constituent_matches", None),
                )
            return

        if isinstance(effective_ast_node, Quantified):
            # For Quantified, we iterate through its constituent matches
            if effective_path_chain:
                expr_node = unwrap_single(effective_ast_node.expr)
                for item_match in effective_path_chain:
                    self._collect_named_captures(
                        expr_node,
                        item_match,
                        captures_dict,
                        getattr(item_match, "_constituent_matches", None),
                    )
            return

        if isinstance(effective_ast_node, Concat):
            # For Concat, we iterate through its constituent matches
            if effective_path_chain and len(effective_path_chain) == len(
                effective_ast_node.parts
            ):
                for i, part_match in enumerate(effective_path_chain):
                    original_part_ast = unwrap_single(effective_ast_node.parts[i])
                    self._collect_named_captures(
                        original_part_ast,
                        part_match,
                        captures_dict,
                        getattr(part_match, "_constituent_matches", None),
                    )
            return

        if isinstance(effective_ast_node, Alt):
            # For Alt, overall_match._constituent_matches should contain chosen option
            if effective_path_chain and len(effective_path_chain) == 1:
                chosen_option_match = effective_path_chain[0]
                ast_for_chosen_option = getattr(chosen_option_match, "_ast_node", None)

                if ast_for_chosen_option:
                    self._collect_named_captures(
                        unwrap_single(ast_for_chosen_option),
                        chosen_option_match,
                        captures_dict,
                        getattr(chosen_option_match, "_constituent_matches", None),
                    )
            return

        if isinstance(effective_ast_node, ListMatch):
            # ListMatch nodes are implicitly named captures if they have an alias
            if effective_ast_node.name:
                captures_dict.setdefault(effective_ast_node.name, []).append(
                    overall_match
                )
            return

        # Fallback for other node types
        if effective_path_chain:
            for constituent_match in effective_path_chain:
                constituent_ast_node = getattr(constituent_match, "_ast_node", None)
                if constituent_ast_node:
                    self._collect_named_captures(
                        constituent_ast_node,
                        constituent_match,
                        captures_dict,
                        getattr(constituent_match, "_constituent_matches", None),
                    )

    def _convert_resolver_config(self, resolver_config) -> Optional[Dict]:
        """Convert AST resolver config to resolver dictionary format."""
        if not resolver_config:
            return None

        config = {
            "method": resolver_config.method,
            "flags": list(resolver_config.flags),
            "args": list(resolver_config.args),
            "optional_tokens": list(resolver_config.optional_tokens),
        }
        return config

    def _get_effective_resolver_config(self, rule_def: RuleDef) -> Optional[Dict]:
        """Get the effective resolver configuration for a rule, considering defaults."""
        # Per-rule resolver takes precedence
        if rule_def.resolver_config:
            return self._convert_resolver_config(rule_def.resolver_config)

        # Fall back to default resolver for dotted rules
        if "." in rule_def.name and self.ast_root.default_resolver:
            return self._convert_resolver_config(self.ast_root.default_resolver)

        # For non-dotted (parent) rules without explicit resolver config:
        # Only provide resolver if this parent rule has child rules
        if "." not in rule_def.name:
            if self._has_child_rules(rule_def.name):
                # Parent has children, provide minimal boundary-only configuration for location enrichment
                return {
                    "method": "boundary-only",
                    "flags": [],
                    "args": [],
                    "optional_tokens": [],
                }
            else:
                # Parent has no children, no resolver needed
                return None

        # Default case (shouldn't reach here for normal rules)
        return None

    def _has_child_rules(self, parent_rule_name: str) -> bool:
        """Check if a parent rule has any corresponding child rules."""
        if "." in parent_rule_name:
            # This is already a child rule, not a parent
            return False

        # Check if any rule in the AST starts with parent_rule_name followed by a dot
        for rule in self.ast_root.rules:
            if rule.name.startswith(parent_rule_name + "."):
                return True
        return False

    def _apply_resolver(
        self, rule_matches: List[RuleMatch], rule_def: RuleDef
    ) -> List[RuleMatch]:
        """Apply entity resolver to rule matches if configured."""
        resolver_config = self._get_effective_resolver_config(rule_def)

        if not resolver_config:
            return rule_matches

        try:
            # Convert RuleMatch objects to resolver input format
            resolver_input = []
            for match in rule_matches:
                resolver_input.append(
                    {
                        "offset": match.offset,
                        "length": match.length,
                        "rule": match.name,
                        "match": match.match.decode("utf-8", errors="replace"),
                    }
                )

            # Apply resolver
            resolved_matches = self.resolver.resolve_matches(
                resolver_input,
                resolver_config,
                self.ast_root.dsl_file_path,
                default_resolver=self.ast_root.default_resolver,
            )

            # Convert back to RuleMatch objects with enrichment
            enriched_rule_matches = []
            for i, match in enumerate(rule_matches):
                if i < len(resolved_matches):
                    resolved = resolved_matches[i]
                    enriched_match = RuleMatch(
                        name=match.name,
                        offset=match.offset,
                        match=match.match,
                        named_captures=match.named_captures,
                        reference=resolved.reference,
                        sentence_end=resolved.sentence_end,
                        paragraph_end=resolved.paragraph_end,
                    )
                    enriched_rule_matches.append(enriched_match)
                else:
                    # Fallback if resolver didn't return enough matches
                    enriched_rule_matches.append(match)

            return enriched_rule_matches

        except ValueError as e:
            # Re-raise validation errors (like missing default resolver) as fatal errors
            if "default resolver" in str(e):
                raise e
            # Other ValueError issues get graceful fallback
            sys.stderr.write(
                f"Warning: Resolver validation failed for rule '{rule_def.name}': {e}\n"
            )
            return rule_matches
        except (ImportError, AttributeError, TypeError, KeyError) as e:
            # Graceful fallback on specific resolver errors
            sys.stderr.write(
                f"Warning: Resolver failed for rule '{rule_def.name}': {e}\n"
            )
            return rule_matches
