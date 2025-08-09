#!/usr/bin/env python3

import argparse
import io
import json
import logging
import sys
import time
from typing import Dict, List, Optional, TypedDict, Any, Tuple, cast

from omega_match.omega_match import get_version
from dsl.omg_evaluator import RuleEvaluator
from dsl.omg_parser import parse_string, OMG_DSL_VERSION
from dsl.omg_resolver import EntityResolver

__version__ = "0.2.0"

# Ensure UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class _StageInfo(TypedDict):
    stage: str
    current: int
    total: int
    timestamp: float


def show_resolution_statistics(
    progress_data: Dict[str, Any],
    input_matches: int,
    output_matches: int,
    rule_match_counts: Optional[Dict[str, int]] = None,
):
    """Display detailed resolution statistics."""
    total_time = time.time() - progress_data["start_time"]
    stages: List[_StageInfo] = cast(List[_StageInfo], progress_data.get("stages", []))

    sys.stderr.write("=== Resolution Statistics ===\n")
    sys.stderr.write(f"Total processing time: {total_time:.2f} seconds\n")
    sys.stderr.write(f"Input matches: {input_matches}\n")
    sys.stderr.write(f"Output matches: {output_matches}\n")
    sys.stderr.write(
        f"Reduction: {input_matches - output_matches} matches ({((input_matches - output_matches) / input_matches * 100) if input_matches else 0:.1f}%)\n"
    )

    if rule_match_counts is not None:
        sys.stderr.write("\nMatch breakdown by rule (input):\n")
        for rule, count in rule_match_counts.items():
            sys.stderr.write(f"  {rule}: {count}\n")

    if stages:
        sys.stderr.write("\nStage breakdown:\n")
        stage_times: Dict[str, List[float]] = {}
        for i, stage_info in enumerate(stages):
            stage = stage_info["stage"]
            timestamp = stage_info["timestamp"]

            # Calculate time spent in this stage
            prev_time = stages[i - 1]["timestamp"] if i > 0 else 0
            stage_time = timestamp - prev_time

            if stage not in stage_times:
                stage_times[stage] = []
            stage_times[stage].append(stage_time)

        for stage, times in stage_times.items():
            total_stage_time = sum(times)
            avg_time = total_stage_time / len(times) if times else 0
            sys.stderr.write(
                f"  {stage}: {total_stage_time:.2f}s (avg: {avg_time:.2f}s per update)\n"
            )

    sys.stderr.write("=============================\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run OMG DSL pattern matching on a haystack file."
    )
    parser.add_argument("dsl_file", nargs="?", help="Path to DSL source file")
    parser.add_argument(
        "haystack_file", nargs="?", help="Path to input text (haystack) file"
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Emit all results in a single pretty-printed JSON array",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set logging level",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all progress updates",
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="Skip entity resolution and emit raw matches",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show detailed resolution statistics",
    )
    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="Show detailed timing information",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write output to this file (UTF-8, LF line endings). If omitted, output goes to stdout.",
    )

    args = parser.parse_args()

    if args.version:
        omega_match_version = get_version()
        print("Version information:")
        print(f"  omega_match: {omega_match_version}")
        print(f"  omg: {__version__}")
        print(f"  DSL: {OMG_DSL_VERSION}")
        sys.exit(0)

    if not args.dsl_file or not args.haystack_file:
        parser.error("the following arguments are required: dsl_file, haystack_file")

    # Start overall timing
    overall_start_time = time.time()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("omg")

    # Time file loading
    file_load_start = time.time()
    with open(args.dsl_file, "r", encoding="utf-8") as f:
        dsl_code = f.read()

    with open(args.haystack_file, "rb") as f:
        haystack = f.read()
    file_load_time = time.time() - file_load_start

    if args.show_timing:
        sys.stderr.write(f"File loading time: {file_load_time:.3f}s\n")

    # Time parsing
    parse_start = time.time()
    logger.info("Parsing DSL and evaluating haystack")
    ast_root = parse_string(dsl_code, dsl_file_path=args.dsl_file)
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)
    parse_time = time.time() - parse_start

    if args.show_timing:
        sys.stderr.write(f"DSL parsing time: {parse_time:.3f}s\n")

    rules = ast_root.rules

    all_matches: List[Any] = []

    # Time rule evaluation
    eval_start_time = time.time()

    # First, evaluate all rules to get raw matches
    rule_timings: List[Tuple[str, float, int]] = []
    for rule in rules:
        rule_start_time = time.time()
        if args.quiet:
            results = evaluator.evaluate_rule(rule)
        else:

            def _status_callback(idx: int, total: int, rn: str = rule.name) -> None:
                pct = (idx / total * 100) if total else 0
                sys.stderr.write(f"\rEvaluating: {rn}: {idx}/{total} ({pct:.1f}%)")
                sys.stderr.flush()

            results: List[Any] = evaluator.evaluate_rule(
                rule, progress_callback=_status_callback
            )
            sys.stderr.write("\n")
        rule_end_time = time.time()
        rule_timings.append((rule.name, rule_end_time - rule_start_time, len(results)))
        all_matches.extend(results)

    eval_time = time.time() - eval_start_time

    if args.show_timing:
        sys.stderr.write("\n=== Rule Evaluation Timings ===\n")
        sys.stderr.write(f"Rule evaluation time: {eval_time:.3f} seconds\n")
        for rule_name, rule_time, rule_count in rule_timings:
            sys.stderr.write(f"  Rule '{rule_name}': {rule_time:.3f}s ({rule_count} matches)\n")
        sys.stderr.write("=============================\n\n")

    # Convert RuleMatch objects to dictionary format expected by resolver
    match_dicts: List[Dict[str, Any]] = []
    rule_match_counts: Dict[str, int] = {}
    for rule_match in all_matches:
        match_dict = {
            "offset": rule_match.offset,
            "length": rule_match.length,
            "rule": rule_match.name,
            "match": rule_match.match.decode("utf-8", errors="replace"),
        }
        match_dicts.append(match_dict)
        rule_match_counts[rule_match.name] = rule_match_counts.get(rule_match.name, 0) + 1

    if args.no_resolve:
        # Skip resolution and use raw matches
        output: List[Dict[str, Any]] = []
        for match_dict in match_dicts:
            output.append(match_dict)

        sys.stderr.write(f"Found {len(output)} raw matches across {len(rules)} rules\n")
        # Create dummy progress_data for stats if needed
        if args.show_stats:
            dummy_progress_data: Dict[str, Any] = {"stages": [], "start_time": overall_start_time}
            show_resolution_statistics(
                dummy_progress_data, len(match_dicts), len(output), rule_match_counts
            )
    else:
        # Extract resolver configuration from AST
        resolver_config: Dict[str, Any] = {}
        for rule in ast_root.rules:
            if hasattr(rule, "resolver_config") and rule.resolver_config:
                resolver_config[rule.name] = rule.resolver_config

        # Use EntityResolver with enhanced features
        resolver = EntityResolver(haystack)
        logger.info("Using EntityResolver (enhanced algorithm)")

        # Track resolution progress and timing
        resolve_start_time = time.time()
        start_time = time.time()
        progress_data: Dict[str, Any] = {"stages": [], "start_time": start_time}

        def progress_callback(stage: str, current: int, total: int) -> None:
            """Progress callback for EntityResolver."""
            if not args.quiet:
                pct = (current / total * 100) if total else 0
                sys.stderr.write(
                    f"\rResolution: {stage}: {current}/{total} ({pct:.1f}%)"
                )
                sys.stderr.flush()

            # Track progress data for statistics
            progress_data["stages"].append(
                {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "timestamp": time.time() - start_time,
                }
            )

        # Apply enhanced resolver with progress tracking to ALL matches at once
        resolved_matches = resolver.resolve_matches(
            match_dicts,
            resolver_config,
            args.dsl_file,
            progress_callback=progress_callback if not args.quiet else None,
            default_resolver=ast_root.default_resolver,
        )

        resolve_time = time.time() - resolve_start_time
        if args.show_timing:
            sys.stderr.write(f"Resolution time: {resolve_time:.3f}s\n")

        if not args.quiet:
            sys.stderr.write("\n")

        # Show detailed statistics if requested
        output: List[Dict[str, Any]] = []
        if args.show_stats:
            show_resolution_statistics(
                progress_data, len(match_dicts), len(resolved_matches), rule_match_counts
            )

        # Process resolved matches for output
        for r in resolved_matches:
            # Filter out unresolved dot rule matches
            # Dot rules (those with '.' in name) should either resolve to parent entities or be dropped
            is_dot_rule = "." in r.rule
            has_reference = hasattr(r, "reference") and r.reference is not None

            if is_dot_rule and not has_reference:
                # Skip unresolved dot rule matches
                continue

            match_dict = {
                "offset": r.offset,
                "length": r.length,
                "rule": r.rule,
                "match": r.match,  # Already decoded string from resolver
            }

            # Include enriched fields if they exist (from resolver)
            if has_reference:
                match_dict["reference"] = r.reference
            if hasattr(r, "sentence_end") and r.sentence_end is not None:
                match_dict["sentence_end"] = r.sentence_end
            if hasattr(r, "paragraph_end") and r.paragraph_end is not None:
                match_dict["paragraph_end"] = r.paragraph_end

            output.append(match_dict)

        sys.stderr.write(
            f"Found {len(output)} resolved matches across {len(rules)} rules\n"
        )

    # Print overall timing
    overall_time = time.time() - overall_start_time
    if args.show_timing:
        sys.stderr.write(f"Overall processing time: {overall_time:.3f} seconds\n")

    # Output results
    output_stream = None
    if args.output:
        output_stream = open(args.output, "w", encoding="utf-8", newline="\n")
    else:
        output_stream = sys.stdout

    try:
        if args.pretty_print:
            json.dump(output, output_stream, indent=2)
            output_stream.write("\n")
        else:
            for item in output:
                output_stream.write(json.dumps(item))
                output_stream.write("\n")
    finally:
        if args.output and output_stream is not sys.stdout:
            output_stream.close()


if __name__ == "__main__":
    main()
