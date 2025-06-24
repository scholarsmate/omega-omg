## Entity Resolution Algorithm

This document defines the entity resolution algorithm for canonicalizing parent matches and resolving child matches in DSL pattern matching, then enriching each match with sentence and paragraph boundary metadata.

---

### 1. Vocabulary

**Parent Rules** (`rule` names without a dot):

* Top‑level patterns (e.g., `person`).
* Canonicalized among themselves; duplicates receive a `reference` to the canonical match.
* Canonicalization uses the rule’s configured resolver (e.g., `exact`, `fuzzy`).

**Canonical Matches**:

* Parent matches that do *not* reference any other match.
* Anchor points for child rule resolution.
* Identified uniquely by their start offset.

**Child Rules** (`rule` names with a dot):

* Must resolve to exactly one canonical parent match.
* If no valid parent, the child match is discarded.
* Only one level of nesting is supported (e.g., `person.surname`, not `person.x.y`).

**Optional Tokens**:

* A per‑rule list of tokens (e.g., honorifics, single letters, suffixes) ignored during matching.
* Configured via `optional-tokens("file.txt")` in the DSL.

---

### 2. Resolution Process Overview

The resolver uses each rule’s **configured algorithm** (`exact` or `fuzzy(threshold)`) and flags (`ignore-case`, `ignore-punctuation`) to compare **bags of tokens**. Optional tokens are loaded from the rule’s `optional-tokens(...)` file and stripped before matching.

1. **Sort All Matches**

   * Order every match by its byte `offset` (ascending).

2. **Resolve Overlapping Matches**

   * Scan sorted matches for any byte-range overlap.
   * When two overlap, drop the lower-priority match according to:

     1. **Match length** (longer wins)
     2. **Offset** (earlier wins)
     3. **Rule name length** (shorter wins)
     4. **Alphabetical rule name**

3. **Canonicalize Parent Matches** (Horizontal Resolution)

   * For each **parent rule** (`rule` without a dot):

     1. **Load configuration:** read `method`, `flags`, `threshold`, `optional-tokens`.
     2. **Preprocess tokens:**

        * Apply `ignore-case` / `ignore-punctuation`.
        * Tokenize on `\w+`.
        * Remove optional tokens from the set.
     3. **Group matches** by their **sorted token bags** (space‑joined string key).
     4. **Within each group**, sort by `length` (desc), then `offset` (asc).
     5. **Select the first** as canonical (no `.reference`); assign others `.reference = canonical.offset`.

4. **Resolve Child Matches** (Vertical Resolution)

   * For each **child rule** (`rule` with a dot):

     1. **Load child resolver config** (same parameters).
     2. **Tokenize child match** with the same preprocessing rules.
     3. **Iterate canonical parents** of the matching parent rule:

        * Tokenize each parent as above.
        * **Exact**: require `child_tokens ⊆ parent_tokens`.
        * **Fuzzy**: require similarity ≥ `threshold` (edit‑distance on joined tokens).
     4. **Assign** `.reference = parent.offset` **on the first matching parent**; otherwise **discard** the child match.

5. **Attach Match Metadata**

   * After all `.reference` assignments, compute and add:

     * `sentence_end`: index of the next sentence boundary after `offset`.
     * `paragraph_end`: index of the next paragraph boundary after `offset`.

---

### 3. Tokenization & Matching

* **Tokens**: contiguous word characters (`\w+`).
* **Transformations**: apply rule flags (e.g., `ignore-case`, `ignore-punctuation`) before tokenizing.
* **Resolvers** per rule:

  * `exact`: token bags must be *identical*.
  * `fuzzy(threshold)`: allow approximate token bag matching via edit distance.

---

### 4. Match Metadata

Each final match includes:

* `offset`, `length`, `rule`, `match` (original text)
* `reference` (offset of canonical parent, if any)
* `sentence_end`: byte offset of sentence boundary
* `paragraph_end`: byte offset of paragraph boundary

---

### 5. Computational Complexity

To understand performance characteristics, consider n total matches and p parent matches per rule:

Sorting (Step 1):

1. O(n log n) to order matches by offset.

Overlap Resolution (Step 2):

1. O(n), a single pass scanning overlaps in the sorted list.

Parent Canonicalization (Step 3):

1. Tokenization: O(n · T) where T is average tokens per match (tokenizing each match once).

2. Grouping: O(n) to insert into a hash map keyed by token-bag string.

3. Group Sorting: ∑ₖ Gₖ log Gₖ across all groups (Gₖ group sizes), worst-case O(n log n).

Child Resolution (Step 4):

1. For c child matches and p canonical parents: O(c · ( T + p · C))

2. Tokenization per child: O(T)

3. Checking subset or fuzzy similarity against up to p parents; subset test and similarity each O(T).

Enrichment (Step 5):

1. O(n) to attach sentence/paragraph boundaries via precomputed lists.

Overall, worst-case time is roughly O(n log n + n T + c p T).  With caching of token bags and content references, the effective constants drop significantly for large, repetitive datasets.

---

### 6. Implementation Complexity

Developing this resolver involves several layers of complexity:

1. **Modular Design** (Medium)

   * Separate components for parsing, matching, tokenization, and resolution.
   * Requires designing extensible interfaces for new resolvers.

2. **Tokenization & Normalization** (Low)

   * Implementing robust regex-based token extraction.
   * Handling flags (`ignore-case`, `ignore-punctuation`) correctly.

3. **Overlap Resolution Logic** (Low)

   * Single-pass algorithm with well-defined tie-breakers.
   * Simple comparisons and list operations.

4. **Horizontal Canonicalization** (Medium)

   * Efficient grouping by token bags (hash maps).
   * Correctly sorting and selecting canonical matches.

5. **Vertical Child Resolution** (Medium)

   * Parent-child matching loops with subset or fuzzy checks.
   * Managing the edge-cases when no match is found.

6. **Caching Mechanisms** (Medium)

   * Implementing and managing caches for token bags and match outcomes.
   * Ensuring thread-safety if parallelized.

7. **Metadata Enrichment** (Low)

   * Precomputing sentence & paragraph boundaries.
   * Efficient lookups via binary search.

8. **Error Handling & Logging** (Medium)

   * Capturing and reporting configuration errors, missing files.
   * Verbose logging for debugging complex resolution paths.

Overall, the implementation is of **medium complexity**, balancing straightforward text-processing tasks with the need for efficient data structures, clear modular separation, and thorough error handling.

---

### 7. Recommendations & Best Practices Recommendations & Best Practices

To enhance reliability, maintainability, and performance, consider the following:

1. Early Discard of Empty Token Bags: Immediately drop any match whose token set becomes empty after stripping optional tokens to avoid unnecessary processing.

2. Parallel Canonicalization: For large match sets, canonicalize parent rules in parallel (per rule) to improve throughput, then merge results.

3. Deterministic Tie-breakers: Clearly document and implement all overlap tie-break criteria (rule name length, alphabetical order) to ensure reproducible outputs.

4. Logging & Metrics: Emit optional debug logs for key steps (token grouping sizes, number of overlaps dropped, child resolution failures) to aid troubleshooting.

5. Status Callbacks: Provide a progress_callback(stage: str, current: int, total: int) hook in resolve_matches that is called at each major step (conversion, boundary addition, parent canonicalization per rule, child resolution per rule, and completion), allowing CLI or UI to display real-time progress indicators.

6. Configurable Ordering: Allow DSL users to customize overlap-resolution priority (e.g., prefer earliest over longest) via resolver flags.

7. Extensible Resolvers: Define a plugin interface so new resolver strategies (e.g., metaphone, regex-backtracking) can be added without core changes.

8. Optional Resolvers: Parent rules without child rules (e.g., phone numbers, locations, national IDs) automatically skip resolution to improve performance. Only parent rules with corresponding child rules receive boundary-only resolvers for canonicalization.

9. Token Bag Caching: Cache preprocessed token bags for each match to avoid redundant tokenization during both horizontal and vertical resolution steps, improving performance on large datasets.

10. Match Content Caching: Cache the resolution outcome (canonical reference or none) for each unique match string so that repeated matches reuse the cached reference rather than re-evaluating token bags.

### 8. Plan of Attack

1. Start with Raw Matches: Load a small sample of raw match_dicts and verify offsets, lengths, and rules.

2. Tokenization Module: Implement and unit-test tokenize(text, optional_tokens, flags) to produce correct token bags for various cases, including optional-token stripping.

3. Overlap Resolution: Code the one-pass overlap remover; write tests covering tie-breaker scenarios (length, offset, rule name).

4. Horizontal Canonicalization: Integrate _resolve_token_matches(); test grouping and canonical reference assignment on a controlled dataset (e.g., ‘John’, ‘President John’).

5. Vertical Child Resolution: Implement child-to-parent matching using subset and fuzzy logic; validate that child rules resolve or discard correctly.

6. Metadata Enrichment: Add sentence and paragraph boundary detection; test boundaries on known text snippets.

7. End-to-End Smoke Tests: Run full resolution on a small document (e.g., metadata with known entities) to verify combined behavior.

8. Performance Optimizations: Introduce token-bag and match-content caches; measure before/after performance on larger datasets.

9. Logging & Config: Add debug logs and CLI flags; test disabling resolution via --no-resolution.

10. Documentation & Examples: Update README and DSL docs with algorithm summary, CLI usage, and resolution examples.

---

### 8. Example DSL Snippet

```omg
version 1.0

import "name_prefix.txt" as prefix
import "name_suffix.txt" as suffix
import "names.txt" as given_name
import "surnames.txt" as surname
import with
import word-boundary

resolver default uses fuzzy(threshold=0.90) with ignore-case

person = ( [[prefix]] \s{1,4} )?
         [[given_name]] (\s{1,4} [[given_name]])?
         (\s{1,4} \w\.? )?
         \s{1,4} [[surname]]
         (\s{0,4} \, \s{1,4} [[suffix]])?
         uses default resolver with optional-tokens("person-opt_tokens.txt")

person.surname = [[surname]]
                 (\s{0,4} \, \s{1,4} [[suffix]])?
                 uses default resolver with optional-tokens("person-opt_tokens.txt")
```

### 9. API Signature:

```python
@dataclass
class ResolvedMatch:
    offset: int
    length: int
    rule: str
    match: str
    reference: Optional[int] = None
    sentence_end: Optional[int] = None
    paragraph_end: Optional[int] = None

    
class EntityResolver:
   def __init__(self, haystack: bytes):

   def resolve_matches(
      self,
      matches: List[Dict],
      resolver_config: Optional[Dict] = None,
      dsl_file_path: Optional[str] = None,
      progress_callback: Optional[Callable[[str, int, int], None]] = None
   ) -> List[ResolvedMatch]:
      """
      Args:
         matches: Raw match dicts with keys 'offset', 'length', 'rule', 'match'.
         resolver_config: Per-rule resolver settings.
         dsl_file_path: Path to DSL file, for resolving optional-tokens paths.
         progress_callback: Called as (stage, current, total). Stages include:
               - "converting"
               - "adding_boundaries"
               - "resolving_parents:<rule>"
               - "resolving_children:<rule>"
               - "complete"
      Returns:
         List of `ResolvedMatch` objects, with `.reference`, `.sentence_end`, `.paragraph_end` set.
      """
```