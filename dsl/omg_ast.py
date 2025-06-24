from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

# === Top-Level Nodes ===


@dataclass(frozen=True)
class Version:
    """Represents the DSL version."""

    value: str


@dataclass(frozen=True)
class Import:
    """Represents an import statement in the DSL."""

    path: str
    alias: str
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuleDef:
    """Represents a rule definition in the DSL."""

    name: str
    pattern: "Expr"  # refers to any expression node
    resolver_config: Optional["ResolverConfig"] = None


# === Top-Level Root Structure ===


@dataclass(frozen=True)
class Root:
    """Represents the root of the DSL abstract syntax tree."""

    version: Version
    imports: tuple[Import, ...]
    rules: tuple[RuleDef, ...]
    default_resolver: Optional["ResolverDefault"] = None
    dsl_file_path: Optional[str] = None


# === Expression Base Class ===


class Expr:
    """Base class for all expression nodes in the AST."""

    pass


# === Resolver AST Nodes ===
@dataclass(frozen=True)
class ResolverDefault:
    """Represents the default resolver configuration for the DSL."""

    method: str
    args: tuple[str, ...] = field(default_factory=tuple)
    flags: tuple[str, ...] = field(default_factory=tuple)
    optional_tokens: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolverConfig:
    """Represents the resolver configuration for a specific rule."""

    method: str
    args: tuple[str, ...] = field(default_factory=tuple)
    flags: tuple[str, ...] = field(default_factory=tuple)
    optional_tokens: tuple[str, ...] = field(default_factory=tuple)


# === Concrete Expressions ===


@dataclass(frozen=True)
class Concat(Expr):
    """Represents a concatenation of expressions."""

    parts: tuple[Expr, ...]


@dataclass(frozen=True)
class Alt(Expr):
    """Represents a choice between alternative expressions."""

    options: tuple[Expr, ...]


@dataclass(frozen=True)
class Quantifier:
    """Represents a quantifier for an expression (e.g., {1,4}, ?)."""

    min: int
    max: Optional[int]


@dataclass(frozen=True)
class Quantified(Expr):
    """Represents a quantified expression."""

    expr: Expr
    quant: Quantifier
    expect_line_start: bool = False  # Added
    expect_line_end: bool = False  # Added


# === Atomic Expressions ===


@dataclass(frozen=True)
class Escape(Expr):
    """Represents an escape sequence (e.g., \\s, \\d)."""

    value: str  # like \\s, \\d, etc.


@dataclass(frozen=True)
class Dot(Expr):
    """Represents the dot (.) wildcard, matching any character."""

    pass


@dataclass(frozen=True)
class Literal(Expr):
    """Represents a literal string in an expression."""

    value: str  # e.g. "abc"


@dataclass(frozen=True)
class Identifier(Expr):
    """Represents an identifier, typically a reference to another rule."""

    name: str


@dataclass(frozen=True)
class NamedCapture(Expr):
    """Represents a named capture group in an expression."""

    name: str
    expr: Expr


@dataclass(frozen=True)
class ListMatch(Expr):
    """Represents a list match expression (e.g., [[my_list]])."""

    name: str
    filter: Optional["FilterExpr"]


@dataclass(frozen=True)
class FilterExpr:
    """Represents a filter expression used with list matches."""

    func: str
    arg: str


@dataclass(frozen=True)
class CharClass(Expr):
    """Represents a character class (e.g., [a-z0-9])."""

    parts: tuple[
        Union["CharRange", Escape, str], ...
    ]  # Mix of ranges, escapes, and raw chars


@dataclass(frozen=True)
class CharRange:
    """Represents a range of characters within a character class (e.g., a-z)."""

    start: str
    end: str


@dataclass(frozen=True)
class LineStart:
    """Represents the start of a line anchor (^)."""

    pass


@dataclass(frozen=True)
class LineEnd:
    """Represents the end of a line anchor ($)."""

    pass


# === Match Result Structures ===


@dataclass(unsafe_hash=True)
class MatchResult:
    """Represents the result of a successful match operation."""

    offset: int
    match: bytes  # The actual matched bytes from the haystack
    # Dynamic attributes used by the evaluator - use Any to avoid circular imports
    _ast_node: Optional[Any] = field(default=None, compare=False, hash=False)
    _constituent_matches: List[Any] = field(
        default_factory=list, compare=False, hash=False
    )

    @property
    def length(self) -> int:
        """
        The length of the match, derived from the match bytes.
        """
        return len(self.match)

    @classmethod
    def from_external_match(cls, external_match) -> "MatchResult":
        """Convert an external omg.omg.MatchResult to our internal MatchResult."""
        return cls(offset=external_match.offset, match=external_match.match)


@dataclass(unsafe_hash=True)
class ListMatchResult(MatchResult):
    """
    Extends MatchResult to represent a ListMatch in the AST, with additional context.
    """

    alias: str = ""  # The alias of the list that was matched (e.g., "word")
    sub_matches: tuple = field(
        default_factory=tuple
    )  # For quantified ListMatch, the individual items

    @classmethod
    def from_match_result(cls, mr: MatchResult, alias: str) -> "ListMatchResult":
        result = cls(
            offset=mr.offset,
            match=mr.match,
            alias=alias,
            sub_matches=(mr,),  # By default, a single ListMatch is its own sub_match
        )
        # Copy the evaluator attributes (ignore protected member warnings)
        result._ast_node = getattr(mr, "_ast_node", None)  # type: ignore
        result._constituent_matches = getattr(mr, "_constituent_matches", [])  # type: ignore
        return result


@dataclass(unsafe_hash=True)
class RuleMatch:
    name: str
    offset: int
    match: bytes  # The actual matched bytes from the haystack for the entire rule
    named_captures: dict
    # Resolver enrichment fields
    reference: Optional[int] = None
    sentence_end: Optional[int] = None
    paragraph_end: Optional[int] = None

    @property
    def length(self) -> int:
        """
        The length of the match, derived from the match bytes.
        """
        return len(self.match)
