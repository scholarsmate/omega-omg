# OmegaOMG: Omega Object Matching Grammar is a high-performance rule-based object matcher

OmegaOMG is a domain-specific language (DSL) and runtime engine for defining and evaluating complex matching rules against large byte-based inputs (“haystacks”). It leverages pre-anchored pattern matches (via the `OmegaMatch` library), advanced AST-based evaluation, and customizable entity resolution to deliver efficient, reliable object extraction pipelines.

## Features

- **Expressive DSL syntax**: Write rules using named list matches, literals, escapes, concatenation, alternation, and bounded quantifiers.
- **Pre-anchored matching**: Integrates with [OmegaMatch](https://github.com/scholarsmate/omega-match) to obtain longest, non-overlapping match streams for each alias.
- **AST-based evaluation**: Transforms parsed rules into an abstract syntax tree (AST), then evaluates with caching and fast-path optimizations.
- **Customizable resolvers**: Attach default or per-rule resolver configurations for entity enrichment (e.g., dates, numbers, custom tokens).
- **Performance optimizations**:
  - Pre-compiled regex patterns and Lark parser reuse
  - Typed cache keys and offset-indexed lookup tables
  - Binary-search anchoring and greedy extension for list matches
  - Adaptive sampling for quantifier anchors
- **Zero external dependencies** aside from standard Python libraries and `lark` / `omega_match`.

## Installation

1. Clone this repository:

   ```powershell
   git clone https://github.com/scholarsmate/omega-omg.git
   cd omega-omg
   ```

2. Create and activate a Python virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install requirements:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### 1. Define a DSL file

Create a `.omg` file with rules, e.g., `demo/demo.omg`:
```dsl
version 1.0

# Import match lists
import "name_prefix.txt" as prefix with word-boundary, ignore-case
import "names.txt" as given_name with word-boundary
import "surnames.txt" as surname with word-boundary
import "name_suffix.txt" as suffix with word-boundary
import "0000-9999.txt" as 4_digits with word-boundary
import "tlds.txt" as tld with word-boundary, ignore-case

# Configure the default resolver
resolver default uses exact with ignore-case, ignore-punctuation

# Top-level rule for matching a person's name
person = ( [[prefix]] \s{1,4} )? \
    [[given_name]] ( \s{1,4} [[given_name]] )? ( \s{1,4} \w | \s{1,4} \w "." )? \
    \s{1,4} [[surname]] \
    (\s{0,4} "," \s{1,4} [[suffix]])? \
    uses default resolver with optional-tokens("person-opt_tokens.txt")

# Dotted-rule references resolve to top-level person matches
person.prefix_surname = [[prefix]] \s{1,4} [[surname]] (\s{0,4} "," \s{1,4} [[suffix]])? \
    uses default resolver with optional-tokens("person-opt_tokens.txt")
person.surname = [[surname]] (\s{0,4} "," \s{1,4} [[suffix]])? \
    uses default resolver with optional-tokens("person-opt_tokens.txt")

# Rule for matching a phone number
phone = "(" \s{0,2} \d{3} \s{0,2} ")" \s{0,2} \d{3} "-" \s{0,2} [[4_digits]]

# Rule for matching email addresses with bounded quantifiers
# Pattern: username@domain.tld
# Username: 1-64 chars (alphanumeric, dots, hyphens, underscores)
# Domain: 1-253 chars total, each label 1-63 chars
email = [a-zA-Z0-9._-]{1,64} "@" [a-zA-Z0-9-]{1,63} ("." [a-zA-Z0-9-]{1,63}){0,10} "." [[tld]]
```

### 2. Parse and evaluate in Python

```python
from dsl.omg_parser import parse_file
from dsl.omg_evaluator import RuleEvaluator

# Load DSL and input haystack
ast = parse_file("demo/demo.omg")
with open("demo/CIA_Briefings_of_Presidential_Candidates_1952-1992.txt", "rb") as f:
    haystack = f.read()

# Evaluate a specific rule
engine = RuleEvaluator(ast_root=ast, haystack=haystack)
matches = engine.evaluate_rule(ast.rules["person"])
for m in matches:
    print(m.offset, m.match.decode())
```

### 3. Command-Line Tool

A command-line interface for batch processing is provided in `omg.py`. Run `python omg.py --help` for options.

#### Demo: End-to-End Object Matching and Highlighting

The following demonstrates how to use the CLI tools to extract and visualize matches from a text file using a demo OMG rule set:

1. **Run the matcher and output results to JSON:**

   ```powershell
   python .\omg.py --show-stats --show-timing --output matches.json .\demo\demo.omg .\demo\CIA_Briefings_of_Presidential_Candidates_1952-1992.txt
   ```
   This command will print timing and statistics to the terminal and write all matches to `matches.json` in UTF-8 with LF line endings.

2. **Render the matches as highlighted HTML:**

   ```powershell
   python .\highlighter.py .\demo\CIA_Briefings_of_Presidential_Candidates_1952-1992.txt .\matches.json CIA_demo.html
   ```
   This will generate an HTML file (`CIA_demo.html`) with all matched objects highlighted for easy review.

You can open the resulting HTML file in a browser to visually inspect the extracted matches.

## Project Structure

```
omg.py            # Object Matching Grammar CLI tool for running .omg files
highlighter.py    # Given found objects and the haystack, create a highlighted HTML
dsl/              # DSL parser, AST, transformer, evaluator, and resolver logic
resolver/         # Built-in resolver modules for entity enrichment
demo/             # Example rules and input files
tests/            # Unit and integration tests (pytest)
```

## Contributing

1. Fork the repo and create a feature branch.
2. Write tests under `tests/` for new features or bug fixes.
3. Run `pytest` to ensure all tests pass.
   ```powershell
   pytest
   ```
4. Submit a pull request.

## License

The OmegaOMG project is licensed under the [Apache License 2.0](LICENSE).

OmegaOMG is not an official Apache Software Foundation (ASF) project.
