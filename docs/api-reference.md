# API Reference — aumai-neurosymbolic

Complete reference for all public classes, methods, and Pydantic models in
`aumai_neurosymbolic`. All classes are importable from their module paths
shown below.

---

## Module: `aumai_neurosymbolic.models`

### `LogicRule`

```python
class LogicRule(BaseModel):
```

A single Horn-clause style logic rule with a confidence weight.

**Fields:**

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `rule_id` | `str` | Yes | — | `min_length=1` | Unique identifier for the rule |
| `head` | `str` | Yes | — | `min_length=1` | The conclusion literal produced when the body is satisfied |
| `body` | `list[str]` | No | `[]` | — | Premise literals that must all be satisfied |
| `confidence` | `float` | No | `1.0` | `ge=0.0, le=1.0` | Probability-like weight representing rule strength |

**Validators:**

- `strip_whitespace`: strips surrounding whitespace from `rule_id` and `head`
- `strip_body_elements`: strips whitespace from each element of `body`

**Examples:**

```python
from aumai_neurosymbolic.models import LogicRule

# Fully specified rule
rule = LogicRule(
    rule_id="r1",
    head="ancestor(X, Z)",
    body=["parent(X, Y)", "ancestor(Y, Z)"],
    confidence=0.95,
)

# Fact-like rule (empty body)
fact_rule = LogicRule(rule_id="f1", head="human(socrates)")

# Minimum example
minimal = LogicRule(rule_id="r2", head="mortal(X)", body=["human(X)"])
```

---

### `KnowledgeBase`

```python
class KnowledgeBase(BaseModel):
```

A collection of rules and ground facts forming a logic program.

**Fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `rules` | `list[LogicRule]` | No | `[]` | Ordered list of LogicRule objects |
| `facts` | `list[str]` | No | `[]` | Ground-truth literals asserted without proof |

**Methods:**

#### `add_rule(rule: LogicRule) -> None`

Appends a rule to the knowledge base.

**Parameters:**
- `rule` — the `LogicRule` to append

**Example:**

```python
from aumai_neurosymbolic.models import KnowledgeBase, LogicRule

kb = KnowledgeBase()
kb.add_rule(LogicRule(rule_id="r1", head="mortal(X)", body=["human(X)"]))
```

#### `add_fact(fact: str) -> None`

Asserts a new ground fact (leading/trailing whitespace is stripped).

**Parameters:**
- `fact` — ground literal string, e.g. `"human(socrates)"`

**Example:**

```python
kb = KnowledgeBase()
kb.add_fact("human(socrates)")
kb.add_fact("parent(tom, bob)")
```

---

### `InferenceResult`

```python
class InferenceResult(BaseModel):
```

The outcome of a single query against a knowledge base.

**Fields:**

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `query` | `str` | Yes | — | — | The goal literal that was queried |
| `result` | `bool` | Yes | — | — | Whether the query was proven true |
| `proof_chain` | `list[str]` | No | `[]` | — | Ordered list of literals and rule applications visited during proof |
| `confidence` | `float` | No | `0.0` | `ge=0.0, le=1.0` | Product of rule confidences along the proof chain |

**Example:**

```python
from aumai_neurosymbolic.models import InferenceResult

res = InferenceResult(
    query="ancestor(tom, ann)",
    result=True,
    proof_chain=["parent(tom, bob)", "[r1] parent(bob, ann) => ancestor(bob, ann)"],
    confidence=0.95,
)
```

---

## Module: `aumai_neurosymbolic.core`

### `SymbolicEngine`

```python
class SymbolicEngine:
```

Forward-chaining inference engine over a `KnowledgeBase`.

Uses iterative forward chaining: repeatedly applies rules whose bodies are
fully satisfied by the current fact set until no new facts are derived or the
goal is proven, capped at 1,000 iterations.

#### `__init__(knowledge_base: KnowledgeBase) -> None`

**Parameters:**
- `knowledge_base` — the initial `KnowledgeBase` to reason over

**Example:**

```python
from aumai_neurosymbolic.models import KnowledgeBase, LogicRule
from aumai_neurosymbolic.core import SymbolicEngine

kb = KnowledgeBase(
    facts=["parent(tom, bob)", "parent(bob, ann)"],
    rules=[LogicRule(rule_id="r1", head="ancestor(X, Y)", body=["parent(X, Y)"])]
)
engine = SymbolicEngine(kb)
```

#### `add_rule(rule: LogicRule) -> None`

Append a rule to the underlying knowledge base.

**Parameters:**
- `rule` — the `LogicRule` to add

#### `add_fact(fact: str) -> None`

Assert a ground fact into both the knowledge base and the current working set.

**Parameters:**
- `fact` — ground literal string

#### `query(goal: str) -> InferenceResult`

Run forward-chaining inference and check whether `goal` is derivable.

**Parameters:**
- `goal` — a ground literal to prove, e.g. `"ancestor(tom, ann)"`

**Returns:**
- `InferenceResult` with `result=True` if the goal is derivable. If
  `result=False`, `confidence` is `0.0` and `proof_chain` is empty.

**Example:**

```python
result = engine.query("ancestor(tom, ann)")
print(result.result)       # True or False
print(result.confidence)   # float in [0, 1]
print(result.proof_chain)  # list of step strings
```

#### `explain(goal: str) -> str`

Return a human-readable proof trace for `goal`.

**Parameters:**
- `goal` — ground literal to explain

**Returns:**
- Multi-line string beginning with `Query:` and `Proven:`, followed by
  `Confidence:` and `Proof chain:` if proven, or `No proof found.` if not.

**Example:**

```python
print(engine.explain("ancestor(tom, ann)"))
# Query: ancestor(tom, ann)
# Proven: True
# Confidence: 0.9500
# Proof chain:
#   -> parent(tom, bob)
#   -> [r1] parent(tom, bob) => ancestor(tom, bob)
```

---

### `KnowledgeCompiler`

```python
class KnowledgeCompiler:
```

Parse and serialise Prolog-like rule text to/from `KnowledgeBase`.

**Rule syntax** (one clause per line):

```
head :- body1, body2.    % rule with optional % confidence=<float>
fact.                    % ground fact
% comment               % ignored
```

#### `from_text(text: str) -> KnowledgeBase`

Parse Prolog-like text into a `KnowledgeBase`.

**Parameters:**
- `text` — multi-line string of Prolog-style rules and facts

**Returns:**
- Populated `KnowledgeBase` instance with auto-assigned rule IDs (`r1`, `r2`, ...)

**Example:**

```python
from aumai_neurosymbolic.core import KnowledgeCompiler

compiler = KnowledgeCompiler()
kb = compiler.from_text("""
parent(tom, bob).
ancestor(X, Y) :- parent(X, Y). % confidence=0.9
""")
print(len(kb.facts))   # 1
print(len(kb.rules))   # 1
```

#### `from_json(path: Path) -> KnowledgeBase`

Load a `KnowledgeBase` from a JSON file (must have been saved with `to_json`).

**Parameters:**
- `path` — `pathlib.Path` to a JSON file

**Returns:**
- Deserialised `KnowledgeBase`

**Raises:**
- `json.JSONDecodeError` if the file is not valid JSON
- `pydantic.ValidationError` if the JSON does not match the `KnowledgeBase`
  schema

#### `to_text(kb: KnowledgeBase) -> str`

Serialise a `KnowledgeBase` to Prolog-like text.

**Parameters:**
- `kb` — the knowledge base to serialise

**Returns:**
- String with one line per fact or rule. Rule confidences are included as
  inline comments: `% confidence=<float>`.

#### `to_json(kb: KnowledgeBase, path: Path) -> None`

Serialise a `KnowledgeBase` to a JSON file.

**Parameters:**
- `kb` — the knowledge base to serialise
- `path` — destination `pathlib.Path`

#### `_split_body(body_raw: str) -> list[str]`

*Internal.* Split a rule body string on commas that are not inside parentheses.
For example, `"parent(X, Y), mortal(X)"` splits to
`["parent(X, Y)", "mortal(X)"]`.

---

### `DifferentiableLogic`

```python
class DifferentiableLogic:
```

Soft (differentiable) logic operators using the **product t-norm**.

All truth values are expected in `[0, 1]`. The operations are:
- `AND(a, b) = a * b`
- `OR(a, b) = a + b - a * b` (probabilistic sum)
- `NOT(a) = 1 - a`

#### `soft_and(a: float, b: float) -> float`

Product t-norm conjunction.

**Parameters:**
- `a` — first truth value in `[0, 1]`
- `b` — second truth value in `[0, 1]`

**Returns:** `a * b`

**Example:**

```python
dl = DifferentiableLogic()
print(dl.soft_and(0.9, 0.8))   # 0.72
```

#### `soft_or(a: float, b: float) -> float`

Probabilistic sum (Lukasiewicz co-norm): `a + b - a * b`.

**Parameters:**
- `a` — first truth value in `[0, 1]`
- `b` — second truth value in `[0, 1]`

**Returns:** `a + b - a * b`

**Example:**

```python
print(dl.soft_or(0.9, 0.8))   # 0.98
```

#### `soft_not(a: float) -> float`

Standard negation: `1 - a`.

**Parameters:**
- `a` — truth value in `[0, 1]`

**Returns:** `1 - a`

#### `evaluate_rule(rule: LogicRule, fact_confidences: dict[str, float]) -> float`

Evaluate the soft truth value of a rule given body literal confidences.

Computes: `rule.confidence * AND(body[0], body[1], ..., body[n])`

**Parameters:**
- `rule` — the `LogicRule` to evaluate
- `fact_confidences` — mapping from literal string to confidence value;
  literals not present default to `0.0`

**Returns:**
- Soft truth value of the rule head, in `[0, 1]`

**Example:**

```python
from aumai_neurosymbolic.models import LogicRule
from aumai_neurosymbolic.core import DifferentiableLogic

rule = LogicRule(rule_id="r1", head="goal(X)", body=["a(X)", "b(X)"], confidence=0.9)
dl = DifferentiableLogic()
conf = dl.evaluate_rule(rule, {"a(alice)": 0.8, "b(alice)": 0.7})
print(conf)  # 0.9 * 0.8 * 0.7 = 0.504
```

---

## Module: `aumai_neurosymbolic.cli`

### `main`

CLI entry point registered as `aumai-neurosymbolic`.

Commands:

| Command | Description |
|---|---|
| `query` | Prove a goal against a JSON knowledge base |
| `compile` | Convert Prolog-like text rules to JSON |

See [README.md](../README.md) for full CLI usage with examples.

---

## Package-level exports

`aumai_neurosymbolic.__version__` — current version string (`"0.1.0"`).

There are no explicit re-exports from `__init__.py`. Import directly from
submodules:

```python
from aumai_neurosymbolic.core import SymbolicEngine, KnowledgeCompiler, DifferentiableLogic
from aumai_neurosymbolic.models import KnowledgeBase, LogicRule, InferenceResult
```
