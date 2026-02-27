# Getting Started with aumai-neurosymbolic

This guide walks you from zero to a working neuro-symbolic reasoning system in
under 15 minutes.

---

## Prerequisites

- Python 3.11 or higher
- `pip` (standard) or `uv` (recommended for speed)
- Basic familiarity with Python; no prior knowledge of Prolog or logic
  programming is required

---

## Installation

### From PyPI (stable)

```bash
pip install aumai-neurosymbolic
```

### From source (development)

```bash
git clone https://github.com/AumAI/aumai-neurosymbolic
cd aumai-neurosymbolic
pip install -e ".[dev]"
```

### Verify the installation

```bash
aumai-neurosymbolic --version
# aumai-neurosymbolic, version 0.1.0

python -c "import aumai_neurosymbolic; print('OK')"
# OK
```

---

## Step-by-Step Tutorial

### Step 1: Write your first rules file

Create a file called `family.txt`:

```prolog
% family.txt — a simple family knowledge base

% Ground facts
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).

% Rules
ancestor(X, Y) :- parent(X, Y). % confidence=1.0
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z). % confidence=0.95
```

### Step 2: Compile to JSON

The JSON format is portable and can be loaded programmatically or by the CLI:

```bash
aumai-neurosymbolic compile --input family.txt --output family.json
# Compiled 2 rule(s) and 4 fact(s) to family.json
```

Inspect `family.json` to see the validated, structured knowledge base.

### Step 3: Query the knowledge base

```bash
aumai-neurosymbolic query --kb family.json --goal "ancestor(tom, ann)"
# PROVEN: ancestor(tom, ann)  (confidence=0.9500)
# Proof chain:
#   parent(tom, bob)
#   parent(bob, ann)
#   [r1] parent(bob, ann) => ancestor(bob, ann)
#   [r2] parent(tom, bob) ^ ancestor(bob, ann) => ancestor(tom, ann)
```

Try a query that cannot be proven:

```bash
aumai-neurosymbolic query --kb family.json --goal "ancestor(ann, tom)"
# NOT PROVEN: ancestor(ann, tom)  (confidence=0.0000)
```

### Step 4: Use the Python API

```python
from pathlib import Path
from aumai_neurosymbolic.core import KnowledgeCompiler, SymbolicEngine

compiler = KnowledgeCompiler()
kb = compiler.from_json(Path("family.json"))
engine = SymbolicEngine(kb)

# Batch query
goals = ["ancestor(tom, ann)", "ancestor(tom, pat)", "ancestor(ann, tom)"]
for goal in goals:
    result = engine.query(goal)
    status = "YES" if result.result else "NO"
    print(f"{status}  {goal}  (confidence={result.confidence:.2f})")
```

Expected output:

```
YES  ancestor(tom, ann)  (confidence=0.95)
YES  ancestor(tom, pat)  (confidence=0.95)
NO   ancestor(ann, tom)  (confidence=0.00)
```

### Step 5: Add facts at runtime

```python
# Dynamically extend the knowledge base
engine.add_fact("parent(ann, sue)")
result = engine.query("ancestor(tom, sue)")
print(result.result)      # True
print(result.confidence)  # 0.9025  (0.95 * 0.95)
```

### Step 6: Use the explain() method

```python
print(engine.explain("ancestor(tom, sue)"))
```

Output:

```
Query: ancestor(tom, sue)
Proven: True
Confidence: 0.9025
Proof chain:
  -> parent(tom, bob)
  -> parent(bob, ann)
  -> [r1] parent(bob, ann) => ancestor(bob, ann)
  -> [r2] parent(tom, bob) ^ ancestor(bob, ann) => ancestor(tom, ann)
  -> parent(ann, sue)
  -> [r1] parent(ann, sue) => ancestor(ann, sue)
  -> [r2] parent(tom, bob) ^ ancestor(bob, ann) ^ ancestor(ann, sue) => ancestor(tom, sue)
```

---

## Common Patterns and Recipes

### Pattern 1: Access control rules

Model fine-grained access control as a knowledge base:

```python
from aumai_neurosymbolic.models import KnowledgeBase, LogicRule
from aumai_neurosymbolic.core import SymbolicEngine

kb = KnowledgeBase(
    facts=[
        "has_role(alice, admin)",
        "has_role(bob, viewer)",
        "resource_sensitivity(report_Q4, high)",
        "resource_sensitivity(readme, low)",
    ],
    rules=[
        LogicRule(
            rule_id="admin_access",
            head="can_read(X, R)",
            body=["has_role(X, admin)"],
            confidence=1.0,
        ),
        LogicRule(
            rule_id="viewer_low",
            head="can_read(X, R)",
            body=["has_role(X, viewer)", "resource_sensitivity(R, low)"],
            confidence=0.99,
        ),
    ],
)

engine = SymbolicEngine(kb)
print(engine.query("can_read(alice, report_Q4)").result)   # True
print(engine.query("can_read(bob, report_Q4)").result)     # False
print(engine.query("can_read(bob, readme)").result)        # True
```

### Pattern 2: Confidence-weighted expert rules

Use confidence weights to express rule reliability:

```python
from aumai_neurosymbolic.core import KnowledgeCompiler, SymbolicEngine

rules = """
% High-confidence rule from validated policy
approved_vendor(X) :- certified(X), insured(X). % confidence=0.99

% Heuristic rule based on historical patterns
likely_approved(X) :- approved_vendor(X), region(X, us). % confidence=0.8

certified(acme_corp).
insured(acme_corp).
region(acme_corp, us).
"""

compiler = KnowledgeCompiler()
kb = compiler.from_text(rules)
engine = SymbolicEngine(kb)

result = engine.query("likely_approved(acme_corp)")
print(f"Confidence: {result.confidence:.4f}")  # 0.99 * 0.8 = 0.792
```

### Pattern 3: Round-trip serialization

Maintain rules as version-controlled text files, compile to JSON for runtime:

```python
from pathlib import Path
from aumai_neurosymbolic.core import KnowledgeCompiler

compiler = KnowledgeCompiler()

# Load from text (version-controlled source of truth)
kb = compiler.from_text(Path("rules/access_control.txt").read_text())

# Save compiled form for deployment
compiler.to_json(kb, Path("dist/access_control.json"))

# Load compiled form in production
kb_prod = compiler.from_json(Path("dist/access_control.json"))
```

### Pattern 4: Soft logic standalone usage

Use `DifferentiableLogic` as a standalone utility for probability-like
reasoning, independent of the full engine:

```python
from aumai_neurosymbolic.core import DifferentiableLogic

dl = DifferentiableLogic()

# Combine neural classifier outputs
classifier_confidence = 0.87   # "is_spam" from a neural model
rule_confidence = 0.95         # policy rule weight

# AND: both conditions must hold
combined = dl.soft_and(classifier_confidence, rule_confidence)
print(f"Combined: {combined:.4f}")  # 0.8265

# NOT: invert
print(f"NOT spam: {dl.soft_not(classifier_confidence):.4f}")  # 0.13

# OR: at least one condition holds
alert_1 = 0.7
alert_2 = 0.4
print(f"Alert OR: {dl.soft_or(alert_1, alert_2):.4f}")  # 0.82
```

### Pattern 5: Building a dynamic fact set from LLM output

```python
from aumai_neurosymbolic.models import KnowledgeBase, LogicRule
from aumai_neurosymbolic.core import SymbolicEngine

# Start with fixed rules
kb = KnowledgeBase(
    rules=[
        LogicRule(
            rule_id="safe",
            head="request_safe(X)",
            body=["no_pii(X)", "no_secrets(X)"],
            confidence=1.0,
        )
    ]
)

engine = SymbolicEngine(kb)

# Inject facts from LLM classifier outputs (structured output)
def classify_request(request_id: str, text: str) -> None:
    # Hypothetical: LLM returns structured classification
    if "john@example.com" not in text:
        engine.add_fact(f"no_pii({request_id})")
    if "password" not in text.lower():
        engine.add_fact(f"no_secrets({request_id})")

classify_request("req_001", "What is the weather today?")
print(engine.query("request_safe(req_001)").result)  # True

classify_request("req_002", "My email is john@example.com, reset password")
print(engine.query("request_safe(req_002)").result)  # False
```

---

## Troubleshooting FAQ

**Q: My rule fires but the confidence is lower than I expected.**

A: Confidence propagates via the product t-norm: each rule in the derivation
chain multiplies its weight. A deep proof chain (e.g., 5 rules each with
confidence 0.9) will yield `0.9^5 = 0.59`. Check your `result.proof_chain` to
see which rules are in the derivation path.

**Q: My recursive rule causes the engine to run for a long time.**

A: The engine caps at 1,000 forward-chaining iterations. If your knowledge
base is very large or deeply recursive, the fixpoint may not be reached within
this limit. Try pre-grounding large fact sets before adding recursive rules.

**Q: Variables in my rule are not being substituted correctly.**

A: Variable names must start with an uppercase letter (e.g., `X`, `Y`, `Person`,
`Role`). Lowercase identifiers are always treated as constants. Check for
accidental lowercase variable names.

**Q: I get `ValueError: Field must not be blank` when creating a `LogicRule`.**

A: `rule_id` and `head` are required non-blank strings. Ensure you are passing
values for both fields.

**Q: The `compile` command produces zero rules.**

A: Verify that your rules end with a period (`.`) and that the body uses `:-`.
Lines starting with `%` are comments and are skipped. A common mistake is
missing the trailing period on a rule line.

**Q: My fact contains parentheses but is not being matched.**

A: Only facts in the form `predicate(arg1, arg2)` or simple atoms (`atom`) are
supported. Ensure your fact literals match the Prolog term format exactly.

---

## Next Steps

- Read the [API Reference](api-reference.md) for complete class/method documentation
- Explore [examples/quickstart.py](../examples/quickstart.py) for runnable demos
- Integrate with [aumai-policyminer](https://github.com/AumAI/aumai-policyminer)
  to convert mined policies into symbolic rules
- Join the [Discord community](https://discord.gg/aumai)
