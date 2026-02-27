"""Quickstart examples for aumai-neurosymbolic.

Run this file directly to verify your installation and see the library in action:

    python examples/quickstart.py

This file demonstrates:
  1. Parsing Prolog-like rules from text
  2. Running forward-chaining inference
  3. Using the explain() method
  4. Building knowledge bases programmatically
  5. Using DifferentiableLogic as a standalone soft-logic calculator
"""

from __future__ import annotations

from pathlib import Path

from aumai_neurosymbolic.core import (
    DifferentiableLogic,
    KnowledgeCompiler,
    SymbolicEngine,
)
from aumai_neurosymbolic.models import InferenceResult, KnowledgeBase, LogicRule


# ---------------------------------------------------------------------------
# Demo 1: Parsing rules from text and querying
# ---------------------------------------------------------------------------


def demo_family_tree() -> None:
    """Parse a family-tree knowledge base and prove ancestry relationships.

    Demonstrates:
    - KnowledgeCompiler.from_text()
    - SymbolicEngine.query()
    - Confidence propagation through recursive rules
    """
    print("=" * 60)
    print("Demo 1: Family Tree Inference")
    print("=" * 60)

    rules_text = """
% Ground facts
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).

% Rules with confidence weights
ancestor(X, Y) :- parent(X, Y). % confidence=1.0
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z). % confidence=0.95
"""

    compiler = KnowledgeCompiler()
    kb = compiler.from_text(rules_text)

    print(f"Loaded {len(kb.facts)} facts and {len(kb.rules)} rules.\n")

    engine = SymbolicEngine(kb)

    # Prove direct ancestry
    goals = [
        "ancestor(tom, bob)",   # direct parent -> True
        "ancestor(tom, ann)",   # grandparent  -> True, conf 0.95
        "ancestor(ann, tom)",   # reverse      -> False
        "ancestor(tom, pat)",   # another grandchild -> True
    ]

    for goal in goals:
        result: InferenceResult = engine.query(goal)
        status = "PROVEN " if result.result else "UNKNOWN"
        print(f"  {status}  {goal:<35} confidence={result.confidence:.4f}")

    print()


# ---------------------------------------------------------------------------
# Demo 2: Human-readable proof traces
# ---------------------------------------------------------------------------


def demo_explain() -> None:
    """Show the full proof trace for a derived fact.

    Demonstrates:
    - SymbolicEngine.explain()
    """
    print("=" * 60)
    print("Demo 2: Proof Trace Explanation")
    print("=" * 60)

    rules_text = """
parent(tom, bob).
parent(bob, ann).
ancestor(X, Y) :- parent(X, Y). % confidence=1.0
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z). % confidence=0.95
"""
    compiler = KnowledgeCompiler()
    kb = compiler.from_text(rules_text)
    engine = SymbolicEngine(kb)

    print(engine.explain("ancestor(tom, ann)"))
    print()

    # A fact that cannot be proven
    print(engine.explain("ancestor(ann, tom)"))
    print()


# ---------------------------------------------------------------------------
# Demo 3: Building knowledge bases programmatically
# ---------------------------------------------------------------------------


def demo_programmatic_kb() -> None:
    """Construct a knowledge base in Python and run access control queries.

    Demonstrates:
    - KnowledgeBase and LogicRule construction
    - SymbolicEngine.add_fact() at runtime
    """
    print("=" * 60)
    print("Demo 3: Programmatic Access Control Knowledge Base")
    print("=" * 60)

    kb = KnowledgeBase(
        facts=[
            "has_role(alice, admin)",
            "has_role(bob, viewer)",
            "has_role(carol, editor)",
            "resource_level(report_q4, confidential)",
            "resource_level(readme, public)",
            "resource_level(dashboard, internal)",
        ],
        rules=[
            # Admins can access everything
            LogicRule(
                rule_id="r_admin",
                head="can_access(X, R)",
                body=["has_role(X, admin)"],
                confidence=1.0,
            ),
            # Viewers can only access public resources
            LogicRule(
                rule_id="r_viewer_public",
                head="can_access(X, R)",
                body=["has_role(X, viewer)", "resource_level(R, public)"],
                confidence=0.99,
            ),
            # Editors can access internal and public resources
            LogicRule(
                rule_id="r_editor_internal",
                head="can_access(X, R)",
                body=["has_role(X, editor)", "resource_level(R, internal)"],
                confidence=0.99,
            ),
            LogicRule(
                rule_id="r_editor_public",
                head="can_access(X, R)",
                body=["has_role(X, editor)", "resource_level(R, public)"],
                confidence=0.99,
            ),
        ],
    )

    engine = SymbolicEngine(kb)

    queries = [
        ("alice", "report_q4"),
        ("alice", "readme"),
        ("bob", "readme"),
        ("bob", "report_q4"),   # viewer cannot access confidential
        ("carol", "dashboard"),
        ("carol", "report_q4"),  # editor cannot access confidential
    ]

    print(f"  {'User':<10} {'Resource':<20} {'Access':<10} Confidence")
    print(f"  {'-'*10} {'-'*20} {'-'*10} ----------")
    for user, resource in queries:
        goal = f"can_access({user}, {resource})"
        result = engine.query(goal)
        access = "ALLOWED" if result.result else "DENIED "
        print(f"  {user:<10} {resource:<20} {access:<10} {result.confidence:.4f}")

    # Add a new role dynamically
    print("\n  [Adding dave as admin at runtime]")
    engine.add_fact("has_role(dave, admin)")
    result = engine.query("can_access(dave, report_q4)")
    print(f"  dave -> report_q4: {'ALLOWED' if result.result else 'DENIED'}")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Serialization round-trip
# ---------------------------------------------------------------------------


def demo_serialization(tmp_dir: Path) -> None:
    """Demonstrate compiling rules to JSON and reloading.

    Demonstrates:
    - KnowledgeCompiler.to_json() and from_json()
    - KnowledgeCompiler.to_text() for inspection
    """
    print("=" * 60)
    print("Demo 4: Serialization Round-Trip")
    print("=" * 60)

    rules_text = """
% Eligibility rules for a loan application
high_income(X) :- income(X, high).
clean_credit(X) :- credit_score(X, good).
eligible(X) :- high_income(X), clean_credit(X). % confidence=0.98

income(applicant_A, high).
credit_score(applicant_A, good).
income(applicant_B, low).
credit_score(applicant_B, good).
"""

    compiler = KnowledgeCompiler()
    kb = compiler.from_text(rules_text)

    # Save to JSON
    json_path = tmp_dir / "eligibility.json"
    compiler.to_json(kb, json_path)
    print(f"  Saved to {json_path}")

    # Reload
    kb2 = compiler.from_json(json_path)
    print(f"  Reloaded: {len(kb2.rules)} rules, {len(kb2.facts)} facts")

    # Serialize back to text
    text_representation = compiler.to_text(kb2)
    print("\n  Prolog text representation:")
    for line in text_representation.splitlines():
        print(f"    {line}")

    # Query the reloaded KB
    engine = SymbolicEngine(kb2)
    for applicant in ["applicant_A", "applicant_B"]:
        result = engine.query(f"eligible({applicant})")
        status = "eligible" if result.result else "not eligible"
        print(f"\n  {applicant}: {status} (confidence={result.confidence:.4f})")
    print()


# ---------------------------------------------------------------------------
# Demo 5: DifferentiableLogic standalone calculator
# ---------------------------------------------------------------------------


def demo_differentiable_logic() -> None:
    """Use DifferentiableLogic directly for soft probability-like reasoning.

    Demonstrates:
    - soft_and, soft_or, soft_not
    - evaluate_rule for combining neural classifier outputs with rule weights
    """
    print("=" * 60)
    print("Demo 5: DifferentiableLogic (Product T-Norm)")
    print("=" * 60)

    dl = DifferentiableLogic()

    print("  Basic operations (product t-norm):")
    print(f"    AND(0.9, 0.8) = {dl.soft_and(0.9, 0.8):.4f}  (expected: 0.7200)")
    print(f"    OR (0.9, 0.8) = {dl.soft_or(0.9, 0.8):.4f}  (expected: 0.9800)")
    print(f"    NOT(0.9)      = {dl.soft_not(0.9):.4f}  (expected: 0.1000)")

    # Combining neural model outputs with symbolic rule confidence
    spam_classifier_confidence = 0.87   # neural model output
    policy_rule_weight = 0.95           # "high-confidence emails from admins are not spam"
    admin_context_confidence = 0.82     # neural NER detected "admin" role

    combined = dl.soft_and(
        dl.soft_and(spam_classifier_confidence, policy_rule_weight),
        admin_context_confidence,
    )
    print(f"\n  Spam detection + policy rule + admin context:")
    print(f"    spam_conf={spam_classifier_confidence}")
    print(f"    rule_weight={policy_rule_weight}")
    print(f"    admin_conf={admin_context_confidence}")
    print(f"    Combined AND: {combined:.4f}")

    # Evaluate a complete rule
    rule = LogicRule(
        rule_id="r_safe",
        head="safe_to_send(X)",
        body=["not_pii(X)", "not_confidential(X)"],
        confidence=0.98,
    )
    fact_confidences = {
        "not_pii(message_42)": 0.91,
        "not_confidential(message_42)": 0.85,
    }
    rule_value = dl.evaluate_rule(rule, fact_confidences)
    print(f"\n  evaluate_rule for 'safe_to_send':")
    print(f"    not_pii=0.91, not_confidential=0.85, rule.confidence=0.98")
    print(f"    Result: {rule_value:.4f}  (= 0.98 * 0.91 * 0.85)")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos in sequence."""
    import tempfile

    print("\naumai-neurosymbolic Quickstart Demos")
    print("=" * 60)
    print()

    demo_family_tree()
    demo_explain()
    demo_programmatic_kb()

    with tempfile.TemporaryDirectory() as tmp:
        demo_serialization(Path(tmp))

    demo_differentiable_logic()

    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
