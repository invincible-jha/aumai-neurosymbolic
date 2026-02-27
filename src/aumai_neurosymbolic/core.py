"""Core neuro-symbolic reasoning engine.

Provides:
- SymbolicEngine: forward-chaining inference over a KnowledgeBase.
- KnowledgeCompiler: parse/serialise Prolog-like text rules.
- DifferentiableLogic: product t-norm soft logic operators.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .models import InferenceResult, KnowledgeBase, LogicRule


# ---------------------------------------------------------------------------
# SymbolicEngine
# ---------------------------------------------------------------------------


class SymbolicEngine:
    """Forward-chaining inference engine over a KnowledgeBase.

    Uses iterative forward chaining: repeatedly applies rules whose bodies
    are fully satisfied by the current fact set until no new facts are derived
    or the query goal is proven.

    Example:
        >>> kb = KnowledgeBase(
        ...     facts=["parent(tom,bob)", "parent(bob,ann)"],
        ...     rules=[
        ...         LogicRule(rule_id="r1", head="ancestor(X,Y)", body=["parent(X,Y)"]),
        ...         LogicRule(rule_id="r2", head="ancestor(X,Z)",
        ...                   body=["parent(X,Y)", "ancestor(Y,Z)"]),
        ...     ],
        ... )
        >>> engine = SymbolicEngine(kb)
        >>> result = engine.query("ancestor(tom,ann)")
        >>> result.result
        True
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base
        # Mutable working set of derived + asserted facts
        self._derived_facts: set[str] = set(knowledge_base.facts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rule(self, rule: LogicRule) -> None:
        """Append a rule to the underlying knowledge base."""
        self._kb.add_rule(rule)

    def add_fact(self, fact: str) -> None:
        """Assert a ground fact and add it to the working set."""
        self._kb.add_fact(fact)
        self._derived_facts.add(fact.strip())

    def query(self, goal: str) -> InferenceResult:
        """Run forward-chaining inference and check whether *goal* is derivable.

        Args:
            goal: A ground literal to prove, e.g. ``"ancestor(tom,ann)"``.

        Returns:
            InferenceResult with ``result=True`` if the goal is derivable.
        """
        proof_chain: list[str] = []
        confidence = self._forward_chain(goal, proof_chain)
        proven = goal in self._derived_facts
        return InferenceResult(
            query=goal,
            result=proven,
            proof_chain=proof_chain,
            confidence=confidence if proven else 0.0,
        )

    def explain(self, goal: str) -> str:
        """Return a human-readable proof trace for *goal*.

        Args:
            goal: A ground literal to explain.

        Returns:
            Multi-line explanation string.
        """
        result = self.query(goal)
        lines: list[str] = [f"Query: {result.query}", f"Proven: {result.result}"]
        if result.result:
            lines.append(f"Confidence: {result.confidence:.4f}")
            lines.append("Proof chain:")
            for step in result.proof_chain:
                lines.append(f"  -> {step}")
        else:
            lines.append("No proof found.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_chain(self, goal: str, proof_chain: list[str]) -> float:
        """Saturate the fact set via forward chaining.

        Returns the confidence of the best proof path reaching *goal*,
        or 0.0 if the goal is not reached.
        """
        # Reset derived facts to base facts for a clean run
        self._derived_facts = set(self._kb.facts)
        # confidence_map tracks the best confidence for each derived literal
        confidence_map: dict[str, float] = {f: 1.0 for f in self._kb.facts}
        # proof_source maps each derived fact to its rule and antecedents
        proof_source: dict[str, tuple[LogicRule, list[str]]] = {}

        changed = True
        max_iterations = 1000
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for rule in self._kb.rules:
                # Attempt to ground the rule against the current fact set
                for grounding in self._ground_rule(rule, self._derived_facts):
                    grounded_head, grounded_body = grounding
                    if grounded_head not in self._derived_facts:
                        body_confidences = [
                            confidence_map.get(b, 1.0) for b in grounded_body
                        ]
                        derived_confidence = rule.confidence
                        for bc in body_confidences:
                            derived_confidence *= bc
                        self._derived_facts.add(grounded_head)
                        confidence_map[grounded_head] = derived_confidence
                        proof_source[grounded_head] = (rule, grounded_body)
                        changed = True

        # Reconstruct proof chain for goal
        if goal in self._derived_facts and goal in proof_source:
            self._build_proof_chain(goal, proof_source, proof_chain)

        return confidence_map.get(goal, 0.0)

    def _build_proof_chain(
        self,
        fact: str,
        proof_source: dict[str, tuple[LogicRule, list[str]]],
        chain: list[str],
    ) -> None:
        """Recursively build the proof chain for a derived fact."""
        if fact not in proof_source:
            if fact not in chain:
                chain.append(fact)
            return
        rule, body = proof_source[fact]
        for body_fact in body:
            self._build_proof_chain(body_fact, proof_source, chain)
        chain.append(f"[{rule.rule_id}] {' ^ '.join(body)} => {fact}")

    def _ground_rule(
        self, rule: LogicRule, facts: set[str]
    ) -> list[tuple[str, list[str]]]:
        """Attempt to ground rule variables against the known fact set.

        Returns a list of (grounded_head, grounded_body_list) pairs.
        Supports single-pass unification for rules with one variable level.
        """
        if not rule.body:
            # Fact-like rule: head is already ground
            return [(rule.head, [])]

        # Extract variable names from the rule (uppercase identifiers)
        variables = self._extract_variables(rule.head) | {
            v for b in rule.body for v in self._extract_variables(b)
        }

        if not variables:
            # Fully ground rule
            if all(b in facts for b in rule.body):
                return [(rule.head, list(rule.body))]
            return []

        # Try to find substitutions by matching the first body literal
        results: list[tuple[str, list[str]]] = []
        bindings_list = self._find_bindings(rule.body, facts, {})
        for bindings in bindings_list:
            grounded_head = self._apply_bindings(rule.head, bindings)
            grounded_body = [self._apply_bindings(b, bindings) for b in rule.body]
            results.append((grounded_head, grounded_body))
        return results

    def _find_bindings(
        self, body: list[str], facts: set[str], current_bindings: dict[str, str]
    ) -> list[dict[str, str]]:
        """Recursively find all variable bindings that satisfy the body."""
        if not body:
            return [current_bindings]

        first, *rest = body
        grounded_first = self._apply_bindings(first, current_bindings)

        all_bindings: list[dict[str, str]] = []
        for fact in facts:
            new_bindings = self._unify(grounded_first, fact, dict(current_bindings))
            if new_bindings is not None:
                extended = self._find_bindings(rest, facts, new_bindings)
                all_bindings.extend(extended)
        return all_bindings

    def _unify(
        self, pattern: str, fact: str, bindings: dict[str, str]
    ) -> Optional[dict[str, str]]:
        """Attempt to unify *pattern* with *fact*, updating *bindings*.

        Returns updated bindings on success, None on failure.
        """
        pattern_name, pattern_args = self._parse_literal(pattern)
        fact_name, fact_args = self._parse_literal(fact)

        if pattern_name != fact_name:
            return None
        if len(pattern_args) != len(fact_args):
            return None

        result = dict(bindings)
        for p_arg, f_arg in zip(pattern_args, fact_args):
            if p_arg.isupper() or (len(p_arg) > 1 and p_arg[0].isupper()):
                # p_arg is a variable
                if p_arg in result:
                    if result[p_arg] != f_arg:
                        return None
                else:
                    result[p_arg] = f_arg
            else:
                # p_arg is a constant
                if p_arg != f_arg:
                    return None
        return result

    def _parse_literal(self, literal: str) -> tuple[str, list[str]]:
        """Parse ``predicate(arg1, arg2, ...)`` into (name, args)."""
        match = re.match(r"(\w+)\(([^)]*)\)", literal.strip())
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = [a.strip() for a in args_str.split(",") if a.strip()]
            return name, args
        # Atom with no arguments
        return literal.strip(), []

    def _extract_variables(self, literal: str) -> set[str]:
        """Extract uppercase variable names from a literal."""
        _, args = self._parse_literal(literal)
        return {a for a in args if a and (a.isupper() or (len(a) > 1 and a[0].isupper()))}

    def _apply_bindings(self, literal: str, bindings: dict[str, str]) -> str:
        """Substitute variable bindings into a literal string."""
        name, args = self._parse_literal(literal)
        if not args:
            return bindings.get(literal, literal)
        grounded_args = [bindings.get(a, a) for a in args]
        return f"{name}({', '.join(grounded_args)})"


# ---------------------------------------------------------------------------
# KnowledgeCompiler
# ---------------------------------------------------------------------------


class KnowledgeCompiler:
    """Parse and serialise Prolog-like rule text.

    Rule syntax (one per line):
        head :- body1, body2.    # rule
        fact.                    # ground fact
        % comment                # ignored

    Example:
        >>> compiler = KnowledgeCompiler()
        >>> kb = compiler.from_text("parent(tom,bob).\\nancestor(X,Y) :- parent(X,Y).")
        >>> len(kb.rules)
        1
        >>> len(kb.facts)
        1
    """

    _RULE_PATTERN = re.compile(r"^(.+?)\s*:-\s*(.+?)\s*\.\s*(%.*)?$")
    _FACT_PATTERN = re.compile(r"^([^%][^:]*)\.$")

    def _split_body(self, body_raw: str) -> list[str]:
        """Split a rule body string on commas that are not inside parentheses.

        Prevents naive splitting of multi-arg predicates like ``parent(X,Y)``
        into ``["parent(X", "Y)"]``.

        Args:
            body_raw: Raw body string from a rule line, e.g. ``"parent(X,Y), mortal(X)"``.

        Returns:
            List of individual body literal strings with surrounding whitespace stripped.
        """
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for char in body_raw:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append("".join(current).strip())
        return [p for p in parts if p]

    def from_text(self, text: str) -> KnowledgeBase:
        """Parse Prolog-like text into a KnowledgeBase.

        Args:
            text: Multi-line string of Prolog-style rules and facts.

        Returns:
            Populated KnowledgeBase instance.
        """
        rules: list[LogicRule] = []
        facts: list[str] = []
        rule_counter = 0

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            rule_match = self._RULE_PATTERN.match(line)
            if rule_match:
                rule_counter += 1
                head = rule_match.group(1).strip()
                body_raw = rule_match.group(2).strip()
                body = self._split_body(body_raw)

                # Extract confidence annotation if present: confidence=0.8
                confidence = 1.0
                conf_match = re.search(r"confidence=([\d.]+)", line)
                if conf_match:
                    confidence = float(conf_match.group(1))

                rules.append(
                    LogicRule(
                        rule_id=f"r{rule_counter}",
                        head=head,
                        body=body,
                        confidence=confidence,
                    )
                )
                continue

            fact_match = self._FACT_PATTERN.match(line)
            if fact_match:
                candidate = fact_match.group(1).strip()
                if candidate and ":-" not in candidate:
                    facts.append(candidate)

        return KnowledgeBase(rules=rules, facts=facts)

    def from_json(self, path: Path) -> KnowledgeBase:
        """Load a KnowledgeBase from a JSON file.

        Args:
            path: Path to a JSON file containing serialised KnowledgeBase.

        Returns:
            Deserialised KnowledgeBase.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return KnowledgeBase.model_validate(data)

    def to_text(self, kb: KnowledgeBase) -> str:
        """Serialise a KnowledgeBase to Prolog-like text.

        Args:
            kb: The knowledge base to serialise.

        Returns:
            String representation of all facts and rules.
        """
        lines: list[str] = []
        for fact in kb.facts:
            lines.append(f"{fact}.")
        for rule in kb.rules:
            if rule.body:
                body_str = ", ".join(rule.body)
                lines.append(f"{rule.head} :- {body_str}. % confidence={rule.confidence:.4f}")
            else:
                lines.append(f"{rule.head}.")
        return "\n".join(lines)

    def to_json(self, kb: KnowledgeBase, path: Path) -> None:
        """Serialise a KnowledgeBase to a JSON file.

        Args:
            kb: The knowledge base to serialise.
            path: Destination file path.
        """
        path.write_text(kb.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# DifferentiableLogic
# ---------------------------------------------------------------------------


class DifferentiableLogic:
    """Soft (differentiable) logic operators using the product t-norm.

    The product t-norm defines:
    - AND(a, b)  = a * b
    - OR(a, b)   = a + b - a * b  (probabilistic sum)
    - NOT(a)     = 1 - a

    All values are expected in [0, 1].

    Example:
        >>> dl = DifferentiableLogic()
        >>> dl.soft_and(0.9, 0.8)
        0.72
        >>> round(dl.soft_or(0.9, 0.8), 10)
        0.98
        >>> dl.soft_not(0.9)
        0.09999999999999998
    """

    def soft_and(self, a: float, b: float) -> float:
        """Product t-norm conjunction.

        Args:
            a: First truth value in [0, 1].
            b: Second truth value in [0, 1].

        Returns:
            Soft conjunction value.
        """
        return a * b

    def soft_or(self, a: float, b: float) -> float:
        """Probabilistic sum (Lukasiewicz co-norm).

        Args:
            a: First truth value in [0, 1].
            b: Second truth value in [0, 1].

        Returns:
            Soft disjunction value.
        """
        return a + b - a * b

    def soft_not(self, a: float) -> float:
        """Standard negation.

        Args:
            a: Truth value in [0, 1].

        Returns:
            1 - a.
        """
        return 1.0 - a

    def evaluate_rule(self, rule: LogicRule, fact_confidences: dict[str, float]) -> float:
        """Evaluate the soft truth value of a rule given body confidences.

        Computes: rule.confidence * AND(body[0], body[1], ..., body[n])

        Args:
            rule: The logic rule to evaluate.
            fact_confidences: Mapping from literal string to confidence value.

        Returns:
            Soft truth value of the rule head, in [0, 1].
        """
        body_values = [fact_confidences.get(b, 0.0) for b in rule.body]
        if not body_values:
            return rule.confidence

        conjunction = body_values[0]
        for value in body_values[1:]:
            conjunction = self.soft_and(conjunction, value)

        return self.soft_and(rule.confidence, conjunction)
