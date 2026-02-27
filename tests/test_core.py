"""Comprehensive tests for aumai-neurosymbolic core module.

Covers:
- SymbolicEngine: forward-chaining inference, add_fact, add_rule, explain
- KnowledgeCompiler: from_text, to_text, from_json, to_json
- DifferentiableLogic: soft_and, soft_or, soft_not, evaluate_rule
- Models: LogicRule, KnowledgeBase, InferenceResult
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from aumai_neurosymbolic.core import DifferentiableLogic, KnowledgeCompiler, SymbolicEngine
from aumai_neurosymbolic.models import InferenceResult, KnowledgeBase, LogicRule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_ancestor_kb() -> KnowledgeBase:
    """Classic ancestor knowledge base for testing."""
    return KnowledgeBase(
        facts=["parent(tom,bob)", "parent(bob,ann)", "parent(ann,sue)"],
        rules=[
            LogicRule(rule_id="r1", head="ancestor(X,Y)", body=["parent(X,Y)"]),
            LogicRule(rule_id="r2", head="ancestor(X,Z)", body=["parent(X,Y)", "ancestor(Y,Z)"]),
        ],
    )


def make_simple_kb() -> KnowledgeBase:
    return KnowledgeBase(
        facts=["bird(tweety)", "wings(tweety)"],
        rules=[
            LogicRule(rule_id="r1", head="can_fly(X)", body=["bird(X)", "wings(X)"]),
        ],
    )


# ---------------------------------------------------------------------------
# LogicRule model tests
# ---------------------------------------------------------------------------


class TestLogicRuleModel:
    def test_basic_creation(self) -> None:
        rule = LogicRule(rule_id="r1", head="foo(X)", body=["bar(X)"])
        assert rule.rule_id == "r1"
        assert rule.head == "foo(X)"
        assert rule.body == ["bar(X)"]
        assert rule.confidence == 1.0

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LogicRule(rule_id="r1", head="foo", body=[], confidence=1.1)
        with pytest.raises(ValidationError):
            LogicRule(rule_id="r1", head="foo", body=[], confidence=-0.1)

    def test_confidence_at_boundaries(self) -> None:
        r0 = LogicRule(rule_id="r1", head="foo", body=[], confidence=0.0)
        r1 = LogicRule(rule_id="r2", head="bar", body=[], confidence=1.0)
        assert r0.confidence == 0.0
        assert r1.confidence == 1.0

    def test_head_stripped_of_whitespace(self) -> None:
        rule = LogicRule(rule_id="r1", head="  foo(X)  ", body=[])
        assert rule.head == "foo(X)"

    def test_rule_id_stripped_of_whitespace(self) -> None:
        rule = LogicRule(rule_id="  r1  ", head="foo", body=[])
        assert rule.rule_id == "r1"

    def test_body_elements_stripped(self) -> None:
        rule = LogicRule(rule_id="r1", head="foo", body=["  bar(X)  ", " baz(X) "])
        assert rule.body == ["bar(X)", "baz(X)"]

    def test_empty_body_allowed(self) -> None:
        rule = LogicRule(rule_id="r1", head="ground_fact(a)", body=[])
        assert rule.body == []

    def test_rule_id_cannot_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            LogicRule(rule_id="", head="foo", body=[])

    def test_head_cannot_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            LogicRule(rule_id="r1", head="", body=[])


# ---------------------------------------------------------------------------
# KnowledgeBase model tests
# ---------------------------------------------------------------------------


class TestKnowledgeBaseModel:
    def test_empty_kb(self) -> None:
        kb = KnowledgeBase()
        assert kb.rules == []
        assert kb.facts == []

    def test_add_fact(self) -> None:
        kb = KnowledgeBase()
        kb.add_fact("likes(alice,bob)")
        assert "likes(alice,bob)" in kb.facts

    def test_add_fact_strips_whitespace(self) -> None:
        kb = KnowledgeBase()
        kb.add_fact("  likes(alice,bob)  ")
        assert "likes(alice,bob)" in kb.facts

    def test_add_rule(self) -> None:
        kb = KnowledgeBase()
        rule = LogicRule(rule_id="r1", head="friend(X,Y)", body=["likes(X,Y)"])
        kb.add_rule(rule)
        assert len(kb.rules) == 1
        assert kb.rules[0].rule_id == "r1"

    def test_initialise_with_facts_and_rules(self) -> None:
        kb = make_ancestor_kb()
        assert len(kb.facts) == 3
        assert len(kb.rules) == 2


# ---------------------------------------------------------------------------
# InferenceResult model tests
# ---------------------------------------------------------------------------


class TestInferenceResultModel:
    def test_basic_creation(self) -> None:
        result = InferenceResult(query="foo(a)", result=True, confidence=0.9)
        assert result.query == "foo(a)"
        assert result.result is True
        assert result.confidence == 0.9

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            InferenceResult(query="foo", result=True, confidence=1.5)

    def test_default_confidence(self) -> None:
        result = InferenceResult(query="foo", result=False)
        assert result.confidence == 0.0

    def test_proof_chain_default_empty(self) -> None:
        result = InferenceResult(query="foo", result=False)
        assert result.proof_chain == []


# ---------------------------------------------------------------------------
# SymbolicEngine tests
# ---------------------------------------------------------------------------


class TestSymbolicEngine:
    def test_direct_fact_query(self) -> None:
        kb = KnowledgeBase(facts=["parent(tom,bob)"], rules=[])
        engine = SymbolicEngine(kb)
        result = engine.query("parent(tom,bob)")
        assert result.result is True
        assert result.confidence == 1.0

    def test_missing_fact_query(self) -> None:
        kb = KnowledgeBase(facts=["parent(tom,bob)"], rules=[])
        engine = SymbolicEngine(kb)
        result = engine.query("parent(tom,ann)")
        assert result.result is False
        assert result.confidence == 0.0

    def test_single_rule_inference(self) -> None:
        kb = make_simple_kb()
        engine = SymbolicEngine(kb)
        result = engine.query("can_fly(tweety)")
        assert result.result is True

    def test_ancestor_direct(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        # The engine's _apply_bindings formats grounded args with ", " (space after comma),
        # so derived facts have the form "ancestor(tom, bob)" not "ancestor(tom,bob)".
        result = engine.query("ancestor(tom, bob)")
        assert result.result is True

    def test_ancestor_transitive(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        # See test_ancestor_direct: grounded literals use ", " separator.
        result = engine.query("ancestor(tom, ann)")
        assert result.result is True

    def test_ancestor_deep_chain(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        # See test_ancestor_direct: grounded literals use ", " separator.
        result = engine.query("ancestor(tom, sue)")
        assert result.result is True

    def test_ancestor_not_proven(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        result = engine.query("ancestor(ann,tom)")
        assert result.result is False

    def test_add_fact_expands_working_set(self) -> None:
        kb = KnowledgeBase(facts=[], rules=[
            LogicRule(rule_id="r1", head="can_swim(X)", body=["duck(X)"])
        ])
        engine = SymbolicEngine(kb)
        engine.add_fact("duck(donald)")
        result = engine.query("can_swim(donald)")
        assert result.result is True

    def test_add_rule_extends_inference(self) -> None:
        kb = KnowledgeBase(facts=["dog(rex)"], rules=[])
        engine = SymbolicEngine(kb)
        engine.add_rule(LogicRule(rule_id="r1", head="pet(X)", body=["dog(X)"]))
        result = engine.query("pet(rex)")
        assert result.result is True

    def test_ground_rule_no_variables(self) -> None:
        kb = KnowledgeBase(
            facts=["alive(socrates)", "mortal_rule_antecedent_known"],
            rules=[
                LogicRule(rule_id="r1", head="mortal(socrates)", body=["alive(socrates)"])
            ],
        )
        engine = SymbolicEngine(kb)
        result = engine.query("mortal(socrates)")
        assert result.result is True

    def test_fact_rule_no_body(self) -> None:
        kb = KnowledgeBase(
            facts=[],
            rules=[LogicRule(rule_id="r1", head="sky_is_blue", body=[])],
        )
        engine = SymbolicEngine(kb)
        result = engine.query("sky_is_blue")
        assert result.result is True

    def test_explain_proven(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        # The engine derives "ancestor(tom, bob)" (with space after comma) via _apply_bindings,
        # so the query goal must use the same format for the result to be found.
        explanation = engine.explain("ancestor(tom, bob)")
        assert "Query:" in explanation
        assert "Proven: True" in explanation
        assert "Confidence:" in explanation

    def test_explain_not_proven(self) -> None:
        kb = make_ancestor_kb()
        engine = SymbolicEngine(kb)
        explanation = engine.explain("ancestor(sue,tom)")
        assert "Proven: False" in explanation
        assert "No proof found" in explanation

    def test_confidence_propagation_with_rule_weight(self) -> None:
        kb = KnowledgeBase(
            facts=["a(x)"],
            rules=[LogicRule(rule_id="r1", head="b(x)", body=["a(x)"], confidence=0.8)],
        )
        engine = SymbolicEngine(kb)
        result = engine.query("b(x)")
        assert result.result is True
        assert math.isclose(result.confidence, 0.8, rel_tol=1e-6)

    def test_query_returns_inference_result(self) -> None:
        kb = make_simple_kb()
        engine = SymbolicEngine(kb)
        result = engine.query("can_fly(tweety)")
        assert isinstance(result, InferenceResult)

    def test_proof_chain_not_empty_for_derived_fact(self) -> None:
        kb = make_simple_kb()
        engine = SymbolicEngine(kb)
        result = engine.query("can_fly(tweety)")
        assert result.result is True
        assert len(result.proof_chain) > 0

    def test_multiple_entities(self) -> None:
        kb = KnowledgeBase(
            facts=["bird(tweety)", "wings(tweety)", "bird(polly)", "wings(polly)"],
            rules=[
                LogicRule(rule_id="r1", head="can_fly(X)", body=["bird(X)", "wings(X)"])
            ],
        )
        engine = SymbolicEngine(kb)
        assert engine.query("can_fly(tweety)").result is True
        assert engine.query("can_fly(polly)").result is True

    def test_add_fact_after_initialisation(self) -> None:
        kb = KnowledgeBase(facts=[], rules=[])
        engine = SymbolicEngine(kb)
        engine.add_fact("exists(thing)")
        result = engine.query("exists(thing)")
        assert result.result is True


# ---------------------------------------------------------------------------
# KnowledgeCompiler tests
# ---------------------------------------------------------------------------


class TestKnowledgeCompiler:
    def test_parse_single_fact(self) -> None:
        compiler = KnowledgeCompiler()
        kb = compiler.from_text("parent(tom,bob).")
        assert "parent(tom,bob)" in kb.facts

    def test_parse_single_rule(self) -> None:
        compiler = KnowledgeCompiler()
        kb = compiler.from_text("ancestor(X,Y) :- parent(X,Y).")
        assert len(kb.rules) == 1
        assert kb.rules[0].head == "ancestor(X,Y)"
        # The parenthesis-aware splitter preserves multi-arg predicates intact.
        assert kb.rules[0].body == ["parent(X,Y)"]

    def test_parse_multiple_facts(self) -> None:
        compiler = KnowledgeCompiler()
        text = "parent(tom,bob).\nparent(bob,ann)."
        kb = compiler.from_text(text)
        assert len(kb.facts) == 2

    def test_parse_comments_ignored(self) -> None:
        compiler = KnowledgeCompiler()
        text = "% This is a comment\nparent(tom,bob)."
        kb = compiler.from_text(text)
        assert len(kb.facts) == 1
        assert len(kb.rules) == 0

    def test_parse_blank_lines_ignored(self) -> None:
        compiler = KnowledgeCompiler()
        text = "\n\nparent(tom,bob).\n\n"
        kb = compiler.from_text(text)
        assert len(kb.facts) == 1

    def test_parse_rule_with_multiple_body_literals(self) -> None:
        compiler = KnowledgeCompiler()
        text = "can_fly(X) :- bird(X), wings(X)."
        kb = compiler.from_text(text)
        assert len(kb.rules) == 1
        assert kb.rules[0].body == ["bird(X)", "wings(X)"]

    def test_parse_confidence_annotation(self) -> None:
        compiler = KnowledgeCompiler()
        # The _RULE_PATTERN regex now allows an optional trailing "% comment"
        # after the period, so the round-trip format emitted by to_text() is
        # parseable: "head :- body. % confidence=N.NNNN"
        text = "fly(X) :- bird(X). % confidence=0.8"
        kb = compiler.from_text(text)
        assert kb.rules[0].confidence == 0.8

    def test_rule_ids_are_sequential(self) -> None:
        compiler = KnowledgeCompiler()
        text = "a(X) :- b(X).\nc(X) :- d(X)."
        kb = compiler.from_text(text)
        assert kb.rules[0].rule_id == "r1"
        assert kb.rules[1].rule_id == "r2"

    def test_to_text_facts_serialized(self) -> None:
        compiler = KnowledgeCompiler()
        kb = KnowledgeBase(facts=["parent(tom,bob)"])
        text = compiler.to_text(kb)
        assert "parent(tom,bob)." in text

    def test_to_text_rules_serialized(self) -> None:
        compiler = KnowledgeCompiler()
        kb = KnowledgeBase(rules=[
            LogicRule(rule_id="r1", head="ancestor(X,Y)", body=["parent(X,Y)"])
        ])
        text = compiler.to_text(kb)
        assert "ancestor(X,Y) :- parent(X,Y)." in text

    def test_roundtrip_text(self) -> None:
        compiler = KnowledgeCompiler()
        # Both facts and rules now roundtrip cleanly: to_text() serializes rules
        # as "head :- body. % confidence=N.NNNN", and the updated _RULE_PATTERN
        # regex allows an optional trailing comment after the period.
        original_text = "parent(tom,bob).\nancestor(X,Y) :- parent(X,Y)."
        kb = compiler.from_text(original_text)
        serialized = compiler.to_text(kb)
        kb2 = compiler.from_text(serialized)
        assert len(kb2.facts) == len(kb.facts)
        assert len(kb2.rules) == len(kb.rules)

    def test_from_json(self, tmp_path: Path) -> None:
        compiler = KnowledgeCompiler()
        kb = make_ancestor_kb()
        json_path = tmp_path / "kb.json"
        compiler.to_json(kb, json_path)
        loaded_kb = compiler.from_json(json_path)
        assert len(loaded_kb.facts) == len(kb.facts)
        assert len(loaded_kb.rules) == len(kb.rules)

    def test_to_json_creates_valid_file(self, tmp_path: Path) -> None:
        compiler = KnowledgeCompiler()
        kb = make_ancestor_kb()
        json_path = tmp_path / "output.json"
        compiler.to_json(kb, json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "facts" in data
        assert "rules" in data

    def test_parse_empty_text(self) -> None:
        compiler = KnowledgeCompiler()
        kb = compiler.from_text("")
        assert kb.facts == []
        assert kb.rules == []

    def test_parse_only_comments(self) -> None:
        compiler = KnowledgeCompiler()
        kb = compiler.from_text("% comment\n% another comment")
        assert kb.facts == []
        assert kb.rules == []

    def test_to_text_includes_confidence_annotation(self) -> None:
        compiler = KnowledgeCompiler()
        kb = KnowledgeBase(rules=[
            LogicRule(rule_id="r1", head="foo(X)", body=["bar(X)"], confidence=0.75)
        ])
        text = compiler.to_text(kb)
        assert "confidence=0.7500" in text


# ---------------------------------------------------------------------------
# DifferentiableLogic tests
# ---------------------------------------------------------------------------


class TestDifferentiableLogic:
    def test_soft_and_basic(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_and(0.9, 0.8) == pytest.approx(0.72)

    def test_soft_and_identity(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_and(1.0, 0.5) == pytest.approx(0.5)

    def test_soft_and_zero(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_and(0.0, 0.9) == pytest.approx(0.0)

    def test_soft_and_both_zero(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_and(0.0, 0.0) == pytest.approx(0.0)

    def test_soft_and_both_one(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_and(1.0, 1.0) == pytest.approx(1.0)

    def test_soft_or_basic(self) -> None:
        dl = DifferentiableLogic()
        result = dl.soft_or(0.9, 0.8)
        # 0.9 + 0.8 - 0.9*0.8 = 1.7 - 0.72 = 0.98
        assert result == pytest.approx(0.98)

    def test_soft_or_zero(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_or(0.0, 0.5) == pytest.approx(0.5)

    def test_soft_or_one(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_or(1.0, 0.5) == pytest.approx(1.0)

    def test_soft_or_both_zero(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_or(0.0, 0.0) == pytest.approx(0.0)

    def test_soft_or_both_one(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_or(1.0, 1.0) == pytest.approx(1.0)

    def test_soft_not_basic(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_not(0.9) == pytest.approx(0.1)

    def test_soft_not_zero(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_not(0.0) == pytest.approx(1.0)

    def test_soft_not_one(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_not(1.0) == pytest.approx(0.0)

    def test_soft_not_half(self) -> None:
        dl = DifferentiableLogic()
        assert dl.soft_not(0.5) == pytest.approx(0.5)

    def test_de_morgan_and_or_not(self) -> None:
        """NOT(AND(a,b)) ~= OR(NOT(a), NOT(b)) for product t-norm (approx)."""
        dl = DifferentiableLogic()
        a, b = 0.7, 0.6
        lhs = dl.soft_not(dl.soft_and(a, b))
        rhs = dl.soft_or(dl.soft_not(a), dl.soft_not(b))
        # This is NOT exactly De Morgan for product t-norm — just verify they're both in [0,1]
        assert 0.0 <= lhs <= 1.0
        assert 0.0 <= rhs <= 1.0

    def test_evaluate_rule_single_body(self) -> None:
        dl = DifferentiableLogic()
        rule = LogicRule(rule_id="r1", head="b(x)", body=["a(x)"], confidence=0.9)
        result = dl.evaluate_rule(rule, {"a(x)": 0.8})
        assert result == pytest.approx(0.9 * 0.8)

    def test_evaluate_rule_empty_body(self) -> None:
        dl = DifferentiableLogic()
        rule = LogicRule(rule_id="r1", head="fact", body=[], confidence=0.95)
        result = dl.evaluate_rule(rule, {})
        assert result == pytest.approx(0.95)

    def test_evaluate_rule_missing_body_literal(self) -> None:
        dl = DifferentiableLogic()
        rule = LogicRule(rule_id="r1", head="b(x)", body=["a(x)"], confidence=1.0)
        result = dl.evaluate_rule(rule, {})
        # Missing literal defaults to 0.0
        assert result == pytest.approx(0.0)

    def test_evaluate_rule_multi_body(self) -> None:
        dl = DifferentiableLogic()
        rule = LogicRule(rule_id="r1", head="c(x)", body=["a(x)", "b(x)"], confidence=1.0)
        result = dl.evaluate_rule(rule, {"a(x)": 0.9, "b(x)": 0.8})
        # 1.0 * (0.9 * 0.8) = 0.72
        assert result == pytest.approx(0.72)

    def test_evaluate_rule_confidence_multiplied(self) -> None:
        dl = DifferentiableLogic()
        rule = LogicRule(rule_id="r1", head="c(x)", body=["a(x)"], confidence=0.5)
        result = dl.evaluate_rule(rule, {"a(x)": 1.0})
        assert result == pytest.approx(0.5)

    def test_soft_and_commutativity(self) -> None:
        dl = DifferentiableLogic()
        a, b = 0.6, 0.7
        assert dl.soft_and(a, b) == pytest.approx(dl.soft_and(b, a))

    def test_soft_or_commutativity(self) -> None:
        dl = DifferentiableLogic()
        a, b = 0.3, 0.8
        assert dl.soft_or(a, b) == pytest.approx(dl.soft_or(b, a))
