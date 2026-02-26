"""Pydantic v2 models for the neuro-symbolic reasoning engine."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LogicRule(BaseModel):
    """A single Horn-clause style logic rule with a confidence weight.

    Attributes:
        rule_id: Unique identifier for the rule.
        head: The conclusion literal produced when the body is satisfied.
        body: List of premise literals that must all be satisfied.
        confidence: Probability-like weight in [0, 1] representing rule strength.

    Example:
        >>> rule = LogicRule(rule_id="r1", head="ancestor(X,Z)",
        ...                  body=["parent(X,Y)", "ancestor(Y,Z)"],
        ...                  confidence=0.95)
    """

    rule_id: str = Field(..., min_length=1)
    head: str = Field(..., min_length=1)
    body: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("head", "rule_id")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace from string fields."""
        return value.strip()

    @field_validator("body")
    @classmethod
    def strip_body_elements(cls, value: list[str]) -> list[str]:
        """Strip whitespace from each body literal."""
        return [item.strip() for item in value]


class KnowledgeBase(BaseModel):
    """A collection of rules and ground facts forming a logic program.

    Attributes:
        rules: Ordered list of LogicRule objects.
        facts: Ground-truth literals asserted without proof.

    Example:
        >>> kb = KnowledgeBase(
        ...     facts=["parent(tom, bob)", "parent(bob, ann)"],
        ...     rules=[LogicRule(rule_id="r1", head="ancestor(X,Y)",
        ...                      body=["parent(X,Y)"])]
        ... )
    """

    rules: list[LogicRule] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    def add_rule(self, rule: LogicRule) -> None:
        """Append a rule to the knowledge base."""
        self.rules.append(rule)

    def add_fact(self, fact: str) -> None:
        """Assert a new ground fact."""
        self.facts.append(fact.strip())


class InferenceResult(BaseModel):
    """The outcome of a single query against a knowledge base.

    Attributes:
        query: The goal literal that was queried.
        result: Whether the query was proven true.
        proof_chain: Ordered list of literals visited during proof.
        confidence: Product of rule confidences along the proof chain.

    Example:
        >>> res = InferenceResult(query="ancestor(tom,ann)", result=True,
        ...                       proof_chain=["parent(tom,bob)", "parent(bob,ann)"],
        ...                       confidence=0.9)
    """

    query: str
    result: bool
    proof_chain: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
