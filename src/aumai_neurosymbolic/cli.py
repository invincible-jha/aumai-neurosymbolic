"""CLI entry point for aumai-neurosymbolic.

Commands:
    query  -- prove a goal against a JSON knowledge base
    compile -- convert Prolog-like text rules to JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .core import KnowledgeCompiler, SymbolicEngine
from .models import KnowledgeBase


@click.group()
@click.version_option()
def main() -> None:
    """AumAI Neurosymbolic -- differentiable logic reasoning CLI."""


@main.command("query")
@click.option(
    "--kb",
    "kb_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a JSON knowledge base file.",
)
@click.option(
    "--goal",
    required=True,
    type=str,
    help='Goal literal to prove, e.g. "ancestor(tom,ann)".',
)
@click.option("--explain", is_flag=True, default=False, help="Print full proof trace.")
def query_command(kb_path: Path, goal: str, explain: bool) -> None:
    """Prove GOAL against the knowledge base at KB_PATH.

    Example:

        aumai-neurosymbolic query --kb rules.json --goal "ancestor(tom,ann)"
    """
    try:
        data = json.loads(kb_path.read_text(encoding="utf-8"))
        kb = KnowledgeBase.model_validate(data)
    except Exception as exc:
        click.echo(f"ERROR loading knowledge base: {exc}", err=True)
        sys.exit(1)

    engine = SymbolicEngine(kb)

    if explain:
        click.echo(engine.explain(goal))
    else:
        result = engine.query(goal)
        status = "PROVEN" if result.result else "NOT PROVEN"
        click.echo(f"{status}: {result.query}  (confidence={result.confidence:.4f})")
        if result.proof_chain:
            click.echo("Proof chain:")
            for step in result.proof_chain:
                click.echo(f"  {step}")


@main.command("compile")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a Prolog-like rules text file.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Destination JSON path. Defaults to <input>.json.",
)
def compile_command(input_path: Path, output_path: Path | None) -> None:
    """Compile a Prolog-like rules file to a JSON knowledge base.

    Example:

        aumai-neurosymbolic compile --input rules.txt --output rules.json
    """
    text = input_path.read_text(encoding="utf-8")
    compiler = KnowledgeCompiler()
    kb = compiler.from_text(text)

    dest = output_path or input_path.with_suffix(".json")
    compiler.to_json(kb, dest)

    click.echo(
        f"Compiled {len(kb.rules)} rule(s) and {len(kb.facts)} fact(s) to {dest}"
    )


if __name__ == "__main__":
    main()
