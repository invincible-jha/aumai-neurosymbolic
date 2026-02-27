"""Tests for aumai-neurosymbolic CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aumai_neurosymbolic.cli import main


def make_kb_json() -> dict:
    return {
        "facts": ["parent(tom,bob)", "parent(bob,ann)"],
        "rules": [
            {"rule_id": "r1", "head": "ancestor(X,Y)", "body": ["parent(X,Y)"], "confidence": 1.0},
            {"rule_id": "r2", "head": "ancestor(X,Z)", "body": ["parent(X,Y)", "ancestor(Y,Z)"], "confidence": 1.0},
        ],
    }


def make_rules_text() -> str:
    return "parent(tom,bob).\nancestor(X,Y) :- parent(X,Y)."


class TestCLIVersion:
    def test_cli_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0


class TestQueryCommand:
    def test_query_proven_goal(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("kb.json").write_text(json.dumps(make_kb_json()))
            result = runner.invoke(main, ["query", "--kb", "kb.json", "--goal", "ancestor(tom,bob)"])
            assert result.exit_code == 0
            assert "PROVEN" in result.output

    def test_query_not_proven(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("kb.json").write_text(json.dumps(make_kb_json()))
            result = runner.invoke(main, ["query", "--kb", "kb.json", "--goal", "ancestor(ann,tom)"])
            assert result.exit_code == 0
            assert "NOT PROVEN" in result.output

    def test_query_with_explain_flag(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("kb.json").write_text(json.dumps(make_kb_json()))
            result = runner.invoke(main, ["query", "--kb", "kb.json", "--goal", "ancestor(tom,bob)", "--explain"])
            assert result.exit_code == 0
            assert "Query:" in result.output

    def test_query_missing_kb_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["query", "--kb", "nonexistent.json", "--goal", "foo"])
        assert result.exit_code != 0

    def test_query_invalid_kb_json(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("bad.json").write_text("NOT VALID JSON")
            result = runner.invoke(main, ["query", "--kb", "bad.json", "--goal", "foo"])
            assert result.exit_code != 0


class TestCompileCommand:
    def test_compile_text_to_json(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.txt").write_text(make_rules_text())
            result = runner.invoke(main, ["compile", "--input", "rules.txt", "--output", "rules.json"])
            assert result.exit_code == 0
            assert Path("rules.json").exists()

    def test_compile_output_is_valid_json(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.txt").write_text(make_rules_text())
            runner.invoke(main, ["compile", "--input", "rules.txt", "--output", "out.json"])
            data = json.loads(Path("out.json").read_text())
            assert "facts" in data
            assert "rules" in data

    def test_compile_missing_input(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--input", "nonexistent.txt"])
        assert result.exit_code != 0

    def test_compile_shows_counts(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.txt").write_text(make_rules_text())
            result = runner.invoke(main, ["compile", "--input", "rules.txt", "--output", "out.json"])
            assert "rule" in result.output.lower()
            assert "fact" in result.output.lower()

    def test_compile_default_output_path(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.txt").write_text(make_rules_text())
            result = runner.invoke(main, ["compile", "--input", "rules.txt"])
            assert result.exit_code == 0
            assert Path("rules.json").exists()
