"""Syntax-only edit operation for a real SWE-smith repository callable.

Candidate code is never imported or executed on the training host.
"""

from __future__ import annotations

import ast
import textwrap


class InvalidEdit(ValueError):
    pass


def _node(source: str, path: list[str]) -> ast.FunctionDef:
    tree = ast.parse(source)
    body = tree.body
    if len(path) == 2:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == path[0]]
        if len(classes) != 1:
            raise InvalidEdit("target class is not unique")
        body = classes[0].body
    elif len(path) != 1:
        raise InvalidEdit("target must be a function or direct method")
    matches = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == path[-1]]
    if len(matches) != 1:
        raise InvalidEdit("target function is not unique")
    return matches[0]


def normalize_replacement_body(completion: str) -> str:
    body = completion.strip("\n")
    if body.lstrip().startswith("```"):
        lines = body.lstrip().splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        body = "\n".join(lines)
    body = textwrap.dedent(body).strip("\n")
    if not body.strip():
        raise InvalidEdit("empty replacement body")
    return body


def current_callable_body(source: str, path: list[str]) -> str:
    node = _node(source, path)
    if node.body[0].lineno == node.lineno:
        return "\n".join(ast.get_source_segment(source, statement) or ""
                         for statement in node.body)
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[node.body[0].lineno - 1:node.end_lineno])).rstrip("\n")


def replace_callable_body(source: str, path: list[str], completion: str) -> str:
    body = normalize_replacement_body(completion)
    original = _node(source, path)
    lines = source.splitlines(keepends=True)
    first_statement = original.body[0]
    indent = " " * first_statement.col_offset
    if first_statement.lineno == original.lineno:
        header = lines[original.lineno - 1][:first_statement.col_offset].rstrip()
        if not header.endswith(":"):
            raise InvalidEdit("invalid inline function header")
        prefix = "".join(lines[:original.lineno - 1]) + header + "\n"
        indent = " " * (original.col_offset + 4)
    else:
        prefix = "".join(lines[:first_statement.lineno - 1])
    edited = (prefix + textwrap.indent(body.rstrip() + "\n", indent)
              + "".join(lines[original.end_lineno:]))
    try:
        ast.parse(edited)
    except SyntaxError as exc:
        raise InvalidEdit(f"invalid replacement body: {exc.msg}") from exc
    return edited
