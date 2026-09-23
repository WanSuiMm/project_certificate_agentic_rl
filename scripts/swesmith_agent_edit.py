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


def replace_callable(source: str, path: list[str], completion: str) -> str:
    replacement = completion.strip()
    if replacement.startswith("```"):
        replacement = "\n".join(replacement.splitlines()[1:]).strip()
        if replacement.endswith("```"):
            replacement = replacement[:-3].strip()
    try:
        tree = ast.parse(replacement)
        candidates = [node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == path[-1]]
        if len(candidates) != 1:
            raise InvalidEdit("output must contain exactly one target synchronous def")
        candidate = candidates[0]
        replacement_lines = replacement.splitlines()
        candidate_first = min([candidate.lineno, *(n.lineno for n in candidate.decorator_list)])
        replacement = "\n".join(replacement_lines[candidate_first - 1:candidate.end_lineno])
        original = _node(source, path)
        if candidate.name != path[-1] or ast.dump(candidate.args) != ast.dump(original.args):
            raise InvalidEdit("function name or signature changed")
        lines = source.splitlines(keepends=True)
        first = min([original.lineno, *(n.lineno for n in original.decorator_list)])
        indent = len(lines[first - 1]) - len(lines[first - 1].lstrip(" "))
        if indent and len(path) != 2:
            raise InvalidEdit("unexpected top-level indentation")
        block = textwrap.indent(replacement.rstrip() + "\n", " " * indent)
        edited = "".join(lines[:first - 1]) + block + "".join(lines[original.end_lineno:])
        ast.parse(edited)
        if ast.dump(_node(edited, path).args) != ast.dump(original.args):
            raise InvalidEdit("post-edit signature changed")
        return edited
    except SyntaxError as exc:
        raise InvalidEdit(f"invalid Python syntax: {exc.msg}") from exc
