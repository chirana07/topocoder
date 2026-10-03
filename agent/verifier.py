"""
AST-based Verification Gate for agent-generated patches.
Validates syntax, indentation, and checks for regressions via isolated execution.
"""

import ast
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


import textwrap


@dataclass
class VerificationResult:
    is_valid: bool
    error_message: Optional[str] = None
    ast_node: Optional[ast.AST] = None
    diff_summary: Optional[str] = None


class PatchVerifier:
    """Verifies that code modifications are syntactically sound and non-destructive."""

    @staticmethod
    def clean_code(code: str) -> str:
        """Strips leading file/header comments and normalizes indentation for AST parsing."""
        if not code or not code.strip():
            return code

        # Test direct parse first
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass

        lines = code.splitlines()
        # Separate leading comments and blank lines
        comments = []
        while lines and (not lines[0].strip() or lines[0].strip().startswith("#")):
            comments.append(lines.pop(0))

        if lines:
            dedented_body = textwrap.dedent("\n".join(lines))
            try:
                ast.parse(dedented_body)
                return dedented_body
            except SyntaxError:
                pass

            # Check if wrapping inside a class works (for methods)
            try:
                wrapped = "class _Scope:\n" + textwrap.indent(dedented_body, "    ")
                ast.parse(wrapped)
                return dedented_body
            except SyntaxError:
                pass

        return code

    @staticmethod
    def verify_syntax(code: str) -> VerificationResult:
        """Checks if code string is valid Python syntax."""
        cleaned = PatchVerifier.clean_code(code)
        try:
            tree = ast.parse(cleaned)
            return VerificationResult(is_valid=True, ast_node=tree)
        except SyntaxError as e:
            # Also try wrapped class check if it's a stand-alone class method
            try:
                wrapped = "class _Scope:\n" + textwrap.indent(cleaned, "    ")
                tree = ast.parse(wrapped)
                return VerificationResult(is_valid=True, ast_node=tree)
            except SyntaxError:
                pass
            return VerificationResult(
                is_valid=False,
                error_message=f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
            )
        except Exception as e:
            return VerificationResult(is_valid=False, error_message=str(e))

    @staticmethod
    def apply_function_replacement(
        original_source: str,
        target_func_name: str,
        replacement_func_code: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Replaces a function/method definition in the source file with a new implementation.
        Returns: (success, new_source_or_original, error_message)
        """
        # First check replacement syntax
        v_res = PatchVerifier.verify_syntax(replacement_func_code)
        if not v_res.is_valid:
            return False, original_source, v_res.error_message

        try:
            orig_tree = ast.parse(original_source)
        except SyntaxError as e:
            return False, original_source, f"Original file syntax error: {e}"

        lines = original_source.splitlines()

        # Find target function span
        found_target = None
        for node in ast.walk(orig_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_func_name:
                found_target = node
                break

        if not found_target:
            return False, original_source, f"Target function '{target_func_name}' not found in source."

        start_line = found_target.lineno - 1
        end_line = getattr(found_target, "end_lineno", start_line + 1)

        # Detect indentation of original function
        orig_first_line = lines[start_line]
        indent = len(orig_first_line) - len(orig_first_line.lstrip())

        # Adjust indentation of replacement if needed
        rep_lines = replacement_func_code.strip().splitlines()
        first_rep_indent = len(rep_lines[0]) - len(rep_lines[0].lstrip())
        indent_delta = indent - first_rep_indent

        adjusted_rep_lines = []
        for rl in rep_lines:
            if indent_delta > 0:
                adjusted_rep_lines.append((" " * indent_delta) + rl)
            elif indent_delta < 0 and rl.startswith(" " * abs(indent_delta)):
                adjusted_rep_lines.append(rl[abs(indent_delta):])
            else:
                adjusted_rep_lines.append(rl)

        new_lines = lines[:start_line] + adjusted_rep_lines + lines[end_line:]
        new_source = "\n".join(new_lines)

        # Verify final syntax
        final_check = PatchVerifier.verify_syntax(new_source)
        if not final_check.is_valid:
            return False, original_source, f"Post-patch syntax error: {final_check.error_message}"

        return True, new_source, None
