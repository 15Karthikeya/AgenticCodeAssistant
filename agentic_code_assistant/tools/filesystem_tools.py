"""Read-only filesystem tools, confined to the project root."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from agentic_code_assistant.config import get_settings
from agentic_code_assistant.context import trim_text
from agentic_code_assistant.paths import (
    PathEscapeError,
    is_ignored_dir,
    is_sensitive_file,
    resolve_in_root,
)
from agentic_code_assistant.tracing import observe


def build_filesystem_tools(project_root: Path):
    max_tokens = get_settings().agent.max_tool_output_tokens

    @tool
    @observe(type="tool", name="read_file")
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a text file from the project (path relative to the project root).

        Optionally pass start_line / end_line (1-based, inclusive) to read just a slice,
        which is how to read the rest of a large file. end_line=0 means "to the end".
        """
        try:
            target = resolve_in_root(path, project_root)
            if is_sensitive_file(target):
                return f"Refusing to read {path}: it may contain secrets."
            if not target.exists():
                return f"File not found: {path}"
            if not target.is_file():
                return f"Not a file: {path} (use list_directory)"
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except PathEscapeError as exc:
            return str(exc)
        except OSError as exc:
            return f"Could not read {path}: {exc}"

        first = max(start_line, 1)
        last = end_line if end_line > 0 else len(lines)
        numbered = "\n".join(
            f"{n}: {line}" for n, line in enumerate(lines[first - 1 : last], start=first)
        )
        return trim_text(numbered or "(empty file)", max_tokens)

    @tool
    @observe(type="tool", name="list_directory")
    def list_directory(path: str = ".") -> str:
        """List the files and folders in a project directory (path relative to the project root)."""
        try:
            target = resolve_in_root(path, project_root)
            if not target.exists():
                return f"Directory not found: {path}"
            if not target.is_dir():
                return f"Not a directory: {path} (use read_file)"
            entries = [
                p.name + ("/" if p.is_dir() else "")
                for p in sorted(target.iterdir())
                if not (p.is_dir() and is_ignored_dir(p.name))
            ]
        except PathEscapeError as exc:
            return str(exc)
        except OSError as exc:
            return f"Could not list {path}: {exc}"
        return trim_text("\n".join(entries) or "(empty directory)", max_tokens)

    return [read_file, list_directory]
