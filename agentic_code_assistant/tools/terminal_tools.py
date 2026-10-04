"""A constrained shell tool.

The agent is for *understanding* a codebase, so it only gets a short allowlist of programs
(read files, grep, read-only git, run tests). Design choices:

* No shell (`shell=False`): the command is split into an argument list, so `;`, `&&`,
  `|`, `>` and `$(...)` have no special meaning and cannot chain extra commands.
* Allowlist instead of blocklist: a blocklist can never list every dangerous command.
* Paths in arguments may not point outside the project root.
* Secrets (API keys, tokens) are stripped from the child process environment.
* Timeout + output trimming keep a runaway command from stalling or flooding the loop.

This is defence in depth, not a sandbox: `python script.py` runs whatever is in the script.
Only point the assistant at projects you trust.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from agentic_code_assistant.config import get_settings
from agentic_code_assistant.context import trim_text
from agentic_code_assistant.paths import PathEscapeError, is_sensitive_file, resolve_in_root
from agentic_code_assistant.tracing import observe

ALLOWED_PROGRAMS = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "git", "python", "python3", "pytest", "ruff"}
)
READONLY_GIT_SUBCOMMANDS = frozenset(
    {"status", "log", "diff", "show", "branch", "ls-files", "grep", "blame", "rev-parse"}
)
SHELL_OPERATORS = frozenset({"|", "||", "&", "&&", ";", ">", ">>", "<", "2>", "2>&1"})
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class CommandRejected(ValueError):
    """The command was refused before running; the message is shown to the agent."""


def parse_command(command: str, project_root: Path) -> list[str]:
    """Validate `command` and return its argv, or raise CommandRejected."""
    try:
        argv = shlex.split(command, posix=(os.name != "nt"))
    except ValueError as exc:
        raise CommandRejected(f"Could not parse the command: {exc}") from exc
    if not argv:
        raise CommandRejected("Empty command.")

    if any(arg in SHELL_OPERATORS for arg in argv):
        raise CommandRejected(
            "Shell operators (| && ; > <) are not supported. Run one simple command at a time."
        )

    program = argv[0].lower()
    if program not in ALLOWED_PROGRAMS:
        raise CommandRejected(
            f"'{argv[0]}' is not allowed. Allowed programs: {', '.join(sorted(ALLOWED_PROGRAMS))}."
        )
    if program == "git":
        sub = next((a for a in argv[1:] if not a.startswith("-")), "")
        if sub not in READONLY_GIT_SUBCOMMANDS:
            raise CommandRejected(
                f"Only read-only git commands are allowed: {', '.join(sorted(READONLY_GIT_SUBCOMMANDS))}."
            )
    if program in {"python", "python3"} and "-c" in argv[1:]:
        raise CommandRejected("`python -c` is not allowed. Run a file or use `python -m pytest`.")

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        if _points_outside(arg, project_root):
            raise CommandRejected(f"Argument points outside the project root: {arg}")
        if is_sensitive_file(arg):
            raise CommandRejected(f"Refusing to touch {arg}: it may contain secrets.")
    return argv


def _points_outside(arg: str, project_root: Path) -> bool:
    path = Path(arg)
    if ".." in path.parts:
        return True
    if path.is_absolute():
        try:
            path.resolve().relative_to(project_root.resolve())
        except ValueError:
            return True
    return False


def _clean_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if not any(hint in k.upper() for hint in SECRET_HINTS)
    }


def run_argv(argv: list[str], cwd: Path, timeout: int, max_tokens: int) -> str:
    try:
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, errors="replace",
            timeout=timeout, env=_clean_env(), stdin=subprocess.DEVNULL, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {' '.join(argv)}"
    except FileNotFoundError:
        return f"Program not found on this machine: {argv[0]}. Try read_file / list_directory instead."
    except OSError as exc:
        return f"Could not run the command: {exc}"

    output = done.stdout
    if done.stderr:
        output += f"\n[stderr]\n{done.stderr}"
    return f"exit_code={done.returncode}\n" + trim_text(output.strip() or "(no output)", max_tokens)


def build_terminal_tool(project_root: Path):
    settings = get_settings()

    @tool
    @observe(type="tool", name="run_command")
    def run_command(command: str, directory: str = ".") -> str:
        """Run ONE simple command inside the project (optionally in a sub-directory).

        Allowed programs: ls, cat, head, tail, wc, grep, read-only git, python, pytest, ruff.
        No pipes, redirects or chaining. Use it to run tests, grep, or check git history.
        """
        try:
            cwd = resolve_in_root(directory, project_root)
            if not cwd.is_dir():
                return f"Not a directory: {directory}"
            argv = parse_command(command, project_root)
        except (PathEscapeError, CommandRejected) as exc:
            return f"Rejected: {exc}"
        return run_argv(
            argv, cwd, settings.terminal.timeout_seconds, settings.agent.max_tool_output_tokens
        )

    return run_command
