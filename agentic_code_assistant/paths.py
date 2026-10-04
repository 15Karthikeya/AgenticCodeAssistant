"""Project-root helpers shared by the indexer and the agent tools."""

from __future__ import annotations

from pathlib import Path

# Directories that are never indexed or listed (dependencies, caches, VCS data, our own index).
IGNORE_DIRS = frozenset(
    {
        ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".aca", ".chroma",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".deepeval", ".idea", ".vscode",
        "dist", "build", "site-packages",
    }
)


def is_ignored_dir(name: str) -> bool:
    return name in IGNORE_DIRS or name.endswith(".egg-info")


def is_sensitive_file(path: Path | str) -> bool:
    """Files that usually hold secrets. Their contents would be sent to the LLM provider."""
    name = Path(path).name.lower()
    if name.startswith(".env") and name not in {".env.example", ".env.sample", ".env.template"}:
        return True
    return name in {"id_rsa", "id_ed25519", ".npmrc", ".pypirc"} or name.endswith((".pem", ".key"))


class PathEscapeError(ValueError):
    """Raised when a tool is asked to touch a path outside the project root."""


def project_root_from_cwd() -> Path:
    return Path.cwd().resolve()


def resolve_in_root(raw: str, project_root: Path) -> Path:
    """Resolve `raw` (relative or absolute) and guarantee it stays inside `project_root`.

    `.resolve()` collapses `..` and follows symlinks *before* the containment check,
    so `../../etc/passwd` and symlink tricks are both rejected.
    """
    root = project_root.resolve()
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathEscapeError(f"Refusing path outside the project root: {raw}") from exc
    return candidate
