"""Split source files into chunks that each hold one meaningful unit of code.

Why not just cut every N lines? A fixed window slices functions in half, so the embedding
of a chunk mixes unrelated code. For Python we use the `ast` module to cut along real
boundaries (functions, classes, methods). Other languages fall back to overlapping
line windows. Every chunk carries a `# file / symbol` header so the embedding also
"knows" where the code lives, which helps queries like "how does ExpenseStore.save work?".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int
    symbol: str  # e.g. "ExpenseStore.save"; "" when unknown
    code: str

    @property
    def text(self) -> str:
        """The string that is embedded and stored (header + code)."""
        label = f" :: {self.symbol}" if self.symbol else ""
        return f"# {self.path}{label}\n{self.code}"


def chunk_source(
    source: str, *, path: str, chunk_lines: int = 60, overlap: int = 10
) -> list[CodeChunk]:
    lines = source.splitlines()
    if not lines:
        return []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None  # broken file: fall back to plain windows
        if tree is not None and tree.body:
            spans = _python_spans(tree.body, 1, len(lines), "", chunk_lines)
            return _spans_to_chunks(spans, lines, path, chunk_lines, overlap)
    return _window(lines, path=path, symbol="", chunk_lines=chunk_lines, overlap=overlap)


# --------------------------------------------------------------------------- python (ast)

_Span = tuple[int, int, str]  # (start_line, end_line, symbol)


def _node_start(node: ast.stmt) -> int:
    """First line of a statement, including its decorators."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _python_spans(
    body: list[ast.stmt], lo: int, hi: int, prefix: str, chunk_lines: int
) -> list[_Span]:
    """Partition lines lo..hi into spans that follow the AST boundaries of `body`.

    Each statement owns the lines from its own start up to the start of the next one, so
    comments and blank lines stay attached to a neighbour. Consecutive plain statements
    (imports, constants) are merged into one span. Big classes are split into their methods.
    """
    starts = [lo if i == 0 else _node_start(node) for i, node in enumerate(body)]
    spans: list[_Span] = []
    pending: list[int] | None = None  # [start, end] of a run of plain statements

    def flush() -> None:
        nonlocal pending
        if pending:
            label = f"{prefix[:-1]} (class header)" if prefix else "(module level)"
            spans.append((pending[0], pending[1], label))
            pending = None

    for i, node in enumerate(body):
        start = starts[i]
        end = starts[i + 1] - 1 if i + 1 < len(body) else hi
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if not is_def:
            pending = [pending[0], end] if pending else [start, end]
            continue
        flush()
        symbol = f"{prefix}{node.name}"
        if isinstance(node, ast.ClassDef) and end - start + 1 > chunk_lines and node.body:
            spans.extend(_python_spans(node.body, start, end, f"{symbol}.", chunk_lines))
        else:
            spans.append((start, end, symbol))
    flush()
    return spans


def _spans_to_chunks(
    spans: list[_Span], lines: list[str], path: str, chunk_lines: int, overlap: int
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for start, end, symbol in spans:
        piece = lines[start - 1 : end]
        if not "".join(piece).strip():
            continue
        if len(piece) <= chunk_lines:
            chunks.append(
                CodeChunk(path, start, end, symbol, "\n".join(piece).strip("\n"))
            )
        else:  # a single huge function: split with overlap
            chunks.extend(
                _window(
                    piece, path=path, symbol=symbol, chunk_lines=chunk_lines,
                    overlap=overlap, line_offset=start - 1,
                )
            )
    return chunks


# --------------------------------------------------------------------------- generic fallback


def _window(
    lines: list[str],
    *,
    path: str,
    symbol: str,
    chunk_lines: int,
    overlap: int,
    line_offset: int = 0,
) -> list[CodeChunk]:
    """Overlapping fixed-size line windows (used for non-Python files)."""
    step = max(chunk_lines - overlap, 1)
    chunks: list[CodeChunk] = []
    for i in range(0, len(lines), step):
        piece = lines[i : i + chunk_lines]
        code = "\n".join(piece).strip("\n")
        if code.strip():
            start = line_offset + i + 1
            chunks.append(CodeChunk(path, start, start + len(piece) - 1, symbol, code))
        if i + chunk_lines >= len(lines):
            break  # last window already reached the end of the file
    return chunks
