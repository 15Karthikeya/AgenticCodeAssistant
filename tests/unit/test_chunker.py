from agentic_code_assistant.rag.chunker import chunk_source

SRC = '''import os

X = 1


def a():
    return 1


@decorator
def b():
    return 2


class C:
    """doc"""

    def m1(self):
        return 1

    def m2(self):
        return 2
'''


def test_python_is_split_on_symbol_boundaries():
    chunks = chunk_source(SRC, path="m.py", chunk_lines=60)
    symbols = [c.symbol for c in chunks]
    assert symbols == ["(module level)", "a", "b", "C"]  # small class stays whole


def test_decorator_is_part_of_the_function_chunk():
    b = next(c for c in chunk_source(SRC, path="m.py") if c.symbol == "b")
    assert b.code.lstrip().startswith("@decorator")


def test_large_class_is_split_into_methods():
    chunks = chunk_source(SRC, path="m.py", chunk_lines=6)
    symbols = [c.symbol for c in chunks]
    assert "C (class header)" in symbols and "C.m1" in symbols and "C.m2" in symbols


def test_line_numbers_point_at_real_lines():
    lines = SRC.splitlines()
    for c in chunk_source(SRC, path="m.py", chunk_lines=6):
        assert c.code.splitlines()[0].strip() == lines[c.start_line - 1].strip()


def test_chunk_text_has_path_and_symbol_header():
    c = next(c for c in chunk_source(SRC, path="pkg/m.py") if c.symbol == "a")
    assert c.text.startswith("# pkg/m.py :: a\n")


def test_oversized_function_is_windowed_with_overlap():
    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    chunks = chunk_source(f"def big():\n{body}\n", path="m.py", chunk_lines=20, overlap=5)
    assert len(chunks) >= 3
    assert chunks[1].start_line < chunks[0].end_line  # overlap
    assert chunks[-1].end_line == 51  # reaches the end of the function


def test_syntax_error_falls_back_to_windows():
    assert chunk_source("def broken(:\n  pass\n", path="bad.py")


def test_non_python_uses_windows_and_empty_file_gives_nothing():
    text = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_source(text, path="notes.md", chunk_lines=40, overlap=10)
    assert [c.start_line for c in chunks] == [1, 31, 61]
    assert chunk_source("", path="a.py") == []
