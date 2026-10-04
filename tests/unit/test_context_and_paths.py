import pytest

from agentic_code_assistant.context import count_tokens, trim_text
from agentic_code_assistant.paths import PathEscapeError, is_sensitive_file, resolve_in_root


def test_short_text_is_untouched():
    assert trim_text("hello", 100) == "hello"


def test_long_text_is_trimmed_and_flagged():
    out = trim_text("word " * 5000, 100)
    assert "truncated" in out
    assert count_tokens(out) < 200


def test_resolve_allows_inside_and_blocks_escape(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert resolve_in_root("a.txt", tmp_path) == (tmp_path / "a.txt").resolve()
    for bad in ["../outside.txt", "/etc/passwd", "sub/../../x"]:
        with pytest.raises(PathEscapeError):
            resolve_in_root(bad, tmp_path)


def test_symlink_escape_is_blocked(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PathEscapeError):
        resolve_in_root("link/secret.txt", root)


def test_sensitive_files():
    assert is_sensitive_file(".env") and is_sensitive_file("cfg/.env.local")
    assert is_sensitive_file("server.pem") and is_sensitive_file("id_rsa")
    assert not is_sensitive_file(".env.example") and not is_sensitive_file("main.py")
