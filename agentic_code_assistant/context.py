"""Keep tool observations small enough for the model's context window.

Every tool result goes back into the conversation, so one huge `read_file` or test run
could crowd out everything else. We count *tokens* (not characters) because the context
window is measured in tokens.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    """tiktoken needs to download its vocabulary once; fall back to an estimate if offline."""
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")  # tokenizer family of the gpt-4o models
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return (len(text) + 3) // 4  # rough rule of thumb: 1 token ~ 4 characters
    return len(enc.encode(text, disallowed_special=()))


def trim_text(text: str, max_tokens: int) -> str:
    """Return `text` unchanged if it fits, otherwise keep the first `max_tokens` tokens."""
    total = count_tokens(text)
    if total <= max_tokens:
        return text
    enc = _encoder()
    if enc is None:
        kept = text[: max_tokens * 4]
    else:
        kept = enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    return f"{kept}\n\n[truncated: showing the first ~{max_tokens} of {total} tokens]"
