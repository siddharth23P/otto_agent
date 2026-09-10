"""Coverage for agent/memory/tokens.py's count_tokens() -- the approximate
token counter agent/memory/queue.py budgets X and Y against. Both the real
tiktoken path and the chars/4 fallback (used when cl100k_base can't be
loaded -- no network, e.g.) need to behave sanely, since queue.py's own
budget math depends on this never crashing.
"""
import agent.memory.tokens as tok


def test_empty_text_is_zero_tokens():
    assert tok.count_tokens("") == 0
    assert tok.count_tokens(None) == 0


def test_fallback_heuristic_used_when_no_encoding_available(monkeypatch):
    monkeypatch.setattr(tok, "_get_encoding", lambda: None)

    # 12 chars / 4 chars-per-token = 3
    assert tok.count_tokens("hello world!") == 3


def test_fallback_never_zero_for_nonempty_text(monkeypatch):
    monkeypatch.setattr(tok, "_get_encoding", lambda: None)

    assert tok.count_tokens("hi") >= 1


def test_uses_real_encoding_when_available(monkeypatch):
    class _FakeEncoding:
        def encode(self, text):
            # one "token" per word, deterministic and easy to assert on
            return text.split()

    monkeypatch.setattr(tok, "_get_encoding", lambda: _FakeEncoding())

    assert tok.count_tokens("a b c d") == 4


def test_get_encoding_only_attempts_the_real_load_once(monkeypatch):
    tok._encoding = None
    tok._encoding_load_attempted = False

    calls = []

    class _BrokenTiktoken:
        @staticmethod
        def get_encoding(name):
            calls.append(name)
            raise RuntimeError("no network")

    monkeypatch.setitem(__import__("sys").modules, "tiktoken", _BrokenTiktoken())

    first = tok._get_encoding()
    second = tok._get_encoding()

    assert first is None
    assert second is None
    assert len(calls) == 1
