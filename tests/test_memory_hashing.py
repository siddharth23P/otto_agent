"""Coverage for agent/memory/hashing.py's content_hash() -- the primary key
for agent/memory/store.py's permanent chunk table, and the identity that
lets agent/memory/queue.py's bullet hash_refs point at the same row across
however many compaction generations cite it.
"""
from agent.memory.hashing import content_hash


def test_same_content_same_hash():
    assert content_hash("the quick brown fox") == content_hash("the quick brown fox")


def test_different_content_different_hash():
    assert content_hash("the quick brown fox") != content_hash("the slow brown fox")


def test_hash_is_a_64_char_hex_sha256_digest():
    h = content_hash("anything")

    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
