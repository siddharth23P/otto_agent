"""Content-addressable hashing for agent/memory/store.py's permanent chunk
table.

sha256 over the raw UTF-8 text, hex digest. Identical content always
hashes identically -- flushing the same text twice (a repeated tool
result, say) naturally dedupes onto the same DB row instead of storing a
second copy, and it's what lets a bullet's `hash_refs` (agent/memory/
queue.py) point at exactly the same permanent row no matter how many
compaction generations later it's cited from.
"""
import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
