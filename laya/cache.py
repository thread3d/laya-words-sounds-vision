"""A bounded, content-addressed cache for an expensive block's output.

The tower pass is the expensive half of a decision about a picture or a sound, and one call that asks
several questions about the same input must not pay it several times. The key is a hash of the
normalised input, so two encodings of the same picture -- or two decodes of the same recording --
are one entry, and the store is bounded so a long-lived server cannot grow without limit.

The same class serves both modalities; each keeps its own thin name for it
(:class:`laya.vision.VisualCache`, :class:`laya.audio.AudioCache`) because the block they cache is
not the same width for the same input -- `summary` changes it, and that has to be part of the key.
"""
from collections import OrderedDict
from typing import Callable


class ContentCache:
    """An LRU store keyed by the content of a tensor, bounded by entry count."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(value, *flags: str) -> str:
        """Content key for `value`, with any `flags` that change the stored block folded in."""
        import hashlib
        digest = hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return digest + "".join("|" + flag for flag in flags if flag)

    def get_or_encode(self, value, encode: Callable, *flags: str):
        """Return the cached block for `value`, or compute and store it with `encode(value)`."""
        k = self.key(value, *flags)
        if k in self._store:
            self.hits += 1
            self._store.move_to_end(k)
            return self._store[k]
        self.misses += 1
        result = encode(value)
        self._store[k] = result
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return result

    def clear(self):
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
