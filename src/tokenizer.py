"""
Character-level and simple BPE tokenizer.

Character-level is a good start: easy to reason about, no external dependencies.
BPE is better for real text: handles rare words, better token efficiency.
"""

import re
from collections import defaultdict
from typing import Optional


class CharTokenizer:
    """Character-level tokenizer. Vocabulary = all unique characters in the training text."""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)


class BPETokenizer:
    """
    Minimal byte pair encoding tokenizer.

    Greedily merges the most frequent pair of adjacent tokens until the
    vocabulary reaches target_vocab_size.
    """

    def __init__(self, target_vocab_size: int = 512):
        self.target_vocab_size = target_vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _get_pairs(self, vocab: dict[tuple, int]) -> dict[tuple, int]:
        pairs: dict[tuple, int] = defaultdict(int)
        for word, freq in vocab.items():
            for a, b in zip(word[:-1], word[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _merge_pair(self, pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
        merged = pair[0] + pair[1]
        new_vocab: dict[tuple, int] = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, text: str) -> None:
        # start with character-level vocabulary
        words = re.findall(r"\S+", text)
        word_freq: dict[tuple, int] = defaultdict(int)
        for word in words:
            word_freq[tuple(word)] += 1

        # initial single-char vocab
        chars = sorted(set(c for word in word_freq for c in word))
        self.vocab = {c: i for i, c in enumerate(chars)}
        next_id = len(self.vocab)

        # merge until target vocab size
        while len(self.vocab) < self.target_vocab_size:
            pairs = self._get_pairs(word_freq)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            word_freq = self._merge_pair(best, word_freq)
            merged = best[0] + best[1]
            self.merges.append(best)
            self.vocab[merged] = next_id
            next_id += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        for pair in self.merges:
            merged = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
