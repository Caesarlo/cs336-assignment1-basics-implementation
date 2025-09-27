import torch
import torch.nn as nn
import os

import regex as re

from typing import BinaryIO, List, Dict, Tuple
from jaxtyping import Float
from collections import Counter, defaultdict

from loguru import logger
from tqdm import tqdm

from cs336_basics.tokenizer.RegexTokenizer import RegexTokenizer


class BPE_Tokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []

        # Initialize empty structures
        self.word_freqs = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}

        self.regex_tokenizer = RegexTokenizer(
            filepath=input_path, special_tokens=self.special_tokens)

    def _initialize_vocab(self) -> Dict[int, bytes]:
        vocab = {}
        inverse_vocab = {}

        for i, token in enumerate(self.special_tokens):
            token_bytes = token.encode('utf-8')
            vocab[i] = token_bytes
            inverse_vocab[token_bytes] = i

        for i in range(256):
            # vocab[len(self.special_tokens)+i] = bytes(i)
            token_id = len(self.special_tokens)+i
            byte_token = bytes([i])
            vocab[token_id] = byte_token
            inverse_vocab[byte_token] = token_id

        return vocab, inverse_vocab

    def _pretokenize_corpus(self) -> Dict[Tuple[bytes, ...], int]:
        word_freqs = defaultdict(int)

        tokens = self.regex_tokenizer.pretokenize()

        for token in tokens:
            word_bytes = token.encode('utf-8')
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            word_freqs[word_tuple] += 1

        return dict(word_freqs)

    def _count_pairs(self, word_freqs: dict[tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
        pair_counts = defaultdict(int)

        for word, freq in word_freqs.items():
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += freq
        return dict(pair_counts)

    def _merge_pair(self, word_freqs: Dict[Tuple[bytes, ...], int], pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
        new_word_freqs = {}
        pair_a, pair_b = pair
        merged_token = pair_a + pair_b

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word)-1 and word[i] == pair_a and word[i+1] == pair_b):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] = freq
        return new_word_freqs

    def _compute_merges(self, word_freqs: dict[tuple[bytes, ...], int]) -> List[tuple[bytes, bytes]]:
        merges = []
        current_vocab_size = len(self.vocab)

        working_freqs = dict(word_freqs)

        target_merges = self.vocab_size-current_vocab_size

        for merge_num in range(target_merges):
            pair_counts = self._count_pairs(working_freqs)
            if not pair_counts:
                break
            most_frequent_pair = max(
                pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            working_freqs = self._merge_pair(working_freqs, most_frequent_pair)
            merges.append(most_frequent_pair)

        return merges

    def _update_vocab_with_merges(self):
        base_vocab_size = 256+len(self.special_tokens)
        for i, (token_a, token_b) in enumerate(self.merges):
            merged_token = token_a + token_b
            token_id = base_vocab_size + i
            self.vocab[token_id] = merged_token
            self.inverse_vocab[merged_token] = token_id

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        min_vocab_size = 256 + len(self.special_tokens)
        assert self.vocab_size >= min_vocab_size, f"Vocab size must be at least {min_vocab_size}"

        self.vocab, self.inverse_vocab = self._initialize_vocab()

        word_freqs = self._pretokenize_corpus()

        self.merges = self._compute_merges(word_freqs)

        self._update_vocab_with_merges()

        return self.vocab, self.merges

    def encode(self, text):
        ...

    def decode(self, tokens):
        ...


if __name__ == "__main__":

    special_token = ["<|endoftext|>"]
    # bpe = BPE_Tokenizer("../../data/TinyStoriesV2-GPT4-valid.txt",
    #                     1000, special_tokens=special_token)
    bpe = BPE_Tokenizer("../../data/test.txt",
                        1000, special_tokens=special_token)
    tokens, merges, = bpe.train()
    logger.info(merges)
    # logger.info(tokens[:20])
