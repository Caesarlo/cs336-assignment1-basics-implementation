import torch
import torch.nn as nn

from jaxtyping import Float

from loguru import logger


class BPE_Tokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.word_freqs = {}
        self.merges = {}
        self.vocab = {}

    def pretokenize(self):
        ...

    def get_freqs(self):
        ...

    def train(self, input_path, vocab_size, special_tokens):
        ...

    def merge_sort(self):
        ...

    def encode(self, text):
        ...

    def decode(self, tokens):
        ...
