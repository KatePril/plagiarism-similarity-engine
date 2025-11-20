import hashlib

import numpy as np


class MinHashGenerator:
    def __init__(self, num_premutations=128, seed=42):
        self.num_premutations = num_premutations
        self.seed = seed

    def generate_minhashes(self, docs):
        min_hashes_dict = {}
        for doc, ngrams in docs.items():
            min_hash = MinHash(self.num_premutations, seed=self.seed)
            for ngram in ngrams:
                min_hash.update(ngram)
            min_hashes_dict[doc] = min_hash
        return min_hashes_dict


class MinHash:

    def __init__(self, num_permutations=128, seed=42):
        self.num_permutations = num_permutations
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self._a = self._rng.integers(1, np.iinfo(np.uint64).max, size=num_permutations, dtype=np.uint64)
        self._b = self._rng.integers(0, np.iinfo(np.uint64).max, size=num_permutations, dtype=np.uint64)

        self.signature = np.full(num_permutations, np.iinfo(np.uint64).max, dtype=np.uint64)

    def update(self, element):
        element_hash = self.get_hash(element)
        h_vals = (self._a * element_hash + self._b)
        self.signature = np.minimum(self.signature, h_vals)

    @staticmethod
    def get_hash(x):
        return np.uint64(int(hashlib.blake2b(str(x).encode(), digest_size=8).hexdigest(), 16))

    def jaccard_similarity(self, other):
        if self.num_permutations != other.num_permutations:
            raise ValueError("num_permutations of MinHashes must match")

        matches = np.sum(self.signature == other.signature)
        return matches / self.num_permutations
