import hashlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from src.min_hash_generator import MinHash

class LshGenerator:
    def __init__(self, num_bands: int, num_rows: int):
        self.num_bands = num_bands
        self.num_rows = num_rows

    def generate_lsh(self, docs: Dict[str, 'MinHash']) -> 'LSH':
        lsh = LSH(num_bands=self.num_bands, num_rows=self.num_rows)
        for doc, minhash in docs.items():
            lsh.insert(doc, minhash)
        return lsh


class LSH:

    def __init__(self, num_bands: int, num_rows: int):
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.num_permutations = num_bands * num_rows

        self.tables = [defaultdict(set) for _ in range(num_bands)]
        self.signatures = {}

    def insert(self, doc_id: str, minhash: 'MinHash'):
        if minhash.num_permutations != self.num_permutations:
            raise ValueError(
                f"MinHash has {minhash.num_permutations} permutations, "
                f"expected {self.num_permutations}"
            )
        self.signatures[doc_id] = minhash.signature

        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = minhash.signature[start:end]

            bucket_hash = self.get_hash(band.tobytes())

            self.tables[band_idx][bucket_hash].add(doc_id)

    @staticmethod
    def get_hash(x: bytes):
        return np.uint64(int(hashlib.blake2b(str(x).encode(), digest_size=8).hexdigest(), 16))

    def query(self, minhash: 'MinHash') -> Set[str]:
        if minhash.num_permutations != self.num_permutations:
            raise ValueError(
                f"MinHash has {minhash.num_permutations} permutations, "
                f"expected {self.num_permutations}"
            )

        candidates = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = minhash.signature[start:end]

            bucket_hash = self.get_hash(band.tobytes())

            if bucket_hash in self.tables[band_idx]:
                candidates.update(self.tables[band_idx][bucket_hash])

        return candidates

    def find_similar(self, doc_id: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        if doc_id not in self.signatures:
            raise ValueError(f"Document {doc_id} not found in index")

        query_minhash = type('MinHash', (), {
            'signature': self.signatures[doc_id],
            'num_permutations': self.num_permutations
        })()

        candidates = self.query(query_minhash)
        candidates.discard(doc_id)

        results = []
        for candidate_id in candidates:
            candidate_minhash = type('MinHash', (), {
                'signature': self.signatures[candidate_id],
                'num_permutations': self.num_permutations
            })()

            similarity = query_minhash.jaccard_similarity(candidate_minhash)

            if similarity >= threshold:
                results.append((candidate_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results