from dataclasses import dataclass
from typing import List

from src.locality_sensitive_hashing import LSH

@dataclass
class SimilarPair:
    doc1_name: str
    doc2_name: str
    similarity_score: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarPair):
            return False
        return (
                self.doc1_name == other.doc1_name and
                self.doc2_name == other.doc2_name and
                self.similarity_score == other.similarity_score
        )

class SimilarityEvaluator:
    def __init__(self, lsh: 'LSH', threshold: float = 0.5):
        self.lsh = lsh
        self.threshold = threshold

    def get_similar_pairs(self, docs: List[str]) -> List[SimilarPair]:
        result = []
        for doc in docs:
            similarities = self.lsh.find_similar(doc, threshold=self.threshold)
            for similar_doc, similarity_score in similarities:
                doc1, doc2 = sorted([doc, similar_doc])
                result.append(SimilarPair(doc1, doc2, similarity_score))
        cleaned_result = self._clean_result(result)
        return cleaned_result

    @staticmethod
    def _clean_result(result: List[SimilarPair]) -> List[SimilarPair]:
        result_deduplicated = list(set(result))
        result_deduplicated.sort(key=lambda x: x.similarity_score, reverse=True)
        return result_deduplicated
