import unittest
from unittest.mock import Mock

from src.similarity_evaluator import SimilarPair, SimilarityEvaluator


class TestSimilarityEvaluator(unittest.TestCase):

    def setUp(self):
        self.mock_lsh = Mock()
        self.evaluator = SimilarityEvaluator(self.mock_lsh, threshold=0.5)

    def test_initialization(self):
        self.assertEqual(self.evaluator.lsh, self.mock_lsh)
        self.assertEqual(self.evaluator.threshold, 0.5)

    def test_initialization_default_threshold(self):
        evaluator = SimilarityEvaluator(self.mock_lsh)
        self.assertEqual(evaluator.threshold, 0.5)

    def test_get_similar_pairs_empty_docs(self):
        result = self.evaluator.get_similar_pairs([])
        self.assertEqual(result, [])
        self.mock_lsh.find_similar.assert_not_called()

    def test_get_similar_pairs_no_similarities(self):
        self.mock_lsh.find_similar.return_value = []
        docs = ["doc1", "doc2", "doc3"]
        result = self.evaluator.get_similar_pairs(docs)

        self.assertEqual(result, [])
        self.assertEqual(self.mock_lsh.find_similar.call_count, 3)

    def test_get_similar_pairs_single_similarity(self):
        self.mock_lsh.find_similar.side_effect = [
            [("doc2", 0.8)],
            [("doc1", 0.8)],
        ]
        docs = ["doc1", "doc2"]
        result = self.evaluator.get_similar_pairs(docs)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], SimilarPair("doc1", "doc2", 0.8))

    def test_get_similar_pairs_deduplication(self):
        self.mock_lsh.find_similar.side_effect = [
            [("doc2", 0.8)],
            [("doc1", 0.8)],
            [],
        ]
        docs = ["doc1", "doc2", "doc3"]
        result = self.evaluator.get_similar_pairs(docs)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], SimilarPair("doc1", "doc2", 0.8))

    def test_get_similar_pairs_sorting(self):
        self.mock_lsh.find_similar.side_effect = [
            [("doc2", 0.6), ("doc3", 0.9)],
            [("doc1", 0.6), ("doc3", 0.75)],
            [("doc1", 0.9), ("doc2", 0.75)],
        ]
        docs = ["doc1", "doc2", "doc3"]
        result = self.evaluator.get_similar_pairs(docs)

        self.assertEqual(len(result), 3)
        self.assertGreaterEqual(result[0].similarity_score, result[1].similarity_score)
        self.assertGreaterEqual(result[1].similarity_score, result[2].similarity_score)

        self.assertEqual(result[0].similarity_score, 0.9)
        self.assertEqual(result[1].similarity_score, 0.75)
        self.assertEqual(result[2].similarity_score, 0.6)

    def test_get_similar_pairs_doc_name_ordering(self):
        self.mock_lsh.find_similar.side_effect = [
            [("doc_a", 0.8)],
        ]
        docs = ["doc_b"]
        result = self.evaluator.get_similar_pairs(docs)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].doc1_name, "doc_a")
        self.assertEqual(result[0].doc2_name, "doc_b")

    def test_get_similar_pairs_threshold_propagation(self):
        self.mock_lsh.find_similar.return_value = []
        evaluator = SimilarityEvaluator(self.mock_lsh, threshold=0.7)
        docs = ["doc1"]
        evaluator.get_similar_pairs(docs)
        self.mock_lsh.find_similar.assert_called_with("doc1", threshold=0.7)

    def test_get_similar_pairs_multiple_similarities(self):
        self.mock_lsh.find_similar.side_effect = [
            [("doc2", 0.85), ("doc3", 0.65)],
            [("doc1", 0.85), ("doc4", 0.55)],
            [("doc1", 0.65), ("doc4", 0.70)],
            [("doc2", 0.55), ("doc3", 0.70)],
        ]
        docs = ["doc1", "doc2", "doc3", "doc4"]

        result = self.evaluator.get_similar_pairs(docs)
        scores = [pair.similarity_score for pair in result]
        self.assertEqual(len(result), 4)
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_clean_result_empty_list(self):
        result = SimilarityEvaluator._clean_result([])
        self.assertEqual(result, [])

    def test_clean_result_deduplication(self):
        pairs = [
            SimilarPair("doc1", "doc2", 0.8),
            SimilarPair("doc1", "doc2", 0.8),
            SimilarPair("doc3", "doc4", 0.9),
        ]
        result = SimilarityEvaluator._clean_result(pairs)
        self.assertEqual(len(result), 2)

    def test_clean_result_sorting(self):
        pairs = [
            SimilarPair("doc1", "doc2", 0.6),
            SimilarPair("doc3", "doc4", 0.9),
            SimilarPair("doc5", "doc6", 0.75),
        ]
        result = SimilarityEvaluator._clean_result(pairs)

        self.assertEqual(result[0].similarity_score, 0.9)
        self.assertEqual(result[1].similarity_score, 0.75)
        self.assertEqual(result[2].similarity_score, 0.6)


if __name__ == '__main__':
    unittest.main()