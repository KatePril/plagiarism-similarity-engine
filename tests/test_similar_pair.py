import unittest

from src.similarity_evaluator import SimilarPair


class TestSimilarPair(unittest.TestCase):

    def test_creation(self):
        pair = SimilarPair("doc1", "doc2", 0.85)
        self.assertEqual(pair.doc1_name, "doc1")
        self.assertEqual(pair.doc2_name, "doc2")
        self.assertEqual(pair.similarity_score, 0.85)

    def test_equality_same_pairs(self):
        pair1 = SimilarPair("doc1", "doc2", 0.85)
        pair2 = SimilarPair("doc1", "doc2", 0.85)
        self.assertEqual(pair1, pair2)

    def test_equality_different_docs(self):
        pair1 = SimilarPair("doc1", "doc2", 0.85)
        pair2 = SimilarPair("doc1", "doc3", 0.85)
        self.assertNotEqual(pair1, pair2)

    def test_equality_different_scores(self):
        pair1 = SimilarPair("doc1", "doc2", 0.85)
        pair2 = SimilarPair("doc1", "doc2", 0.90)
        self.assertNotEqual(pair1, pair2)

    def test_equality_wrong_type(self):
        pair = SimilarPair("doc1", "doc2", 0.85)
        self.assertNotEqual(pair, "not a pair")
        self.assertNotEqual(pair, None)
        self.assertNotEqual(pair, {"doc1": "doc1", "doc2": "doc2"})


if __name__ == '__main__':
    unittest.main()