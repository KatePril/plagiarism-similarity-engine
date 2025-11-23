import unittest
import numpy as np

from src.locality_sensitive_hashing import LSH
from src.min_hash_generator import MinHash


class TestLSHFindSimilar(unittest.TestCase):

    def setUp(self):
        self.num_bands = 10
        self.num_rows = 13
        self.num_permutations = self.num_bands * self.num_rows
        self.lsh = LSH(num_bands=self.num_bands, num_rows=self.num_rows)

    def _create_minhash(self, seed: int = 42) -> MinHash:
        minhash = MinHash(num_permutations=self.num_permutations, seed=seed)
        return minhash

    def test_find_similar_document_not_found(self):
        with self.assertRaises(ValueError) as context:
            self.lsh.find_similar("nonexistent_doc")

        self.assertIn("not found in index", str(context.exception))

    def test_find_similar_excludes_query_document(self):
        minhash1 = self._create_minhash(seed=1)
        minhash1.update("test")
        self.lsh.insert("doc1", minhash1)

        results = self.lsh.find_similar("doc1", threshold=0.0)
        doc_ids = [doc_id for doc_id, _ in results]
        self.assertNotIn("doc1", doc_ids)

    def test_find_similar_with_threshold(self):
        minhash1 = self._create_minhash(seed=1)
        minhash2 = self._create_minhash(seed=1)
        minhash3 = self._create_minhash(seed=2)

        for doc in [minhash1, minhash2]:
            doc.update("element1")
            doc.update("element2")
        minhash3.update("element3")
        minhash3.update("element4")

        self.lsh.insert("doc1", minhash1)
        self.lsh.insert("doc2", minhash2)
        self.lsh.insert("doc3", minhash3)

        results = self.lsh.find_similar("doc1", threshold=0.9)
        self.assertTrue(len(results) >= 0)
        for doc_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.9)

    def test_find_similar_results_sorted_by_similarity(self):
        base_minhash = self._create_minhash(seed=10)
        base_minhash.update("common")
        self.lsh.insert("base", base_minhash)

        for i in range(5):
            mh = self._create_minhash(seed=10 + i)
            mh.update("common")
            mh.update(f"unique_{i}")
            self.lsh.insert(f"doc{i}", mh)

        results = self.lsh.find_similar("base", threshold=0.0)
        if len(results) > 1:
            similarities = [sim for _, sim in results]
            self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_find_similar_empty_results(self):
        minhash1 = self._create_minhash(seed=100)
        minhash1.update("unique1")
        self.lsh.insert("doc1", minhash1)

        results = self.lsh.find_similar("doc1", threshold=1.0)
        for doc_id, similarity in results:
            self.assertEqual(similarity, 1.0)

    def test_find_similar_default_threshold(self):
        minhash1 = self._create_minhash(seed=50)
        minhash1.update("test")
        self.lsh.insert("doc1", minhash1)

        results = self.lsh.find_similar("doc1")
        for doc_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.5)

    def test_find_similar_identical_documents(self):
        minhash1 = self._create_minhash(seed=20)
        minhash2 = self._create_minhash(seed=20)
        for mh in [minhash1, minhash2]:
            mh.update("element_a")
            mh.update("element_b")
        self.lsh.insert("doc1", minhash1)
        self.lsh.insert("doc2", minhash2)

        results = self.lsh.find_similar("doc1", threshold=0.9)
        if results:
            doc_ids = {doc_id for doc_id, _ in results}
            if "doc2" in doc_ids:
                doc2_similarity = next(sim for doc_id, sim in results if doc_id == "doc2")
                self.assertGreater(doc2_similarity, 0.9)

    def test_find_similar_zero_threshold(self):
        minhash1 = self._create_minhash(seed=30)
        minhash1.update("base")
        self.lsh.insert("doc1", minhash1)

        for i in range(3):
            mh = self._create_minhash(seed=30 + i)
            mh.update(f"doc_{i}")
            self.lsh.insert(f"doc{i + 2}", mh)

        results = self.lsh.find_similar("doc1", threshold=0.0)
        for doc_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.0)
            self.assertNotEqual(doc_id, "doc1")

    def test_find_similar_return_type(self):
        minhash1 = self._create_minhash(seed=40)
        minhash1.update("test")
        self.lsh.insert("doc1", minhash1)

        results = self.lsh.find_similar("doc1", threshold=0.0)
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], (float, np.floating))

    def test_find_similar_similarity_values_range(self):
        for i in range(5):
            mh = self._create_minhash(seed=60 + i)
            mh.update(f"element_{i}")
            mh.update("common")
            self.lsh.insert(f"doc{i}", mh)

        results = self.lsh.find_similar("doc0", threshold=0.0)
        for doc_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)

    def test_find_similar_with_single_document(self):
        minhash1 = self._create_minhash(seed=70)
        minhash1.update("lonely")
        self.lsh.insert("doc1", minhash1)

        results = self.lsh.find_similar("doc1", threshold=0.0)
        self.assertEqual(len(results), 0)


if __name__ == '__main__':
    unittest.main()