import unittest
import numpy as np
from src.min_hash_generator import MinHash
from src.locality_sensitive_hashing import LSH


class TestLSH(unittest.TestCase):

    def setUp(self):
        self.num_perms = 128
        self.num_bands = 16
        self.num_rows = 8
        self.lsh = LSH(num_bands=self.num_bands, num_rows=self.num_rows)

    def create_minhash(self, words, num_perms=128, seed=42):
        mh = MinHash(num_permutations=num_perms, seed=seed)
        for word in words:
            mh.update(word)
        return mh

    def test_lsh_initialization(self):
        self.assertEqual(self.lsh.num_bands, 16)
        self.assertEqual(self.lsh.num_rows, 8)
        self.assertEqual(self.lsh.num_permutations, 128)
        self.assertEqual(len(self.lsh.tables), 16)
        self.assertEqual(len(self.lsh.signatures), 0)

    def test_lsh_initialization_custom_params(self):
        lsh = LSH(num_bands=32, num_rows=4)
        self.assertEqual(lsh.num_bands, 32)
        self.assertEqual(lsh.num_rows, 4)
        self.assertEqual(lsh.num_permutations, 128)
        self.assertEqual(len(lsh.tables), 32)

    def test_insert_single_document(self):
        mh = self.create_minhash(["hello", "world"])
        self.lsh.insert("doc1", mh)

        self.assertIn("doc1", self.lsh.signatures)
        self.assertTrue(np.array_equal(self.lsh.signatures["doc1"], mh.signature))

    def test_insert_multiple_documents(self):
        mh1 = self.create_minhash(["hello", "world"])
        mh2 = self.create_minhash(["foo", "bar"])
        mh3 = self.create_minhash(["baz", "qux"])

        self.lsh.insert("doc1", mh1)
        self.lsh.insert("doc2", mh2)
        self.lsh.insert("doc3", mh3)

        self.assertEqual(len(self.lsh.signatures), 3)
        self.assertIn("doc1", self.lsh.signatures)
        self.assertIn("doc2", self.lsh.signatures)
        self.assertIn("doc3", self.lsh.signatures)

    def test_insert_wrong_num_permutations(self):
        mh = MinHash(num_permutations=64)
        mh.update("test")

        with self.assertRaises(ValueError) as context:
            self.lsh.insert("doc1", mh)

        self.assertIn("64", str(context.exception))
        self.assertIn("128", str(context.exception))

    def test_insert_updates_buckets(self):
        mh = self.create_minhash(["test"])
        self.lsh.insert("doc1", mh)

        found = False
        for table in self.lsh.tables:
            for bucket in table.values():
                if "doc1" in bucket:
                    found = True
                    break
            if found:
                break

        self.assertTrue(found, "Document not found in any bucket")

    def test_insert_duplicate_id_overwrites(self):
        mh1 = self.create_minhash(["hello", "world"])
        mh2 = self.create_minhash(["different", "text"])

        self.lsh.insert("doc1", mh1)
        self.lsh.insert("doc1", mh2)

        self.assertEqual(len(self.lsh.signatures), 1)
        self.assertTrue(np.array_equal(self.lsh.signatures["doc1"], mh2.signature))

    def test_query_empty_index(self):
        mh = self.create_minhash(["test"])
        candidates = self.lsh.query(mh)

        self.assertEqual(len(candidates), 0)
        self.assertIsInstance(candidates, set)

    def test_query_identical_document(self):
        mh = self.create_minhash(["hello", "world"], seed=42)
        query_mh = self.create_minhash(["hello", "world"], seed=42)
        candidates = self.lsh.query(query_mh)

        self.lsh.insert("doc1", mh)
        self.assertIn("doc1", candidates)

    def test_query_similar_documents(self):
        seed = 42

        mh1 = self.create_minhash(["the", "quick", "brown", "fox"], seed=seed)
        mh2 = self.create_minhash(["the", "quick", "brown", "dog"], seed=seed)
        mh3 = self.create_minhash(["completely", "different", "words"], seed=seed)

        self.lsh.insert("doc1", mh1)
        self.lsh.insert("doc2", mh2)
        self.lsh.insert("doc3", mh3)

        query_mh = self.create_minhash(["the", "quick", "brown", "fox"], seed=seed)
        candidates = self.lsh.query(query_mh)

        self.assertIn("doc1", candidates)

    def test_query_wrong_num_permutations(self):
        mh = MinHash(num_permutations=64)
        mh.update("test")

        with self.assertRaises(ValueError) as context:
            self.lsh.query(mh)

        self.assertIn("64", str(context.exception))
        self.assertIn("128", str(context.exception))

    def test_query_returns_set(self):
        mh1 = self.create_minhash(["test"])
        query_mh = self.create_minhash(["test"])
        candidates = self.lsh.query(query_mh)

        self.lsh.insert("doc1", mh1)
        self.assertIsInstance(candidates, set)

    def test_query_does_not_modify_index(self):
        mh1 = self.create_minhash(["test"])
        self.lsh.insert("doc1", mh1)

        initial_sig_count = len(self.lsh.signatures)
        query_mh = self.create_minhash(["query"])
        self.lsh.query(query_mh)

        self.assertEqual(len(self.lsh.signatures), initial_sig_count)

    def test_query_dissimilar_documents(self):
        mh1 = self.create_minhash(["aaa", "bbb", "ccc", "ddd"], seed=1)
        mh2 = self.create_minhash(["eee", "fff", "ggg", "hhh"], seed=2)
        mh3 = self.create_minhash(["iii", "jjj", "kkk", "lll"], seed=3)

        self.lsh.insert("doc1", mh1)
        self.lsh.insert("doc2", mh2)
        self.lsh.insert("doc3", mh3)

        query_mh = self.create_minhash(["aaa", "bbb", "ccc", "ddd"], seed=1)
        candidates = self.lsh.query(query_mh)

        self.assertIn("doc1", candidates)


if __name__ == '__main__':
    unittest.main(verbosity=2)