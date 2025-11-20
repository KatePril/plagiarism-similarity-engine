import unittest
import numpy as np
from src.min_hash_generator import MinHash


class TestMinHash(unittest.TestCase):

    def test_init_default_parameters(self):
        mh = MinHash()
        self.assertEqual(mh.num_permutations, 128)
        self.assertEqual(mh.seed, 42)
        self.assertEqual(len(mh.signature), 128)
        self.assertTrue(np.all(mh.signature == np.iinfo(np.uint64).max))

    def test_init_custom_parameters(self):
        mh = MinHash(num_permutations=64, seed=123)
        self.assertEqual(mh.num_permutations, 64)
        self.assertEqual(mh.seed, 123)
        self.assertEqual(len(mh.signature), 64)
        self.assertEqual(len(mh._a), 64)
        self.assertEqual(len(mh._b), 64)

    def test_init_deterministic_with_same_seed(self):
        mh1 = MinHash(num_permutations=64, seed=100)
        mh2 = MinHash(num_permutations=64, seed=100)
        np.testing.assert_array_equal(mh1._a, mh2._a)
        np.testing.assert_array_equal(mh1._b, mh2._b)

    def test_init_different_with_different_seed(self):
        mh1 = MinHash(num_permutations=64, seed=100)
        mh2 = MinHash(num_permutations=64, seed=200)
        self.assertFalse(np.array_equal(mh1._a, mh2._a))
        self.assertFalse(np.array_equal(mh1._b, mh2._b))

    def test_get_hash_deterministic(self):
        hash1 = MinHash.get_hash("test")
        hash2 = MinHash.get_hash("test")
        self.assertEqual(hash1, hash2)

    def test_get_hash_different_inputs(self):
        hash1 = MinHash.get_hash("test1")
        hash2 = MinHash.get_hash("test2")
        self.assertNotEqual(hash1, hash2)

    def test_get_hash_returns_uint64(self):
        result = MinHash.get_hash("test")
        self.assertEqual(result.dtype, np.uint64)

    def test_update_modifies_signature(self):
        mh = MinHash(num_permutations=128, seed=42)
        initial_signature = mh.signature.copy()
        mh.update("element1")
        self.assertFalse(np.array_equal(initial_signature, mh.signature))

    def test_update_multiple_elements(self):
        mh = MinHash(num_permutations=128, seed=42)
        mh.update("element1")
        sig_after_first = mh.signature.copy()
        mh.update("element2")
        self.assertFalse(np.array_equal(sig_after_first, mh.signature))

    def test_update_same_element_idempotent(self):
        mh = MinHash(num_permutations=128, seed=42)
        mh.update("element1")
        sig_after_first = mh.signature.copy()
        mh.update("element1")
        np.testing.assert_array_equal(sig_after_first, mh.signature)

    def test_jaccard_similarity_identical_sets(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=42)

        elements = ["a", "b", "c", "d", "e"]
        for elem in elements:
            mh1.update(elem)
            mh2.update(elem)

        similarity = mh1.jaccard_similarity(mh2)
        self.assertEqual(similarity, 1.0)

    def test_jaccard_similarity_disjoint_sets(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=42)

        for elem in ["a", "b", "c"]:
            mh1.update(elem)

        for elem in ["x", "y", "z"]:
            mh2.update(elem)

        similarity = mh1.jaccard_similarity(mh2)
        self.assertLess(similarity, 0.2)

    def test_jaccard_similarity_partial_overlap(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=42)

        for elem in ["a", "b", "c", "d"]:
            mh1.update(elem)

        for elem in ["c", "d", "e", "f"]:
            mh2.update(elem)

        similarity = mh1.jaccard_similarity(mh2)
        self.assertGreater(similarity, 0.15)
        self.assertLess(similarity, 0.55)

    def test_jaccard_similarity_empty_sets(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=42)

        similarity = mh1.jaccard_similarity(mh2)
        self.assertEqual(similarity, 1.0)

    def test_jaccard_similarity_mismatched_permutations_raises_error(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=64, seed=42)

        with self.assertRaises(ValueError) as context:
            mh1.jaccard_similarity(mh2)

        self.assertIn("num_permutations", str(context.exception))

    def test_jaccard_similarity_different_seeds_same_elements(self):
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=100)

        elements = ["a", "b", "c", "d", "e"]
        for elem in elements:
            mh1.update(elem)
            mh2.update(elem)

        similarity = mh1.jaccard_similarity(mh2)
        self.assertLess(similarity, 0.3)

    def test_signature_maintains_minimum(self):
        mh = MinHash(num_permutations=10, seed=42)

        mh.update("element1")
        sig1 = mh.signature.copy()

        mh.update("element2")
        sig2 = mh.signature.copy()

        np.testing.assert_array_equal(sig2, np.minimum(sig1, sig2))

    def test_update_with_various_types(self):
        mh = MinHash(num_permutations=64, seed=42)

        mh.update("string")
        mh.update(123)
        mh.update(45.67)
        mh.update(True)

        self.assertTrue(np.any(mh.signature != np.iinfo(np.uint64).max))


if __name__ == '__main__':
    unittest.main()