import unittest
from unittest.mock import Mock, patch, call
from src.min_hash_generator import MinHashGenerator, MinHash


class TestMinHashGenerator(unittest.TestCase):

    def test_init_default_parameters(self):
        generator = MinHashGenerator()
        self.assertEqual(generator.num_permutations, 128)
        self.assertEqual(generator.seed, 42)

    def test_init_custom_parameters(self):
        generator = MinHashGenerator(num_permutations=64, seed=123)
        self.assertEqual(generator.num_permutations, 64)
        self.assertEqual(generator.seed, 123)

    def test_generate_minhashes_empty_docs(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        result = generator.generate_minhashes({})
        self.assertEqual(result, {})

    def test_generate_minhashes_single_doc(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{"ngram1": 0.33}, {"ngram2": 0.33}, {"ngram3": 0.33}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 1)
        self.assertIn("doc1", result)
        self.assertIsInstance(result["doc1"], MinHash)

    def test_generate_minhashes_multiple_docs(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{"ngram1": 0.5}, {"ngram2": 0.5}],
            "doc2": [{"ngram3": 0.5}, {"ngram4": 0.5}],
            "doc3": [{"ngram5": 1.0}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 3)
        self.assertIn("doc1", result)
        self.assertIn("doc2", result)
        self.assertIn("doc3", result)
        for minhash in result.values():
            self.assertIsInstance(minhash, MinHash)

    def test_generate_minhashes_empty_ngram_list(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 1)
        self.assertIn("doc1", result)
        self.assertIsInstance(result["doc1"], MinHash)

    @patch('src.min_hash_generator.MinHash')
    def test_generate_minhashes_creates_correct_minhash_instances(self, MockMinHash):
        mock_minhash_instance = Mock()
        MockMinHash.return_value = mock_minhash_instance

        generator = MinHashGenerator(num_permutations=64, seed=100)
        docs = {
            "doc1": [{"ngram1": 0.5}, {"ngram2": 0.5}],
            "doc2": [{"ngram3": 1.0}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(MockMinHash.call_count, 2)
        MockMinHash.assert_any_call(64, seed=100)

    @patch('src.min_hash_generator.MinHash')
    def test_generate_minhashes_calls_update_correctly(self, MockMinHash):
        mock_minhash_instance = Mock()
        MockMinHash.return_value = mock_minhash_instance

        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [('a', 'b'), ('b', 'c'), ('c', 'd')]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(mock_minhash_instance.update.call_count, 3)
        expected_calls = [call(('a', 'b')), call(('b', 'c')), call(('c', 'd'))]
        mock_minhash_instance.update.assert_has_calls(expected_calls, any_order=False)

    @patch('src.min_hash_generator.MinHash')
    def test_generate_minhashes_updates_all_docs(self, MockMinHash):
        mock_instances = [Mock(), Mock()]
        MockMinHash.side_effect = mock_instances

        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{"ngram1": 0.5}, {"ngram2": 0.5}],
            "doc2": [{"ngram3": 0.33}, {"ngram4": 0.33}, {"ngram5": 0.33}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(mock_instances[0].update.call_count, 2)
        self.assertEqual(mock_instances[1].update.call_count, 3)

    def test_generate_minhashes_returns_correct_keys(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "document_a": [{"ngram1": 1.0}],
            "document_b": [{"ngram2": 1.0}],
            "document_c": [{"ngram3": 1.0}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(set(result.keys()), set(docs.keys()))

    def test_generate_minhashes_deterministic_with_same_seed(self):
        docs = {
            "doc1": [{"ngram1": 0.33}, {"ngram2": 0.33}, {"ngram3": 0.33}]
        }

        generator1 = MinHashGenerator(num_permutations=128, seed=42)
        result1 = generator1.generate_minhashes(docs)

        generator2 = MinHashGenerator(num_permutations=128, seed=42)
        result2 = generator2.generate_minhashes(docs)

        import numpy as np
        np.testing.assert_array_equal(
            result1["doc1"].signature,
            result2["doc1"].signature
        )

    def test_generate_minhashes_different_with_different_seed(self):
        docs = {
            "doc1": [{"ngram1": 0.33}, {"ngram2": 0.33}, {"ngram3": 0.33}]
        }

        generator1 = MinHashGenerator(num_permutations=128, seed=42)
        result1 = generator1.generate_minhashes(docs)

        generator2 = MinHashGenerator(num_permutations=128, seed=100)
        result2 = generator2.generate_minhashes(docs)

        import numpy as np
        self.assertFalse(
            np.array_equal(
                result1["doc1"].signature,
                result2["doc1"].signature
            )
        )

    def test_generate_minhashes_with_various_ngram_types(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{"string_ngram": 0.2}, {123: 0.2}, {(1, 2): 0.2}, {"another_ngram": 0.2}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result["doc1"], MinHash)

    def test_generate_minhashes_preserves_doc_keys_with_special_chars(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc-1": [{"ngram1": 1.0}],
            "doc_2": [{"ngram2": 1.0}],
            "doc.3": [{"ngram3": 1.0}],
            "doc 4": [{"ngram4": 1.0}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(set(result.keys()), set(docs.keys()))

    def test_generate_minhashes_num_permutations_propagated(self):
        generator = MinHashGenerator(num_permutations=64, seed=42)
        docs = {
            "doc1": [{"ngram1": 0.5}, {"ngram2": 0.5}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(result["doc1"].num_permutations, 64)

    def test_generate_minhashes_seed_propagated(self):
        generator = MinHashGenerator(num_permutations=128, seed=999)
        docs = {
            "doc1": [{"ngram1": 0.5}, {"ngram2": 0.5}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(result["doc1"].seed, 999)

    def test_generate_minhashes_large_number_of_docs(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {f"doc{i}": [{f"ngram{i}": 1.0}] for i in range(1000)}
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 1000)
        for i in range(1000):
            self.assertIn(f"doc{i}", result)
            self.assertIsInstance(result[f"doc{i}"], MinHash)

    def test_generate_minhashes_duplicate_ngrams_in_doc(self):
        generator = MinHashGenerator(num_permutations=128, seed=42)
        docs = {
            "doc1": [{"ngram1": 0.6}, {"ngram2": 0.4}, {"ngram1": 0.6}, {"ngram2": 0.4}, {"ngram1": 0.6}]
        }
        result = generator.generate_minhashes(docs)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result["doc1"], MinHash)


if __name__ == '__main__':
    unittest.main()