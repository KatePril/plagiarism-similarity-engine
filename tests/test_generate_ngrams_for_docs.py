import unittest

from ngram_base_test_case import NgramBaseTestCase


class TestGenerateNgramsForDocs(NgramBaseTestCase):

    def test_generate_ngrams_for_docs_single_document(self):
        documents = {
            'doc1': ['the', 'quick', 'brown', 'fox']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result), 1)
        self.assertIn('doc1', result)

        expected_ngrams = {('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')}
        self.assertEqual(set(result['doc1'].keys()), expected_ngrams)

        for prob in result['doc1'].values():
            self.assertGreater(prob, 0)
            self.assertLessEqual(prob, 1)

    def test_generate_ngrams_for_docs_multiple_documents(self):
        documents = {
            'doc1': ['the', 'cat', 'sat'],
            'doc2': ['the', 'dog', 'ran'],
            'doc3': ['a', 'bird', 'flew']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result), 3)
        self.assertIn('doc1', result)
        self.assertIn('doc2', result)
        self.assertIn('doc3', result)

        self.assertEqual(set(result['doc1'].keys()), {('the', 'cat'), ('cat', 'sat')})
        self.assertEqual(set(result['doc2'].keys()), {('the', 'dog'), ('dog', 'ran')})
        self.assertEqual(set(result['doc3'].keys()), {('a', 'bird'), ('bird', 'flew')})

    def test_generate_ngrams_for_docs_empty_documents(self):
        documents = {}
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(result, {})

    def test_generate_ngrams_for_docs_empty_token_list(self):
        documents = {
            'doc1': []
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result), 1)
        self.assertIn('doc1', result)
        self.assertEqual(result['doc1'], {})

    def test_generate_ngrams_for_docs_insufficient_tokens(self):
        documents = {
            'doc1': ['hello'],
            'doc2': ['world']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(result['doc1'], {})
        self.assertEqual(result['doc2'], {})

    def test_generate_ngrams_for_docs_exact_n_tokens(self):
        documents = {
            'doc1': ['hello', 'world'],
            'doc2': ['foo', 'bar']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['doc1']), 1)
        self.assertEqual(len(result['doc2']), 1)
        self.assertIn(('hello', 'world'), result['doc1'])
        self.assertIn(('foo', 'bar'), result['doc2'])

    def test_generate_ngrams_for_docs_with_duplicates(self):
        documents = {
            'doc1': ['the', 'cat', 'the', 'cat']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        ngrams = result['doc1']
        self.assertEqual(len(ngrams), 2)

        # ('the', 'cat'): (2+1)/(3+2) = 3/5 = 0.6
        # ('cat', 'the'): (1+1)/(3+2) = 2/5 = 0.4
        self.assertAlmostEqual(ngrams[('the', 'cat')], 0.6)
        self.assertAlmostEqual(ngrams[('cat', 'the')], 0.4)

    def test_generate_ngrams_for_docs_trigrams(self):
        documents = {
            'doc1': ['the', 'quick', 'brown', 'fox', 'jumps']
        }
        result = self.trigram_gen.generate_ngrams_for_docs(documents)

        expected_trigrams = {
            ('the', 'quick', 'brown'),
            ('quick', 'brown', 'fox'),
            ('brown', 'fox', 'jumps')
        }
        self.assertEqual(set(result['doc1'].keys()), expected_trigrams)

    def test_generate_ngrams_for_docs_unigrams(self):
        documents = {
            'doc1': ['hello', 'world', 'hello']
        }
        result = self.unigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['doc1']), 2)
        self.assertIn(('hello',), result['doc1'])
        self.assertIn(('world',), result['doc1'])

        # 'hello': (2+1)/(3+2) = 0.6
        # 'world': (1+1)/(3+2) = 0.4
        self.assertAlmostEqual(result['doc1'][('hello',)], 0.6)
        self.assertAlmostEqual(result['doc1'][('world',)], 0.4)

    def test_generate_ngrams_for_docs_different_vocab_sizes(self):
        documents = {
            'doc1': ['a', 'b', 'a'],  # vocab_size = 2
            'doc2': ['x', 'y', 'z', 'x']  # vocab_size = 3
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        # ('a', 'b'): (1+1)/(2+2) = 0.5
        # ('b', 'a'): (1+1)/(2+2) = 0.5
        self.assertAlmostEqual(result['doc1'][('a', 'b')], 0.5)
        self.assertAlmostEqual(result['doc1'][('b', 'a')], 0.5)

        # Each bigram appears once: (1+1)/(3+3) = 1/3
        for prob in result['doc2'].values():
            self.assertAlmostEqual(prob, 1 / 3)

    def test_generate_ngrams_for_docs_preserves_document_keys(self):
        documents = {
            'document_1': ['a', 'b', 'c'],
            'file_xyz': ['d', 'e', 'f'],
            'text_123': ['g', 'h', 'i']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(set(result.keys()), {'document_1', 'file_xyz', 'text_123'})

    def test_generate_ngrams_for_docs_probability_properties(self):
        documents = {
            'doc1': ['the', 'cat', 'sat', 'on', 'the', 'mat']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        for prob in result['doc1'].values():
            self.assertGreater(prob, 0)

        total_prob = sum(result['doc1'].values())
        self.assertLessEqual(total_prob, 1.0)

    def test_generate_ngrams_for_docs_mixed_document_sizes(self):
        documents = {
            'short': ['a', 'b'],
            'medium': ['x', 'y', 'z', 'w'],
            'long': ['m', 'n', 'o', 'p', 'q', 'r', 's']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['short']), 1)  # 1 bigram
        self.assertEqual(len(result['medium']), 3)  # 3 bigrams
        self.assertEqual(len(result['long']), 6)  # 6 bigrams

    def test_generate_ngrams_for_docs_independence(self):
        documents = {
            'doc1': ['cat', 'dog', 'cat'],
            'doc2': ['cat', 'dog', 'cat']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(result['doc1'], result['doc2'])

    def test_generate_ngrams_for_docs_numeric_document_ids(self):
        documents = {
            1: ['hello', 'world'],
            2: ['foo', 'bar'],
            100: ['test', 'case']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result), 3)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(100, result)


if __name__ == '__main__':
    unittest.main()