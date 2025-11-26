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

        expected_ngrams = [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')]
        self.assertEqual(len(result['doc1']), 3)

        for i in range(len(result['doc1'])):
            self.assertEqual(result['doc1'][i], expected_ngrams[i])

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

        self.assertEqual(result['doc1'], [('the', 'cat'), ('cat', 'sat')])
        self.assertEqual(result['doc2'], [('the', 'dog'), ('dog', 'ran')])
        self.assertEqual(result['doc3'], [('a', 'bird'), ('bird', 'flew')])

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
        self.assertEqual(result['doc1'], [])

    def test_generate_ngrams_for_docs_insufficient_tokens(self):
        documents = {
            'doc1': ['hello'],
            'doc2': ['world']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(result['doc1'], [])
        self.assertEqual(result['doc2'], [])

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
        self.assertEqual(len(ngrams), 3)
        self.assertEqual(len([n for n in ngrams if n == ('the', 'cat')]), 2)
        self.assertEqual(len([n for n in ngrams if n == ('cat', 'the')]), 1)

    def test_generate_ngrams_for_docs_trigrams(self):
        documents = {
            'doc1': ['the', 'quick', 'brown', 'fox', 'jumps']
        }
        result = self.trigram_gen.generate_ngrams_for_docs(documents)

        expected_trigrams = [
            ('the', 'quick', 'brown'),
            ('quick', 'brown', 'fox'),
            ('brown', 'fox', 'jumps')
        ]
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result['doc1']), 3)
        self.assertEqual(result['doc1'], expected_trigrams)

    def test_generate_ngrams_for_docs_unigrams(self):
        documents = {
            'doc1': ['hello', 'world', 'hello']
        }
        result = self.unigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['doc1']), 3)
        self.assertIn(('hello',), result['doc1'])
        self.assertIn(('world',), result['doc1'])

    def test_generate_ngrams_for_docs_different_vocab_sizes(self):
        documents = {
            'doc1': ['a', 'b', 'a'],
            'doc2': ['x', 'y', 'z', 'x']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['doc1']), 2)
        self.assertIn(('a', 'b'), result['doc1'])
        self.assertIn(('b', 'a'), result['doc1'])

        self.assertEqual(len(result['doc2']), 3)
        self.assertIn(('x', 'y'), result['doc2'])
        self.assertIn(('y', 'z'), result['doc2'])
        self.assertIn(('z', 'x'), result['doc2'])

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

        self.assertEqual(len(result['doc1']), 5)
        self.assertIn(('the', 'cat'), result['doc1'])
        self.assertIn(('cat', 'sat'), result['doc1'])
        self.assertIn(('sat', 'on'), result['doc1'])
        self.assertIn(('on', 'the'), result['doc1'])
        self.assertIn(('the', 'mat'), result['doc1'])

    def test_generate_ngrams_for_docs_mixed_document_sizes(self):
        documents = {
            'short': ['a', 'b'],
            'medium': ['x', 'y', 'z', 'w'],
            'long': ['m', 'n', 'o', 'p', 'q', 'r', 's']
        }
        result = self.bigram_gen.generate_ngrams_for_docs(documents)

        self.assertEqual(len(result['short']), 1)
        self.assertEqual(len(result['medium']), 3)
        self.assertEqual(len(result['long']), 6)

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