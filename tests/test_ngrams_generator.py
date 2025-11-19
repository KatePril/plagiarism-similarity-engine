import unittest

from src.ngrams_generator import NGramsGenerator


class TestNGramsGenerator(unittest.TestCase):

    def setUp(self):
        self.bigram_gen = NGramsGenerator(n=2)
        self.trigram_gen = NGramsGenerator(n=3)
        self.unigram_gen = NGramsGenerator(n=1)

    def test_generate_ngrams_bigrams(self):
        tokens = ['the', 'quick', 'brown', 'fox']
        expected = [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')]
        result = self.bigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_trigrams(self):
        tokens = ['the', 'quick', 'brown', 'fox']
        expected = [('the', 'quick', 'brown'), ('quick', 'brown', 'fox')]
        result = self.trigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_unigrams(self):
        tokens = ['the', 'quick', 'brown']
        expected = [('the',), ('quick',), ('brown',)]
        result = self.unigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_empty_list(self):
        tokens = []
        expected = []
        result = self.bigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_single_token(self):
        tokens = ['hello']
        expected = []
        result = self.bigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_exact_n_tokens(self):
        tokens = ['hello', 'world']
        expected = [('hello', 'world')]
        result = self.bigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_fewer_than_n_tokens(self):
        tokens = ['hello', 'world']
        expected = []
        result = self.trigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_generate_ngrams_with_duplicates(self):
        tokens = ['the', 'the', 'the']
        expected = [('the', 'the'), ('the', 'the')]
        result = self.bigram_gen._generate_ngrams(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_unique_tokens(self):
        tokens = ['the', 'quick', 'brown', 'fox']
        expected = 4
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_with_duplicates(self):
        tokens = ['the', 'cat', 'the', 'dog', 'the', 'cat']
        expected = 3
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_single_token(self):
        tokens = ['hello']
        expected = 1
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_empty_list(self):
        tokens = []
        expected = 0
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_all_same(self):
        tokens = ['apple', 'apple', 'apple', 'apple']
        expected = 1
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_get_vocab_size_mixed_tokens(self):
        tokens = ['a', 'b', 'c', 'a', 'b', 'c', 'd']
        expected = 4
        result = NGramsGenerator._get_vocab_size(tokens)
        self.assertEqual(result, expected)

    def test_apply_laplace_smoothing_basic(self):
        ngrams = [('the', 'cat'), ('the', 'dog'), ('the', 'cat')]
        vocab_size = 2
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        # ('the', 'cat') appears 2 times: (2+1)/(3+2) = 3/5 = 0.6
        # ('the', 'dog') appears 1 time: (1+1)/(3+2) = 2/5 = 0.4
        expected = {
            ('the', 'cat'): 0.6,
            ('the', 'dog'): 0.4
        }
        self.assertEqual(result, expected)

    def test_apply_laplace_smoothing_all_unique(self):
        ngrams = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        vocab_size = 3
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        # (1+1)/(3+3) = 2/6 = 1/3
        expected = {
            ('a', 'b'): 2 / 6,
            ('c', 'd'): 2 / 6,
            ('e', 'f'): 2 / 6
        }
        self.assertAlmostEqual(result[('a', 'b')], expected[('a', 'b')])
        self.assertAlmostEqual(result[('c', 'd')], expected[('c', 'd')])
        self.assertAlmostEqual(result[('e', 'f')], expected[('e', 'f')])

    def test_apply_laplace_smoothing_single_ngram(self):
        ngrams = [('hello', 'world'), ('hello', 'world')]
        vocab_size = 1
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        expected = {('hello', 'world'): 1.0}
        self.assertEqual(result, expected)

    def test_apply_laplace_smoothing_probabilities_sum(self):
        ngrams = [('a', 'b'), ('c', 'd'), ('a', 'b')]
        vocab_size = 2
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        total = sum(result.values())
        self.assertLessEqual(total, 1.0)

    def test_apply_laplace_smoothing_high_frequency(self):
        ngrams = [('the',)] * 100 + [('a',)] * 50
        vocab_size = 2
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        # ('the',) appears 100 times: (100+1)/(150+2) = 101/152
        # ('a',) appears 50 times: (50+1)/(150+2) = 51/152
        expected_the = 101 / 152
        expected_a = 51 / 152

        self.assertAlmostEqual(result[('the',)], expected_the, places=5)
        self.assertAlmostEqual(result[('a',)], expected_a, places=5)

    def test_apply_laplace_smoothing_empty_ngrams(self):
        ngrams = []
        vocab_size = 0
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        expected = {}
        self.assertEqual(result, expected)

    def test_apply_laplace_smoothing_zero_vocab_size(self):
        ngrams = [('word',), ('word',)]
        vocab_size = 0
        result = NGramsGenerator._apply_laplace_smoothing(ngrams, vocab_size)

        # (2+1)/(2+0) = 3/2 = 1.5 (can be > 1 with vocab_size=0)
        expected = {('word',): 1.5}
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()