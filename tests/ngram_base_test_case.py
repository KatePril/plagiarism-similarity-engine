import unittest

from src.ngrams_generator import NGramsGenerator


class NgramBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.bigram_gen = NGramsGenerator(n=2)
        self.trigram_gen = NGramsGenerator(n=3)
        self.unigram_gen = NGramsGenerator(n=1)
