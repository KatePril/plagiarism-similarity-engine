from collections import Counter


class NGramsGenerator:
    def __init__(self, n):
        self.n = n

    def generate_ngrams_for_docs(self, documents):
        ngrams_dict = {}
        for doc, tokens in documents.items():
            ngrams = self._generate_ngrams(tokens)
            vocab_size = self._get_vocab_size(tokens)
            ngrams_dict[doc] = self._apply_laplace_smoothing(ngrams, vocab_size)
        return ngrams_dict

    def _generate_ngrams(self, tokens):
        ngrams = [
            tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)
        ]
        return ngrams

    @staticmethod
    def _get_vocab_size(tokens):
        return len(set(tokens))

    @staticmethod
    def _apply_laplace_smoothing(ngrams, vocab_size):
        ngrams_counts = Counter(ngrams)
        smoothed_ngrams = {
            ngram: (count + 1) / (len(ngrams) + vocab_size) for ngram, count in ngrams_counts.items()
        }
        return smoothed_ngrams


