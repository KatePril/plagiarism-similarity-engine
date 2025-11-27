import pytest
import numpy as np

from src.ngrams_generator import NGramsGenerator
from src.min_hash_generator import MinHashGenerator


class TestNGramsMinHashIntegration:

    @pytest.fixture
    def sample_documents(self):
        return {
            'doc1': ['the', 'quick', 'brown', 'fox', 'jumps'],
            'doc2': ['the', 'quick', 'brown', 'fox', 'runs'],
            'doc3': ['a', 'lazy', 'dog', 'sleeps']
        }

    @pytest.fixture
    def identical_documents(self):
        return {
            'doc1': ['hello', 'world', 'hello', 'world'],
            'doc2': ['hello', 'world', 'hello', 'world']
        }

    @pytest.fixture
    def completely_different_documents(self):
        return {
            'doc1': ['apple', 'banana', 'cherry'],
            'doc2': ['dog', 'elephant', 'fox']
        }

    def test_full_pipeline_basic(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen = MinHashGenerator(num_permutations=128)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)

        # Assert
        assert len(minhashes) == 3
        assert all(doc in minhashes for doc in ['doc1', 'doc2', 'doc3'])
        assert all(hasattr(mh, 'signature') for mh in minhashes.values())
        assert len(ngrams) == 3
        assert all(doc in ngrams for doc in ['doc1', 'doc2', 'doc3'])
        assert all(isinstance(ng, tuple) for ngrams_list in ngrams.values() for ng in ngrams_list)

    def test_similar_documents_high_similarity(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=3)
        minhash_gen = MinHashGenerator(num_permutations=256)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        similarity = minhashes['doc1'].jaccard_similarity(minhashes['doc2'])

        # Assert
        assert similarity > 0.3, f"Expected similarity > 0.3, got {similarity}"

    def test_identical_documents_perfect_similarity(self, identical_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen = MinHashGenerator(num_permutations=128)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(identical_documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        similarity = minhashes['doc1'].jaccard_similarity(minhashes['doc2'])

        # Assert
        assert similarity == 1.0, f"Expected perfect similarity 1.0, got {similarity}"

    def test_different_documents_low_similarity(self, completely_different_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen = MinHashGenerator(num_permutations=256)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(completely_different_documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        similarity = minhashes['doc1'].jaccard_similarity(minhashes['doc2'])

        # Assert
        assert similarity < 0.2, f"Expected low similarity < 0.2, got {similarity}"

    def test_different_ngram_sizes(self):
        # Arrange
        documents = {
            'doc1': ['a', 'b', 'c', 'd', 'e'],
            'doc2': ['a', 'b', 'c', 'd', 'f']
        }
        similarities = {}
        for n in [1, 2, 3, 4]:
            ngrams_gen = NGramsGenerator(n=n)
            minhash_gen = MinHashGenerator(num_permutations=128)

            # Act
            ngrams = ngrams_gen.generate_ngrams_for_docs(documents)
            minhashes = minhash_gen.generate_minhashes(ngrams)
            similarities[n] = minhashes['doc1'].jaccard_similarity(minhashes['doc2'])

            # Assert
            assert all(len(ng) == n for ngrams_list in ngrams.values() for ng in ngrams_list)
        assert all(0 <= sim <= 1 for sim in similarities.values())

    def test_empty_documents_handling(self):
        # Arrange
        documents = {
            'doc1': ['a', 'b'],
            'doc2': ['a', 'b', 'c']
        }
        ngrams_gen = NGramsGenerator(n=3)
        minhash_gen = MinHashGenerator(num_permutations=128)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)

        # Assert
        assert len(ngrams['doc1']) == 0
        assert len(ngrams['doc2']) == 1
        assert 'doc1' in minhashes
        assert 'doc2' in minhashes

    def test_seed_reproducibility(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen1 = MinHashGenerator(num_permutations=128, seed=42)
        minhash_gen2 = MinHashGenerator(num_permutations=128, seed=42)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        minhashes1 = minhash_gen1.generate_minhashes(ngrams)
        minhashes2 = minhash_gen2.generate_minhashes(ngrams)

        # Assert
        for doc in sample_documents.keys():
            assert np.array_equal(minhashes1[doc].signature, minhashes2[doc].signature)

    def test_different_seeds_different_signatures(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen1 = MinHashGenerator(num_permutations=128, seed=42)
        minhash_gen2 = MinHashGenerator(num_permutations=128, seed=99)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        minhashes1 = minhash_gen1.generate_minhashes(ngrams)
        minhashes2 = minhash_gen2.generate_minhashes(ngrams)

        # Assert
        for doc in sample_documents.keys():
            assert not np.array_equal(minhashes1[doc].signature, minhashes2[doc].signature)

    def test_permutation_count_consistency(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        for num_perms in [64, 128, 256]:
            minhash_gen = MinHashGenerator(num_permutations=num_perms)

            # Act
            minhashes = minhash_gen.generate_minhashes(ngrams)

            # Assert
            for minhash in minhashes.values():
                assert len(minhash.signature) == num_perms

    def test_large_document_set(self):
        # Arrange
        documents = {
            f'doc{i}': ['word' + str(j) for j in range(i, i + 10)]
            for i in range(10)
        }
        ngrams_gen = NGramsGenerator(n=3)
        minhash_gen = MinHashGenerator(num_permutations=128)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        sim_adjacent = minhashes['doc0'].jaccard_similarity(minhashes['doc1'])
        sim_distant = minhashes['doc0'].jaccard_similarity(minhashes['doc9'])

        # Assert
        assert len(minhashes) == 10
        assert sim_adjacent > sim_distant

    def test_special_characters_in_tokens(self):
        # Arrange
        documents = {
            'doc1': ['hello!', 'world?', '#test', '@user'],
            'doc2': ['hello!', 'world?', '#test', '@admin']
        }
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen = MinHashGenerator(num_permutations=128)

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        similarity = minhashes['doc1'].jaccard_similarity(minhashes['doc2'])

        # Assert
        assert 0 <= similarity <= 1

    def test_all_pairwise_similarities(self, sample_documents):
        # Arrange
        ngrams_gen = NGramsGenerator(n=2)
        minhash_gen = MinHashGenerator(num_permutations=256)
        docs = list(sample_documents.keys())
        similarity_matrix = {}

        # Act
        ngrams = ngrams_gen.generate_ngrams_for_docs(sample_documents)
        minhashes = minhash_gen.generate_minhashes(ngrams)
        for i, doc1 in enumerate(docs):
            for doc2 in docs[i:]:
                sim = minhashes[doc1].jaccard_similarity(minhashes[doc2])
                similarity_matrix[(doc1, doc2)] = sim

                # Assert
                if doc1 == doc2:
                    assert sim == 1.0
                else:
                    assert 0 <= sim < 1.0
        assert len(similarity_matrix) == len(docs) * (len(docs) + 1) // 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])