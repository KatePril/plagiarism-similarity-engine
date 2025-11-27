import pytest
import numpy as np
from typing import Dict, List, Tuple

from src.min_hash_generator import MinHashGenerator, MinHash
from src.locality_sensitive_hashing import LshGenerator, LSH


class TestMinHashLSHIntegration:

    @pytest.fixture
    def sample_documents(self) -> Dict[str, List[Tuple[str, ...]]]:
        return {
            "doc1": [("the", "quick", "brown"), ("quick", "brown", "fox"), ("brown", "fox", "jumps")],
            "doc2": [("the", "quick", "brown"), ("quick", "brown", "fox"), ("brown", "fox", "jumps")],
            "doc3": [("a", "lazy", "dog"), ("lazy", "dog", "sleeps"), ("dog", "sleeps", "here")],
            "doc4": [("the", "fast", "brown"), ("fast", "brown", "fox"), ("brown", "fox", "runs")],
        }

    @pytest.fixture
    def identical_documents(self) -> Dict[str, List[Tuple[str, ...]]]:
        ngrams = [("hello", "world"), ("world", "test"), ("test", "data")]
        return {
            "doc_a": ngrams,
            "doc_b": ngrams,
            "doc_c": ngrams,
        }

    @pytest.fixture
    def disjoint_documents(self) -> Dict[str, List[Tuple[str, ...]]]:
        return {
            "doc_x": [("alpha", "beta"), ("beta", "gamma")],
            "doc_y": [("one", "two"), ("two", "three")],
            "doc_z": [("red", "blue"), ("blue", "green")],
        }

    def test_pipeline_basic(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)

        # Assert
        assert len(min_hashes) == len(sample_documents)
        assert all(isinstance(mh, MinHash) for mh in min_hashes.values())
        assert isinstance(lsh, LSH)
        assert lsh.num_bands == 16
        assert lsh.num_rows == 8
        assert len(lsh.signatures) == len(sample_documents)

    def test_similar_documents_are_detected(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc1", threshold=0.3)
        similar_ids = [doc_id for doc_id, _ in similar]

        # Assert
        assert "doc2" in similar_ids or "doc4" in similar_ids
        assert "doc3" not in similar_ids

    def test_dissimilar_documents_are_not_detected(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc3", threshold=0.5)
        similar_ids = [doc_id for doc_id, _ in similar]

        # Assert
        assert "doc1" not in similar_ids
        assert "doc2" not in similar_ids
        assert "doc4" not in similar_ids

    def test_identical_documents_high_similarity(self, identical_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(identical_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc_a", threshold=0.8)
        similar_ids = [doc_id for doc_id, _ in similar]

        # Assert
        assert len(similar) == 2
        assert "doc_b" in similar_ids
        assert "doc_c" in similar_ids
        for _, similarity in similar:
            assert similarity >= 0.8

    def test_threshold_filtering(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar_low = lsh.find_similar("doc1", threshold=0.1)
        similar_high = lsh.find_similar("doc1", threshold=0.7)

        # Assert
        assert len(similar_high) <= len(similar_low)
        for _, similarity in similar_high:
            assert similarity >= 0.7

    def test_query_returns_candidates(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        candidates = lsh.query(min_hashes["doc1"])

        # Assert
        assert "doc1" in candidates
        assert isinstance(candidates, set)

    def test_different_seeds_produce_different_results(self, sample_documents):
        # Arrange
        gen1 = MinHashGenerator(num_permutations=128, seed=42)
        gen2 = MinHashGenerator(num_permutations=128, seed=123)

        # Act
        min_hashes1 = gen1.generate_minhashes(sample_documents)
        min_hashes2 = gen2.generate_minhashes(sample_documents)
        sig1 = min_hashes1["doc1"].signature
        sig2 = min_hashes2["doc1"].signature

        # Assert
        assert not np.array_equal(sig1, sig2)

    def test_band_row_configuration(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=64, seed=42)
        lsh_gen1 = LshGenerator(num_bands=16, num_rows=4)
        lsh_gen2 = LshGenerator(num_bands=8, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh1 = lsh_gen1.generate_lsh(min_hashes)
        lsh2 = lsh_gen2.generate_lsh(min_hashes)
        similar1 = lsh1.find_similar("doc1", threshold=0.3)
        similar2 = lsh2.find_similar("doc1", threshold=0.3)

        # Assert
        assert isinstance(similar1, list)
        assert isinstance(similar2, list)

    def test_empty_document_handling(self):
        # Arrange
        empty_docs = {
            "empty1": [],
            "empty2": [],
        }
        min_hash_generator = MinHashGenerator(num_permutations=64, seed=42)
        lsh_generator = LshGenerator(num_bands=8, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(empty_docs)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("empty1", threshold=0.5)

        # Assert
        assert isinstance(similar, list)

    def test_large_number_of_documents(self):
        # Arrange
        docs = {}
        for i in range(100):
            base = [("word", str(i)), (str(i), "test"), ("test", "data")]
            if i % 10 == 0:
                base.extend([("common", "pattern"), ("pattern", "here")])
            docs[f"doc_{i}"] = base
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(docs)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc_0", threshold=0.3)

        # Assert
        assert len(lsh.signatures) == 100
        assert isinstance(similar, list)

    def test_error_handling_missing_document(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)

        # Assert
        with pytest.raises(ValueError, match="not found in index"):
            lsh.find_similar("nonexistent_doc", threshold=0.5)

    def test_permutation_mismatch_error(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=64, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act & Assert
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        with pytest.raises(ValueError, match="permutations"):
            lsh_generator.generate_lsh(min_hashes)

    def test_results_are_sorted_by_similarity(self, sample_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(sample_documents)
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc1", threshold=0.1)
        similarities = [sim for _, sim in similar]

        # Assert
        assert similarities == sorted(similarities, reverse=True)

    def test_jaccard_similarity_consistency(self, identical_documents):
        # Arrange
        min_hash_generator = MinHashGenerator(num_permutations=128, seed=42)
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        min_hashes = min_hash_generator.generate_minhashes(identical_documents)
        direct_similarity = min_hashes["doc_a"].jaccard_similarity(min_hashes["doc_b"])
        lsh = lsh_generator.generate_lsh(min_hashes)
        similar = lsh.find_similar("doc_a", threshold=0.0)
        lsh_similarity = None
        for doc_id, sim in similar:
            if doc_id == "doc_b":
                lsh_similarity = sim
                break

        # Assert
        assert lsh_similarity is not None
        assert abs(direct_similarity - lsh_similarity) < 0.01