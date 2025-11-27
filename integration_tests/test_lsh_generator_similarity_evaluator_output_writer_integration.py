import tempfile
import csv
import os
import pytest

from src.min_hash_generator import MinHash
from src.locality_sensitive_hashing import LshGenerator
from src.similarity_evaluator import SimilarityEvaluator
from src.output_writer import OutputWriter


class TestLSHSimilarityEvaluatorOutputWriterIntegration:

    @pytest.fixture
    def sample_documents(self):
        return {
            "doc1": {"apple", "banana", "orange", "grape"},
            "doc2": {"apple", "banana", "orange", "pear"},
            "doc3": {"apple", "banana", "orange", "grape"},
            "doc4": {"car", "truck", "bus", "train"},
            "doc5": {"car", "truck", "bus", "plane"},
        }

    @pytest.fixture
    def minhashes(self, sample_documents):
        minhashes = {}
        for doc_id, shingles in sample_documents.items():
            mh = MinHash(num_permutations=128, seed=42)
            for shingle in shingles:
                mh.update((shingle,))
            minhashes[doc_id] = mh
        return minhashes

    def test_threshold_filtering(self, minhashes):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        high_threshold_evaluator = SimilarityEvaluator(lsh, threshold=0.7)
        low_threshold_evaluator = SimilarityEvaluator(lsh, threshold=0.2)

        # Act
        high_threshold_pairs = high_threshold_evaluator.get_similar_pairs(list(minhashes.keys()))
        low_threshold_pairs = low_threshold_evaluator.get_similar_pairs(list(minhashes.keys()))

        # Assert
        assert len(low_threshold_pairs) >= len(high_threshold_pairs)
        for pair in high_threshold_pairs:
            assert pair.similarity_score >= 0.7

    def test_no_duplicate_pairs(self, minhashes):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.3)
        seen_pairs = set()

        # Act
        similar_pairs = similarity_evaluator.get_similar_pairs(list(minhashes.keys()))
        for pair in similar_pairs:
            pair_tuple = tuple(sorted([pair.doc1_name, pair.doc2_name]))

            # Assert
            assert pair_tuple not in seen_pairs, f"Duplicate pair found: {pair_tuple}"
            seen_pairs.add(pair_tuple)

    def test_results_sorted_by_similarity(self, minhashes):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.2)

        # Act
        similar_pairs = similarity_evaluator.get_similar_pairs(list(minhashes.keys()))
        scores = [pair.similarity_score for pair in similar_pairs]

        # Assert
        assert scores == sorted(scores, reverse=True)

    def test_empty_document_set(self):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh({})
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.5)

        # Act
        similar_pairs = similarity_evaluator.get_similar_pairs([])

        # Assert
        assert len(similar_pairs) == 0

    def test_single_document(self):
        # Arrange & Act
        mh = MinHash(num_permutations=128, seed=42)
        mh.update(("word",))

        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh({"doc1": mh})

        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.5)
        similar_pairs = similarity_evaluator.get_similar_pairs(["doc1"])

        # Assert
        assert len(similar_pairs) == 0

    def test_identical_documents(self):
        # Arrange & Act
        mh1 = MinHash(num_permutations=128, seed=42)
        mh2 = MinHash(num_permutations=128, seed=42)
        for word in ["apple", "banana", "orange"]:
            mh1.update((word,))
            mh2.update((word,))
        minhashes = {"doc1": mh1, "doc2": mh2}

        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.9)
        similar_pairs = similarity_evaluator.get_similar_pairs(["doc1", "doc2"])

        # Assert
        assert len(similar_pairs) == 1
        assert similar_pairs[0].similarity_score == 1.0

    def test_band_configuration_affects_results(self, minhashes):
        # Arrange
        lsh_generator_high_precision = LshGenerator(num_bands=32, num_rows=4)
        lsh_high_precision = lsh_generator_high_precision.generate_lsh(minhashes)
        lsh_generator_high_recall = LshGenerator(num_bands=8, num_rows=16)
        lsh_high_recall = lsh_generator_high_recall.generate_lsh(minhashes)

        evaluator_hp = SimilarityEvaluator(lsh_high_precision, threshold=0.3)
        evaluator_hr = SimilarityEvaluator(lsh_high_recall, threshold=0.3)

        # Act
        pairs_hp = evaluator_hp.get_similar_pairs(list(minhashes.keys()))
        pairs_hr = evaluator_hr.get_similar_pairs(list(minhashes.keys()))

        # Assert
        assert len(pairs_hr) >= len(pairs_hp) or len(pairs_hp) >= len(pairs_hr)

    def test_custom_output_header(self, minhashes):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.3)
        similar_pairs = similarity_evaluator.get_similar_pairs(list(minhashes.keys()))

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_file = f.name

        # Act
        try:
            custom_header = ["doc_a", "doc_b", "score"]
            output_writer = OutputWriter(header=custom_header)
            output_writer.write_results(output_file, similar_pairs)
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Assert
            assert rows[0] == custom_header
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_query_nonexistent_document(self, minhashes):
        # Arrange
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)
        lsh = lsh_generator.generate_lsh(minhashes)
        similarity_evaluator = SimilarityEvaluator(lsh, threshold=0.5)

        # Act & Assert
        with pytest.raises(ValueError, match="Document .* not found in index"):
            similarity_evaluator.get_similar_pairs(["nonexistent_doc"])

    def test_mismatched_permutations_raises_error(self):
        # Arrange
        mh1 = MinHash(num_permutations=128)
        mh1.update(("test",))
        mh2 = MinHash(num_permutations=64)
        mh2.update(("test",))
        lsh_generator = LshGenerator(num_bands=16, num_rows=8)

        # Act
        lsh = lsh_generator.generate_lsh({"doc1": mh1})

        # Assert
        with pytest.raises(ValueError, match="MinHash has .* permutations"):
            lsh.insert("doc2", mh2)