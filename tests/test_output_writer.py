import csv
import unittest
from unittest.mock import mock_open, patch

from src.output_writer import OutputWriter
from src.similarity_evaluator import SimilarPair


class TestOutputWriter(unittest.TestCase):

    def setUp(self):
        self.writer = OutputWriter()
        self.output_file = "test_output.csv"

    def test_init_sets_correct_header(self):
        expected_header = ["document1", "document2", "similarity_score"]
        self.assertEqual(self.writer.header, expected_header)

    def test_write_results_with_empty_list(self):
        m = mock_open()
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, [])

        m.assert_called_once_with(self.output_file, "w", encoding="utf-8")
        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        self.assertIn("document1", written_content)
        self.assertIn("document2", written_content)
        self.assertIn("similarity_score", written_content)

    def test_write_results_with_single_pair(self):
        pair = SimilarPair(
            doc1_name="doc1.txt",
            doc2_name="doc2.txt",
            similarity_score=0.85
        )

        m = mock_open()
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, [pair])
        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        self.assertIn("document1", written_content)
        self.assertIn("doc1.txt", written_content)
        self.assertIn("doc2.txt", written_content)
        self.assertIn("0.85", written_content)

    def test_write_results_with_multiple_pairs(self):
        pairs = [
            SimilarPair("doc1.txt", "doc2.txt", 0.85),
            SimilarPair("doc3.txt", "doc4.txt", 0.92),
            SimilarPair("doc5.txt", "doc6.txt", 0.67)
        ]

        m = mock_open()
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, pairs)
        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        for pair in pairs:
            self.assertIn(pair.doc1_name, written_content)
            self.assertIn(pair.doc2_name, written_content)
            self.assertIn(str(pair.similarity_score), written_content)

    def test_write_results_creates_valid_csv(self):
        pairs = [
            SimilarPair("doc1.txt", "doc2.txt", 0.85),
            SimilarPair("doc3.txt", "doc4.txt", 0.92)
        ]

        written_data = []
        def write_side_effect(data):
            written_data.append(data)
            return len(data)

        m = mock_open()
        m().write.side_effect = write_side_effect
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, pairs)

        csv_content = "".join(written_data)
        lines = csv_content.strip().split('\n')
        reader = csv.reader(lines)
        rows = list(reader)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ["document1", "document2", "similarity_score"])
        self.assertEqual(rows[1], ["doc1.txt", "doc2.txt", "0.85"])
        self.assertEqual(rows[2], ["doc3.txt", "doc4.txt", "0.92"])

    def test_write_results_with_special_characters(self):
        pairs = [
            SimilarPair("doc with spaces.txt", "doc,with,commas.txt", 0.75),
            SimilarPair("doc\"with\"quotes.txt", "doc'with'apostrophes.txt", 0.88)
        ]

        m = mock_open()
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, pairs)
        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        self.assertIn("doc with spaces.txt", written_content)

    def test_write_results_with_different_output_paths(self):
        pairs = [SimilarPair("doc1.txt", "doc2.txt", 0.85)]
        different_paths = [
            "output.csv",
            "results/output.csv",
            "/tmp/output.csv"
        ]

        for path in different_paths:
            m = mock_open()
            with patch("builtins.open", m):
                self.writer.write_results(path, pairs)
            m.assert_called_once_with(path, "w", encoding="utf-8")

    def test_write_results_with_float_precision(self):
        pairs = [
            SimilarPair("doc1.txt", "doc2.txt", 0.123456789),
            SimilarPair("doc3.txt", "doc4.txt", 1.0),
            SimilarPair("doc5.txt", "doc6.txt", 0.0)
        ]

        m = mock_open()
        with patch("builtins.open", m):
            self.writer.write_results(self.output_file, pairs)
        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        self.assertIn("0.123456789", written_content)
        self.assertIn("1.0", written_content)
        self.assertIn("0.0", written_content)


if __name__ == "__main__":
    unittest.main()