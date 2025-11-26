import os
import tempfile
import shutil
import csv

from utils import create_input_files, run_pipeline, EXPECTED_CSV_HEADERS

def run_pipeline_successfully_with_provided_threshold_output(files_content, output, threshold):
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, output)

    try:
        create_input_files(temp_dir, files_content, encoding='utf-8')
        result = run_pipeline(temp_dir, output_file=output_file, threshold=threshold)

        assert result.returncode == 0, f"Pipeline failed. Stderr: {result.stderr}"
        assert os.path.exists(output_file)

        with open(output_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            rows = list(reader)

        assert header == EXPECTED_CSV_HEADERS
        assert len(rows) >= 1

        for row in rows:
            assert len(row) == 3
            try:
                float(row[2])
            except ValueError:
                assert False, f"Similarity score '{row[2]}' is not a valid number."

        default_output_file = os.path.join(temp_dir, "result.csv")
        run_pipeline(temp_dir, default_output_file)

        with open(default_output_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            rows1 = list(reader)
        assert len(rows) <= len(rows1)
    finally:
        shutil.rmtree(temp_dir)

def test_pipeline_runs_successfully_with_three_default_args_on_three_files():
    files_content = {
        "file_a.txt": "The quick brown fox jumps over the lazy dog in the meadow during sunset. "
                      "This classic pangram has been used in typography for decades to test fonts. "
                      "It contains every letter of the English alphabet at least once, making it "
                      "invaluable for designers who need to preview how different typefaces appear.",
        "file_b.txt": "The quick brown fox jumps over the lazy dog in the meadow during sunset. "
                      "This classic pangram has been used in typography for decades to test fonts. "
                      "It includes every letter of the English alphabet at least once, making it "
                      "invaluable for designers who need to preview how different typefaces look.",
        "file_c.txt": "Machine learning algorithms process vast amounts of data efficiently and accurately. "
                      "Neural networks represent powerful computational tools that mimic human brain structure. "
                      "Deep learning models have revolutionized artificial intelligence applications across "
                      "industries including healthcare, finance, and autonomous vehicle development."
    }
    run_pipeline_successfully_with_provided_threshold_output(files_content, "res.csv", 0.5)