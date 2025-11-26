import os
import tempfile
import shutil
import subprocess
import csv
from typing import Dict


MAIN_SCRIPT_PATH = "/src/main.py"
PYTHON_EXECUTABLE = "python3"
EXPECTED_CSV_HEADERS = ["document1", "document2", "similarity_score"]

def run_pipeline(input_dir: str, output_file: str, encoding: str = None,
                 threshold: float = None) -> subprocess.CompletedProcess:
    command = [
        PYTHON_EXECUTABLE,
        MAIN_SCRIPT_PATH,
        "--input-dir", input_dir,
        "--output-file", output_file
    ]
    if encoding:
        command.extend(["--encoding", encoding])
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])

    return subprocess.run(command, capture_output=True, text=True, check=False)


def create_input_files(temp_dir: str, files: Dict[str, str], encoding: str = 'utf-8'):
    for filename, content in files.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)


def test_pipeline_runs_successfully_for_small_files_csv():
    files_content = {
        "file_a.txt": "The quick brown fox jumps over the lazy dog.",
        "file_b.txt": "A quick brown fox leaps over a lazy dog.",
        "file_c.txt": "Completely different text content here."
    }
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, "results.csv")

    try:
        create_input_files(temp_dir, files_content, encoding='utf-8')
        result = run_pipeline(temp_dir, output_file)

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

    finally:
        shutil.rmtree(temp_dir)