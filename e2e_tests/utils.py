import csv
import os
import subprocess
from typing import Dict

MAIN_SCRIPT_PATH = "src.main"
PYTHON_EXECUTABLE = "python3"
EXPECTED_CSV_HEADERS = ["document1", "document2", "similarity_score"]

def run_pipeline(input_dir: str, output_file: str, encoding: str = None,
                 threshold: float = None) -> subprocess.CompletedProcess:
    command = [
        PYTHON_EXECUTABLE,
        "-m",
        MAIN_SCRIPT_PATH,
        "--input", input_dir,
        "--output", output_file
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


def validate_result(result, output_file, expected_rows_num=1):
    assert result.returncode == 0, f"Pipeline failed. Stderr: {result.stderr}"
    assert os.path.exists(output_file)

    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = list(reader)
    assert header == EXPECTED_CSV_HEADERS
    assert len(rows) >= expected_rows_num

    for row in rows:
        assert len(row) == 3
        try:
            float(row[2])
        except ValueError:
            assert False, f"Similarity score '{row[2]}' is not a valid number."
    return rows