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