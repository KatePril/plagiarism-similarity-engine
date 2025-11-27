import os
import tempfile
import shutil

from utils import create_input_files, run_pipeline, validate_result


def run_pipeline_successfully_with_provided_encoding(files_content, encoding):
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, "results.csv")

    try:
        create_input_files(temp_dir, files_content, encoding=encoding)
        result = run_pipeline(temp_dir, output_file, encoding=encoding)
        validate_result(result, output_file)
    finally:
        shutil.rmtree(temp_dir)

def test_pipeline_runs_successfully_with_provided_encoding_on_three_files():
    files_content = {
        "file_a.txt": "The café on the corner serves excellent coffee and pastries every morning. "
                       "Students often gather there to discuss their résumés and career prospects. "
                       "Despite being naïve about business, the owner has created a welcoming atmosphere "
                       "that attracts customers from diverse backgrounds and cultures.",
        "file_b.txt": "The café on the corner serves excellent coffee and pastries every morning. "
                      "Students often gather there to discuss their résumés and career prospects. "
                      "Despite being naïve about business, the owner has created a welcoming atmosphere "
                      "that attracts customers from diverse backgrounds and cultures.",
        "file_c.txt": "The café on the corner serves excellent coffee and pastries every morning. "
                      "Students often gather there to discuss their résumés and career prospects. "
                      "Despite being naïve about business, the owner has created a welcoming atmosphere "
                      "that attracts customers from diverse backgrounds and cultures."
    }
    run_pipeline_successfully_with_provided_encoding(files_content, "latin-1")