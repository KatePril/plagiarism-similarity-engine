# Plagiarism similarity engine
An engine for plagiarism was developed in the scope of this project. Project core components:
- Input manager
- NTLK tokenizer
- NGrams generator
- MinHash generator
- LSH generator
- Similarity evaluator
- Output writer

## Prepare the project
Clone the repository
```shell
git clone https://github.com/KatePril/plagiarism-similarity-engine.git
```
Navigate to project dir
```shell
cd plagiarism-similarity-engine/
```
Create virtual environment
```shell
python -m venv .venv
```
Activate .venv<br>
_For Linux/MacOS_
```shell
source .venv/bin/activate
```
_For Windows_
```shell
.venv/Scripts/activate
```
Install the required dependencies
```shell
pip install -r requirements.txt
```
## Run the plagiarism similarity engine
Example cli command for running plagiarism similarity engine:
```shell
python -m src.main --input <path to directory files>
```
Supported parameters: <br>
- `--input` - A path to directory with files that need to be evaluated
- `--output` - A path to output file
- `--threshold` - A threshold for determining plagiarism
- `--encoding` - The encoding name
- `--language` - The language of the files in the input directory

## To run tests
Command to run unit tests:
```shell
python -m unittest discover -s tests -p "test_*.py"
```
Command to run integration tests:
```shell
pytest integration_tests/
```
Command to run end-to-end tests:
```shell
pytest e2e_tests/
```
