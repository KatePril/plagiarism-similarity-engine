import argparse

from src.input_manager import InputManager
from src.ngrams_generator import NGramsGenerator


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, type=str)
    parser.add_argument('--output', '-o', default='report.csv', type=str)
    parser.add_argument('--threshold', '-t', default=0.75, type=float)
    parser.add_argument('--encoding', '-e', default='utf-8', type=str)
    parser.add_argument('--language', '-l', default='english', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    input_manager = InputManager(encoding=args.encoding, language=args.language)
    files_tokens = input_manager.read_files(args.file_directory)
    ngrams_generator = NGramsGenerator(3)
    ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)