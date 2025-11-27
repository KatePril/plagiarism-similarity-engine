import argparse

from src.input_manager import InputManager
from src.ngrams_generator import NGramsGenerator
from src.min_hash_generator import MinHashGenerator
from src.locality_sensitive_hashing import LshGenerator
from src.similarity_evaluator import SimilarityEvaluator
from src.output_writer import OutputWriter


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, type=str,
                        help='A path to directory with files that need to be evaluated')
    parser.add_argument('--output', '-o', default='report.csv', type=str, help='A path to output file')
    parser.add_argument('--threshold', '-t', default=0.7, type=float,
                        help='A threshold for determining plagiarism')
    parser.add_argument('--encoding', '-e', default='utf-8', type=str, help='The encoding name')
    parser.add_argument('--language', '-l', default='english', type=str,
                        help='The language of the files in the input directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    input_manager = InputManager(encoding=args.encoding, language=args.language)
    files_tokens = input_manager.read_files(args.input)
    filenames = list(files_tokens.keys())

    ngrams_generator = NGramsGenerator(3)
    ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)
    min_hash_generator = MinHashGenerator()
    min_hash = min_hash_generator.generate_minhashes(ngrams)

    lsh_generator = LshGenerator(num_bands=16, num_rows=8)
    lsh = lsh_generator.generate_lsh(min_hash)
    similarity_evaluator = SimilarityEvaluator(lsh, threshold=args.threshold)
    similar_pairs = similarity_evaluator.get_similar_pairs(filenames)

    output_writer = OutputWriter()
    output_writer.write_results(args.output, similar_pairs)
