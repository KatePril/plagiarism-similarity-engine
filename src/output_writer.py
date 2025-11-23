import csv
from typing import List

from src.similarity_evaluator import SimilarPair

class OutputWriter:
    def __init__(self, header=None):
        if header:
            self.header = header
        else:
            self.header = ["document1", "document2", "similarity_score"]

    def write_results(self, output_file: str, results: List['SimilarPair']):
        data = []
        for item in results:
            data.append([item.doc1_name, item.doc2_name, item.similarity_score])
        with open(output_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(data)
