import os
import string
import re
from typing import Dict

from src.ntlk_tokenizer import NtlkTokenizer


class InputManager:
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.tokenizer = NtlkTokenizer()

    def read_files(self, directory_path: str) -> Dict[str, str]:
        contents = {}
        for file in os.listdir(directory_path):
            if file.endswith(".txt"):
                full_path = os.path.join(directory_path, file)
                file_content = self._read_file(full_path)
                lowered_content = self._to_lower(file_content)
                cleaned_content = self._clean_punctuation(lowered_content)
                tokens = self.tokenizer.tokenize(cleaned_content)
                contents[file] = tokens
        return contents

    def _read_file(self, filepath: str) -> str:
        content = []
        with open(filepath, 'r', encoding=self.encoding) as f:
            for line in f:
                content.append(line.strip())
        return " ".join(content)

    @staticmethod
    def _to_lower(file_content: str) -> str:
        return file_content.lower()

    @staticmethod
    def _clean_punctuation(file_content: str, clean_unicode: bool = False) -> str:
        if clean_unicode:
            regex = r"[^\w\s]"
        else:
            regex = rf"[{re.escape(string.punctuation)}]"
        result = re.sub(regex, "", file_content)
        return result