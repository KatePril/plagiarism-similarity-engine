from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download('stopwords', quiet=True)


class NtlkTokenizer:
    def __init__(self, lang: str = 'english'):
        self.lang = lang

    def tokenize(self, text: str) -> List[str]:
        stop_words = set(stopwords.words(self.lang))
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return filtered_tokens
