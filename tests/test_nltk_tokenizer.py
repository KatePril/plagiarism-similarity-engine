import pytest
import nltk

from src.ntlk_tokenizer import NtlkTokenizer


def test_basic_tokenization_and_stopword_removal():
    tokenizer = NtlkTokenizer(lang="english")
    text = "This is a simple test sentence."
    tokens = tokenizer.tokenize(text)
    expected_remaining = {'This', "simple", "test", "sentence", "."}

    assert set(tokens) == expected_remaining


def test_custom_language_spanish():
    tokenizer = NtlkTokenizer(lang="spanish")
    text = "Este es el texto de prueba."
    tokens = tokenizer.tokenize(text)

    assert "texto" in tokens or "texto" in [t.lower() for t in tokens]


def test_empty_string():
    tokenizer = NtlkTokenizer()
    assert tokenizer.tokenize("") == []


def test_punctuation_is_kept():
    tokenizer = NtlkTokenizer()
    text = "Hello, world!"
    tokens = tokenizer.tokenize(text)

    assert "," in tokens
    assert "!" in tokens
    assert "Hello" in tokens
    assert "world" in tokens


def test_all_tokens_removed_if_stopwords_only():
    tokenizer = NtlkTokenizer(lang="english")
    text = "the and is or but"
    tokens = tokenizer.tokenize(text)

    assert tokens == []
