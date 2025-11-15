import nltk
import pytest
from src.input_manager import InputManager


@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


def test_read_files_basic(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello, WORLD!", encoding="utf-8")

    manager = InputManager()
    result = manager.read_files(str(tmp_path))

    assert "example.txt" in result
    assert result["example.txt"] == ["hello", "world"]


def test_read_files_ignores_non_txt(tmp_path):
    (tmp_path / "file1.txt").write_text("Some CONTENT!", encoding="utf-8")
    (tmp_path / "image.png").write_text("not text", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore this", encoding="utf-8")

    manager = InputManager()
    result = manager.read_files(str(tmp_path))

    assert len(result) == 1
    assert "file1.txt" in result


def test_multiple_files(tmp_path):
    (tmp_path / "a.txt").write_text("Hello there!", encoding="utf-8")
    (tmp_path / "b.txt").write_text("General Kenobi.", encoding="utf-8")

    manager = InputManager()
    result = manager.read_files(str(tmp_path))

    assert len(result) == 2
    assert set(result.keys()) == {"a.txt", "b.txt"}
    assert result["a.txt"] == ["hello", "there"]
    assert result["b.txt"] == ["general", "kenobi"]


def test_unicode_punctuation_is_not_cleaned_by_default(tmp_path):
    text = 'Hello “world”'
    (tmp_path / "u.txt").write_text(text, encoding="utf-8")

    manager = InputManager()
    result = manager.read_files(str(tmp_path))

    tokens = result["u.txt"]

    assert "hello" in tokens
    assert "world" in tokens
    assert any("“" in t or "”" in t for t in tokens)


def test_file_with_stopwords_removed(tmp_path):
    (tmp_path / "stop.txt").write_text("This is a test sentence.", encoding="utf-8")
    manager = InputManager()
    result = manager.read_files(str(tmp_path))
    tokens = result["stop.txt"]

    assert "test" in tokens
    assert "sentence" in tokens
    assert "this" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
