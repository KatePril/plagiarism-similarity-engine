import unittest
import tempfile
import os

from src.input_manager import InputManager


class InputManagerTestCase(unittest.TestCase):

    def setUp(self):
        self.input_manager = InputManager()

    @staticmethod
    def _create_temp_file(content: str, encoding: str):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp.name
        tmp.close()

        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
        return tmp_path

    def test_to_lower_capital_letters_in_str(self):
        text = "Hello World"
        expected = 'hello world'
        actual = self.input_manager._to_lower(text)
        self.assertEqual(expected, actual)

    def test_to_lower_all_capital_letters_in_str(self):
        text = "HELLO WORLD"
        expected = 'hello world'
        actual = self.input_manager._to_lower(text)
        self.assertEqual(expected, actual)

    def test_to_lower_no_capital_letters_in_str(self):
        expected = 'hello world'
        actual = self.input_manager._to_lower(expected)
        self.assertEqual(expected, actual)

    def test_clean_punctuation(self):
        text = "Hello, World!#@"
        expected = 'Hello World'
        actual = self.input_manager._clean_punctuation(text)
        self.assertEqual(expected, actual)

    def test_clean_punctuation_clean_unicode(self):
        text = "Hello —“World”"
        expected = 'Hello World'
        actual = self.input_manager._clean_punctuation(text, clean_unicode=True)
        self.assertEqual(expected, actual)

    def test_clean_punctuation_no_punctuation(self):
        expected = 'Hello World'
        actual = self.input_manager._clean_punctuation(expected)
        self.assertEqual(expected, actual)

    def test_clean_punctuation_clean_unicode_no_punctuation(self):
        expected = 'Hello World'
        actual = self.input_manager._clean_punctuation(expected, clean_unicode=True)
        self.assertEqual(expected, actual)

    def test_read_utf8_file(self):
        text = "Hello UTF-8 Привіт"
        path = self._create_temp_file(text, "utf-8")

        result = self.input_manager._read_file(path)
        self.assertEqual(result, text.replace("\n", "").replace("\r", ""))

        os.remove(path)

    def test_read_utf16_file(self):
        text = "Hello UTF-16 こんにちは"
        path = self._create_temp_file(text, "utf-16")
        input_manager = InputManager("utf-16")

        result = input_manager._read_file(path)
        self.assertEqual(result, text.replace("\n", "").replace("\r", ""))

        os.remove(path)

    def test_read_latin1_file(self):
        text = "Café crème - latin1"
        path = self._create_temp_file(text, "latin-1")
        input_manager = InputManager("latin-1")

        result = input_manager._read_file(path)
        self.assertEqual(result, text.replace("\n", "").replace("\r", ""))

        os.remove(path)

    def test_read_multiline_file(self):
        text = "line1\nline2\nline3"
        path = self._create_temp_file(text, "utf-8")

        result = self.input_manager._read_file(path)
        self.assertEqual(result, "line1 line2 line3")

        os.remove(path)


if __name__ == '__main__':
    unittest.main()
