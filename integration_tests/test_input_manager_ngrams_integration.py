import unittest
import tempfile
import shutil
import os

from src.input_manager import InputManager
from src.ngrams_generator import NGramsGenerator


class TestInputManagerNGramsIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.encoding = 'utf-8'
        self.language = 'english'

        self.test_files = {
            'document1.txt': 'Hello World! This is a test document.',
            'document2.txt': 'Natural Language Processing is fascinating.\nIt involves many techniques.',
            'document3.txt': 'Machine learning, deep learning, and AI are related fields.',
            'empty.txt': '',
            'single_word.txt': 'Hello',
            'punctuation.txt': 'Test!!! Question? Comma, semicolon; period.',
            'unicode.txt': 'Café résumé naïve—test',
        }

        for filename, content in self.test_files.items():
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w', encoding=self.encoding) as f:
                f.write(content)

        with open(os.path.join(self.test_dir, 'ignore.csv'), 'w') as f:
            f.write('data,data')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_pipeline_basic(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        self.assertEqual(len(files_tokens), 7)
        self.assertNotIn('ignore.csv', files_tokens)
        self.assertEqual(len(ngrams), 7)
        for filename in self.test_files.keys():
            self.assertIn(filename, ngrams)
        for doc_ngrams in ngrams.values():
            self.assertIsInstance(doc_ngrams, list)
            if doc_ngrams:
                self.assertIsInstance(doc_ngrams[0], tuple)

    def test_pipeline_with_trigrams(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=3)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        for doc_name, doc_ngrams in ngrams.items():
            if doc_ngrams:
                self.assertEqual(len(doc_ngrams[0]), 3)
                for ngram in doc_ngrams:
                    self.assertEqual(len(ngram), 3)

    def test_pipeline_with_unigrams(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=1)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        for filename, tokens in files_tokens.items():
            self.assertEqual(len(ngrams[filename]), len(tokens))

    def test_punctuation_cleaning_integration(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        punctuation_tokens = files_tokens['punctuation.txt']
        punctuation_ngrams = ngrams['punctuation.txt']
        for token in punctuation_tokens:
            self.assertNotIn('!', token)
            self.assertNotIn('?', token)
            self.assertNotIn(',', token)
            self.assertNotIn(';', token)
            self.assertNotIn('.', token)
        if len(punctuation_tokens) >= 2:
            self.assertGreater(len(punctuation_ngrams), 0)

    def test_case_normalization_integration(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        doc1_tokens = files_tokens['document1.txt']
        doc1_ngrams = ngrams['document1.txt']
        for token in doc1_tokens:
            self.assertEqual(token, token.lower())
        for ngram in doc1_ngrams:
            for word in ngram:
                self.assertEqual(word, word.lower())

    def test_empty_file_handling(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        self.assertIn('empty.txt', files_tokens)
        self.assertEqual(len(files_tokens['empty.txt']), 0)
        self.assertEqual(len(ngrams['empty.txt']), 0)

    def test_single_word_file_with_bigrams(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        single_word_tokens = files_tokens['single_word.txt']
        single_word_ngrams = ngrams['single_word.txt']
        self.assertEqual(len(single_word_tokens), 1)
        self.assertEqual(len(single_word_ngrams), 0)

    def test_different_encodings(self):
        # Arrange
        test_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(test_dir, 'latin1.txt')
            with open(filepath, 'w', encoding='latin-1') as f:
                f.write('Testing encoding')
            input_manager = InputManager(encoding='latin-1', language='english')
            ngrams_generator = NGramsGenerator(n=2)

            # Act
            files_tokens = input_manager.read_files(test_dir)
            ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

            # Assert
            self.assertIn('latin1.txt', files_tokens)
            self.assertGreater(len(ngrams['latin1.txt']), 0)
        finally:
            shutil.rmtree(test_dir)

    def test_ngram_size_variations(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)

        # Act & Assert
        for n in [1, 2, 3, 4, 5]:
            ngrams_generator = NGramsGenerator(n=n)
            files_tokens = input_manager.read_files(self.test_dir)
            ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

            doc2_ngrams = ngrams['document2.txt']
            if doc2_ngrams:
                for ngram in doc2_ngrams:
                    self.assertEqual(len(ngram), n)

    def test_multiline_file_handling(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams = ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        doc2_tokens = files_tokens['document2.txt']
        doc2_ngrams = ngrams['document2.txt']
        self.assertGreater(len(doc2_tokens), 0)
        self.assertGreater(len(doc2_ngrams), 0)

    def test_vocab_size_calculation(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        doc1_tokens = files_tokens['document1.txt']
        vocab_size = ngrams_generator._get_vocab_size(doc1_tokens)

        # Assert
        unique_tokens = set(doc1_tokens)
        self.assertEqual(vocab_size, len(unique_tokens))

    def test_laplace_smoothing_integration(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams_dict = ngrams_generator.generate_ngrams_for_docs(files_tokens)
        doc1_tokens = files_tokens['document1.txt']
        doc1_ngrams = ngrams_dict['document1.txt']
        vocab_size = ngrams_generator._get_vocab_size(doc1_tokens)
        smoothed = ngrams_generator._apply_laplace_smoothing(doc1_ngrams, vocab_size)

        # Assert
        if doc1_ngrams:
            self.assertIsInstance(smoothed, dict)
            for prob in smoothed.values():
                self.assertGreater(prob, 0)
            for prob in smoothed.values():
                self.assertLess(prob, 1)

    def test_nonexistent_directory(self):
        # Arrange
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        NGramsGenerator(n=2)

        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            input_manager.read_files('/nonexistent/directory/path')

    def test_directory_with_subdirectories(self):
        # Arrange
        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir)
        with open(os.path.join(subdir, 'nested.txt'), 'w') as f:
            f.write('This should be ignored')
        input_manager = InputManager(encoding=self.encoding, language=self.language)
        ngrams_generator = NGramsGenerator(n=2)

        # Act
        files_tokens = input_manager.read_files(self.test_dir)
        ngrams_generator.generate_ngrams_for_docs(files_tokens)

        # Assert
        filenames = list(files_tokens.keys())
        self.assertNotIn('nested.txt', filenames)
        self.assertNotIn('subdir/nested.txt', filenames)

        shutil.rmtree(subdir)


if __name__ == '__main__':
    unittest.main()