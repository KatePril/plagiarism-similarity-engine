import unittest
from unittest.mock import patch
import sys

from main import parse_arg


class TestParseArg(unittest.TestCase):

    def test_required_file_directory_long_form(self):
        test_args = ['main.py', '--file-directory', '/path/to/files']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/path/to/files')

    def test_required_file_directory_short_form(self):
        test_args = ['main.py', '-fd', '/another/path']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/another/path')

    def test_missing_required_file_directory(self):
        test_args = ['main.py']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit):
                parse_arg()

    def test_encoding_default_value(self):
        test_args = ['main.py', '--file-directory', '/path']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'utf-8')

    def test_encoding_custom_value_long_form(self):
        test_args = ['main.py', '--file-directory', '/path', '--encoding', 'latin-1']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'latin-1')

    def test_encoding_custom_value_short_form(self):
        test_args = ['main.py', '-fd', '/path', '-e', 'ascii']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'ascii')

    def test_language_default_value(self):
        """Test that language defaults to 'english'"""
        test_args = ['main.py', '--file-directory', '/path']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'english')

    def test_language_custom_value_long_form(self):
        test_args = ['main.py', '--file-directory', '/path', '--language', 'spanish']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'spanish')

    def test_language_custom_value_short_form(self):
        test_args = ['main.py', '-fd', '/path', '-l', 'french']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'french')

    def test_all_arguments_together(self):
        test_args = [
            'main.py',
            '--file-directory', '/my/files',
            '--encoding', 'utf-16',
            '--language', 'german'
        ]
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/my/files')
            self.assertEqual(args.encoding, 'utf-16')
            self.assertEqual(args.language, 'german')

    def test_all_arguments_short_form(self):
        test_args = ['main.py', '-fd', '/data', '-e', 'iso-8859-1', '-l', 'italian']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/data')
            self.assertEqual(args.encoding, 'iso-8859-1')
            self.assertEqual(args.language, 'italian')

    def test_mixed_long_and_short_forms(self):
        test_args = ['main.py', '-fd', '/mixed', '--encoding', 'cp1252', '-l', 'portuguese']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/mixed')
            self.assertEqual(args.encoding, 'cp1252')
            self.assertEqual(args.language, 'portuguese')

    def test_file_directory_with_spaces(self):
        test_args = ['main.py', '--file-directory', '/path/with spaces/in it']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.file_directory, '/path/with spaces/in it')


if __name__ == '__main__':
    unittest.main()