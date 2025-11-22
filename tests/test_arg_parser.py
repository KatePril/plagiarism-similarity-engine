import unittest
from unittest.mock import patch
import sys

from src.main import parse_arg


class TestParseArg(unittest.TestCase):

    def test_required_input_long_form(self):
        test_args = ['main.py', '--input', '/path/to/input.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, '/path/to/input.txt')

    def test_required_input_short_form(self):
        test_args = ['main.py', '-i', 'data.csv']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, 'data.csv')

    def test_missing_required_input(self):
        test_args = ['main.py']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit):
                parse_arg()

    def test_output_default_value(self):
        test_args = ['main.py', '--input', 'input.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.output, 'report.csv')

    def test_output_custom_value_long_form(self):
        test_args = ['main.py', '--input', 'input.txt', '--output', 'results.json']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.output, 'results.json')

    def test_output_custom_value_short_form(self):
        test_args = ['main.py', '-i', 'input.txt', '-o', 'output.xml']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.output, 'output.xml')

    def test_threshold_default_value(self):
        test_args = ['main.py', '--input', 'data.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, 0.75)

    def test_threshold_custom_value_long_form(self):
        test_args = ['main.py', '--input', 'data.txt', '--threshold', '0.9']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, 0.9)

    def test_threshold_custom_value_short_form(self):
        test_args = ['main.py', '-i', 'data.txt', '-t', '0.5']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, 0.5)

    def test_threshold_type_is_float(self):
        test_args = ['main.py', '--input', 'data.txt', '--threshold', '1']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertIsInstance(args.threshold, float)
            self.assertEqual(args.threshold, 1.0)

    def test_threshold_invalid_type(self):
        test_args = ['main.py', '--input', 'data.txt', '--threshold', 'invalid']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit):
                parse_arg()

    def test_encoding_default_value(self):
        test_args = ['main.py', '--input', 'file.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'utf-8')

    def test_encoding_custom_value_long_form(self):
        test_args = ['main.py', '--input', 'file.txt', '--encoding', 'latin-1']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'latin-1')

    def test_encoding_custom_value_short_form(self):
        test_args = ['main.py', '-i', 'file.txt', '-e', 'ascii']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.encoding, 'ascii')

    def test_language_default_value(self):
        test_args = ['main.py', '--input', 'file.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'english')

    def test_language_custom_value_long_form(self):
        test_args = ['main.py', '--input', 'file.txt', '--language', 'spanish']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'spanish')

    def test_language_custom_value_short_form(self):
        test_args = ['main.py', '-i', 'file.txt', '-l', 'french']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.language, 'french')

    def test_all_arguments_together(self):
        test_args = [
            'main.py',
            '--input', 'data.csv',
            '--output', 'analysis.txt',
            '--threshold', '0.85',
            '--encoding', 'utf-16',
            '--language', 'german'
        ]
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, 'data.csv')
            self.assertEqual(args.output, 'analysis.txt')
            self.assertEqual(args.threshold, 0.85)
            self.assertEqual(args.encoding, 'utf-16')
            self.assertEqual(args.language, 'german')

    def test_all_arguments_short_form(self):
        test_args = ['main.py', '-i', 'input.json', '-o', 'out.csv', '-t', '0.6', '-e', 'iso-8859-1', '-l', 'italian']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, 'input.json')
            self.assertEqual(args.output, 'out.csv')
            self.assertEqual(args.threshold, 0.6)
            self.assertEqual(args.encoding, 'iso-8859-1')
            self.assertEqual(args.language, 'italian')

    def test_mixed_long_and_short_forms(self):
        test_args = ['main.py', '-i', 'mixed.txt', '--output', 'result.csv', '-t', '0.8', '--encoding', 'cp1252', '-l',
                     'portuguese']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, 'mixed.txt')
            self.assertEqual(args.output, 'result.csv')
            self.assertEqual(args.threshold, 0.8)
            self.assertEqual(args.encoding, 'cp1252')
            self.assertEqual(args.language, 'portuguese')

    def test_input_with_spaces(self):
        test_args = ['main.py', '--input', '/path/with spaces/file.txt']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.input, '/path/with spaces/file.txt')

    def test_threshold_boundary_values(self):
        test_args = ['main.py', '--input', 'file.txt', '--threshold', '0.0']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, 0.0)

        test_args = ['main.py', '--input', 'file.txt', '--threshold', '1.0']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, 1.0)

        test_args = ['main.py', '--input', 'file.txt', '--threshold', '-0.5']
        with patch.object(sys, 'argv', test_args):
            args = parse_arg()
            self.assertEqual(args.threshold, -0.5)


if __name__ == '__main__':
    unittest.main()