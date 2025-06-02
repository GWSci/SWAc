import unittest
import swacmod_run
import argparse
import swacmod.utils as u

class Test_Input_File_Arguments(unittest.TestCase):
    def test_not_specifying_input_file_raises_exception(self):
        parser = make_parser()
        args = parser.parse_args(args=[])
        with self.assertRaisesRegex(u.ArgumentError,'No input file specified. Use "-i" or "--input_yml" to specify the path to "input.yml"'):
            swacmod_run.check_arguments(args)
    
    def test_specifying_input_file_does_not_raise_exception(self):
        parser = make_parser()
        args = parser.parse_args(args=["-i", "input.yml"])
        try:
            swacmod_run.check_arguments(args)
        except Exception as err:
            self.fail(f"Unexpected exception was raised: {err}")

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_yml")
    return parser


