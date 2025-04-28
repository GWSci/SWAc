import unittest
from swacmod.input_files.parsed_input_data import ParsedInputData

class Test_Parsed_Input_Data(unittest.TestCase):
    def test_parsed_input_data_prints_errors_and_warnings(self):
        self.assertEqual("", print_adaptor(ParsedInputData(None, [], [])))

def print_adaptor(testee):
    result = ""
    def printer_spy(x):
        result += x
    testee.print(printer = printer_spy)
    return result
