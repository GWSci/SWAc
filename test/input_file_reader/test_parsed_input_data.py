import unittest
from swacmod.input_files.parsed_input_data import ParsedInputData

class Test_Parsed_Input_Data(unittest.TestCase):
    def test_parsed_input_data_prints_errors_and_warnings(self):
        self.assertEqual("", print_adaptor(ParsedInputData(None, [], [])))
        self.assertEqual("a", print_adaptor(ParsedInputData(None, ["a"], [])))
        self.assertEqual("ab", print_adaptor(ParsedInputData(None, ["a", "b"], [])))
        self.assertEqual("c", print_adaptor(ParsedInputData(None, [], ["c"])))
        self.assertEqual("cd", print_adaptor(ParsedInputData(None, [], ["c", "d"])))
        self.assertEqual("abcd", print_adaptor(ParsedInputData(None, ["a", "b"], ["c", "d"])))

class Printer_Spy:
    def __init__(self):
        self.result = ""
    
    def do_print(self, x):
        self.result += x

def print_adaptor(testee):
    spy = Printer_Spy()
    testee.print(printer = spy.do_print)
    return spy.result
