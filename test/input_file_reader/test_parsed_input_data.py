import unittest
from swacmod.input_files.parsed_input_data import ParsedInputData

class Test_Parsed_Input_Data(unittest.TestCase):
    def test_x(self):
        testee = ParsedInputData(None, [], [])
        actual = ""
        def printer_spy(x):
            actual += x
        testee.print(printer = printer_spy)
        self.assertEqual("", actual)
