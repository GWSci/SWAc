import unittest
import swacmod.finalization as finalization

class Test_Finalization(unittest.TestCase):
    def test_fin_run_name_does_not_convert_lower_case_letters(self):
        self.assertEqual("aardvark", fin_run_name_adaptor("aardvark"))

    def test_fin_run_name_does_not_convert_upper_case_letters(self):
        self.assertEqual("AARDVARK", fin_run_name_adaptor("AARDVARK"))

    def test_fin_run_name_does_not_convert_digits(self):
        self.assertEqual("1234567890", fin_run_name_adaptor("1234567890"))

    def test_fin_run_name_does_not_convert_dashed(self):
        self.assertEqual("---", fin_run_name_adaptor("---"))

    def test_fin_run_name_converts_symbols_to_underscores(self):
        self.assertEqual("_____", fin_run_name_adaptor("!@£$%"))

def fin_run_name_adaptor(input_run_name):
    data = {"params": {"run_name": input_run_name}}
    finalization.fin_run_name(data, "run_name")
    return data["params"]["run_name"]
