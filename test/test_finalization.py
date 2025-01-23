import unittest
import swacmod.finalization as finalization

class Test_Finalization(unittest.TestCase):
    def test_fin_run_name_when_params_has_ordinary_string_for_run_name(self):
        self.assertEqual("aardvark", fin_run_name_adaptor("aardvark"))

def fin_run_name_adaptor(input_run_name):
    data = {"params": {"run_name": input_run_name}}
    name = "run_name"
    finalization.fin_run_name(data, name)
    return data["params"][name]
