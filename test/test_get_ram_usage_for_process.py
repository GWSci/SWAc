import unittest
import swacmod.utils as utils

class Test_Get_Ram_Usage_For_Process(unittest.TestCase):
	def test_get_ram_usage_for_process(self):
		actual = utils.get_ram_usage_for_process()
		message = f"Actual RAM usage was {actual}."
		self.assertTrue(actual > 0, msg = message)
