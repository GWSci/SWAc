import unittest
import swacmod.finalization as finalization

class Test_Attenuate_Sfr_Finalization(unittest.TestCase):
	def test_output_sfr_is_false_when_missing(self):
		data = {
			"params": {},
		}
		finalization.fin_attenuate_sfr_flows(data, "attenuate_sfr_flows")
		self.assertEqual(False, data["params"]["attenuate_sfr_flows"])

	def test_output_sfr_is_retained_when_supplied(self):
		data = {
			"params": {"attenuate_sfr_flows" : True},
		}
		finalization.fin_attenuate_sfr_flows(data, "attenuate_sfr_flows")
		self.assertEqual(True, data["params"]["attenuate_sfr_flows"])
