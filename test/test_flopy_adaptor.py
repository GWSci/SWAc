import unittest
import swacmod.flopy_adaptor as flopy_adaptor

class Test_Flopy_Adaptor(unittest.TestCase):
	def test_mf_simulation(self):
		sim = flopy_adaptor.mf_simulation()
		self.assertEqual("sim", sim.name)
		self.assertEqual("mf6", sim.version)
		self.assertEqual("mf6.exe", sim.exe_name)

	def test_mf_model(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		self.assertEqual(sim, model.simulation)
		self.assertEqual("aardvark", model.name)
		self.assertEqual("gwf6", model.model_type)
		self.assertEqual("aardvark.nam", model.model_nam_file)
		self.assertEqual("mf6", model.version)
		self.assertEqual("mf6.exe", model.exe_name)
