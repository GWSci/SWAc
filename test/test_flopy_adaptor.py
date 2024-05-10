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

	def test_mf_gwf_disv(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disv = flopy_adaptor.mf_gwf_disv(model)

		self.assertEqual(model, disv.model_or_sim)
		self.assertEqual(model, disv.parent)
		self.assertEqual(["disv"], disv.name)

		self.assertEqual("", disv.length_units.get_file_entry())
		self.assertEqual("", disv.nogrb.get_file_entry())
		self.assertEqual("", disv.xorigin.get_file_entry())
		self.assertEqual("", disv.yorigin.get_file_entry())
		self.assertEqual("", disv.angrot.get_file_entry())
		self.assertEqual("", disv.nlay.get_file_entry())
		self.assertEqual("", disv.ncpl.get_file_entry())
		self.assertEqual("", disv.nvert.get_file_entry())
		self.assertEqual("", disv.top.get_file_entry())
		self.assertEqual("", disv.botm.get_file_entry())
		self.assertEqual("", disv.idomain.get_file_entry())
		self.assertEqual("", disv.vertices.get_file_entry())
		self.assertEqual("", disv.cell2d.get_file_entry())

	def test_mf_gwf_disv_adds_disv_to_model(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disv = flopy_adaptor.mf_gwf_disv(model)

		self.assertEqual(disv, model.get_package("disv"))
