import unittest
import swacmod.flopy_adaptor as flopy_adaptor
import numpy as np

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

	def test_mf_gwf_disu(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disu = flopy_adaptor.mf_gwf_disu(model, 3, 5)

		self.assertEqual(["disu"], disu.name)
		self.assertEqual(model, disu.model_or_sim)
		self.assertEqual(model, disu.parent)

		self.assertEqual("  NODES  3\n", disu.nodes.get_file_entry())
		self.assertEqual("  NJA  5\n", disu.nja.get_file_entry())
		np.testing.assert_almost_equal(np.zeros(5), disu.ja.get_data())
		self.assertEqual("  ihc\n    CONSTANT  1\n", disu.ihc.get_file_entry())
		self.assertEqual("  iac\n    CONSTANT  1\n", disu.iac.get_file_entry())

	def test_mf_gwf_disu_adds_disu_to_model(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disu = flopy_adaptor.mf_gwf_disu(model, 3, 5)

		self.assertEqual(disu, model.get_package("disu"))

	def test_mf_gwf_disu_does_not_set_area(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disu = flopy_adaptor.mf_gwf_disu(model, 3, 5)

		self.assertEqual("", disu.area.get_file_entry())

	def test_mf_gwf_disu_does_sets_area_when_provided(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		disu = flopy_adaptor.mf_gwf_disu(model, 3, 5, 7.0)

		self.assertEqual("  area\n    CONSTANT       7.00000000\n", disu.area.get_file_entry())
