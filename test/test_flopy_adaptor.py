import unittest
import swacmod.flopy_adaptor as flopy_adaptor
import numpy as np
import warnings
import math

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

	def test_mf_tdis(self):
		sim = flopy_adaptor.mf_simulation()
		tdis = flopy_adaptor.mf_tdis(sim, 3)

		self.assertEqual(sim, tdis.model_or_sim)
		self.assertEqual("  NPER  3\n", tdis.nper.get_file_entry())
		self.assertFalse(tdis.loading_package)
		self.assertEqual("", tdis.start_date_time.get_file_entry())
		self.assertEqual("sim.tdis", tdis.filename)
		# self.assertEqual("", tdis.pname.get_file_entry())
		self.assertIsNone(tdis.parent_file)

	def test_mf_tdis_adds_disu_to_simulation(self):
		sim = flopy_adaptor.mf_simulation()
		tdis = flopy_adaptor.mf_tdis(sim, 3)

		self.assertEqual(tdis, sim.get_package("tdis"))

	def test_modflow_model(self):
		model = flopy_adaptor.modflow_model("aardvark", "mfusg", False)
		self.assertEqual("aardvark", model.name)
		self.assertEqual("mfusg", model.version)
		self.assertFalse(model.structured)

	def test_modflow_model_with_structured_true(self):
		model = flopy_adaptor.modflow_model("aardvark", "mf2005", True)
		self.assertEqual("aardvark", model.name)
		self.assertEqual("mf2005", model.version)
		self.assertTrue(model.structured)

	def test_modflow_model_with_version(self):
		model = flopy_adaptor.modflow_model("aardvark", "mfusg", False)
		self.assertEqual("aardvark", model.name)
		self.assertEqual("mfusg", model.version)
		self.assertFalse(model.structured)

	def test_modflow_disu(self):
		model = flopy_adaptor.modflow_model("aardvark", "mfusg", False)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			disu = flopy_adaptor.modflow_disu(model, 3, 5, 7, 2)

		self.assertEqual(3, disu.nodes)
		self.assertEqual(5, disu.nper)
		np.testing.assert_almost_equal([7, 0, 0], disu.iac.array)
		np.testing.assert_almost_equal([0, 0, 0, 0, 0, 0, 0], disu.ja.array)
		self.assertEqual(7, disu.njag)
		self.assertEqual(1, disu.idsymrd)
		np.testing.assert_almost_equal([0, 0], disu.cl1.array)
		np.testing.assert_almost_equal([0, 0], disu.cl2.array)
		np.testing.assert_almost_equal([0, 0], disu.fahl.array)

	def test_modflow_disu_adds_disu_to_model(self):
		model = flopy_adaptor.modflow_model("aardvark", "mfusg", False)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			disu = flopy_adaptor.modflow_disu(model, 3, 5, 7, 2)

		self.assertEqual(disu, model.get_package("disu"))

	def test_make_empty_modflow_gwf_rch_stress_period_data(self):
		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		flopy_adaptor.mf_gwf_disv(model)
		actual = flopy_adaptor.make_empty_modflow_gwf_rch_stress_period_data(model, 3, 5)

		expected_node_count = 3
		expected_stress_period_count = 5
		expected_field_count = 2
		self.assertEqual(expected_stress_period_count, len(actual))

		for per in range(expected_stress_period_count):
			self.assertEqual(expected_node_count, len(actual[per]))

		for per in range(expected_stress_period_count):
			for node in range(expected_node_count):
				self.assertEqual(expected_field_count, len(actual[per][node]))

		for per in range(expected_stress_period_count):
			for node in range(expected_node_count):
				self.assertIsNone(None, actual[per][node][0])
				self.assertTrue(math.isnan(actual[per][node][1]))

	def test_mf_gwf_rch(self):
		node_count = 3
		stress_period_count = 5
		njag = node_count + 2

		sim = flopy_adaptor.mf_simulation()
		model = flopy_adaptor.mf_model(sim, "aardvark")
		flopy_adaptor.mf_gwf_disu(model, node_count, njag, area=1.0)
		flopy_adaptor.mf_tdis(sim, stress_period_count)
		spd = flopy_adaptor.make_empty_modflow_gwf_rch_stress_period_data(model, node_count, stress_period_count)

		rch = flopy_adaptor.mf_gwf_rch(model, node_count, spd)

		self.assertEqual(model, rch.model_or_sim)
		self.assertEqual('  MAXBOUND  3\n', rch.maxbound.get_file_entry())

		actual_spd = rch.stress_period_data.get_data()

		self.assertEqual(stress_period_count, len(actual_spd))

		for per in range(stress_period_count):
			self.assertEqual(node_count, len(actual_spd[per]))

		for per in range(stress_period_count):
			for node in range(node_count):
				self.assertEqual(2, len(actual_spd[per][node]))

		for per in range(stress_period_count):
			for node in range(node_count):
				self.assertIsNone(None, actual_spd[per][node][0])
				self.assertTrue(math.isnan(actual_spd[per][node][1]))
