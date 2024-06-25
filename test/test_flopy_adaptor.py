import unittest
import swacmod.flopy_adaptor as flopy_adaptor
import numpy as np
import warnings
import math
import tempfile
import test.file_test_helpers as file_test_helpers

class Test_Flopy_Adaptor(unittest.TestCase):
	def test__mf_simulation(self):
		sim = flopy_adaptor._mf_simulation()
		self.assertEqual("sim", sim.name)
		self.assertEqual("mf6", sim.version)
		self.assertEqual("mf6.exe", sim.exe_name)

	def test__mf_model(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		self.assertEqual(sim, model.simulation)
		self.assertEqual("aardvark", model.name)
		self.assertEqual("gwf6", model.model_type)
		self.assertEqual("aardvark.nam", model.model_nam_file)
		self.assertEqual("mf6", model.version)
		self.assertEqual("mf6.exe", model.exe_name)

	def test__mf_gwf_disv(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disv = flopy_adaptor._mf_gwf_disv(model)

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

	def test__mf_gwf_disv_adds_disv_to_model(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disv = flopy_adaptor._mf_gwf_disv(model)

		self.assertEqual(disv, model.get_package("disv"))

	def test__mf_gwf_disu(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disu = flopy_adaptor._mf_gwf_disu(model, 3, 5)

		self.assertEqual(["disu"], disu.name)
		self.assertEqual(model, disu.model_or_sim)
		self.assertEqual(model, disu.parent)

		self.assertEqual("  NODES  3\n", disu.nodes.get_file_entry())
		self.assertEqual("  NJA  5\n", disu.nja.get_file_entry())
		np.testing.assert_almost_equal(np.zeros(5), disu.ja.get_data())
		self.assertEqual("  ihc\n    CONSTANT  1\n", disu.ihc.get_file_entry())
		self.assertEqual("  iac\n    CONSTANT  1\n", disu.iac.get_file_entry())

	def test__mf_gwf_disu_adds_disu_to_model(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disu = flopy_adaptor._mf_gwf_disu(model, 3, 5)

		self.assertEqual(disu, model.get_package("disu"))

	def test__mf_gwf_disu_does_not_set_area(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disu = flopy_adaptor._mf_gwf_disu(model, 3, 5)

		self.assertEqual("", disu.area.get_file_entry())

	def test__mf_gwf_disu_does_sets_area_when_provided(self):
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		disu = flopy_adaptor._mf_gwf_disu(model, 3, 5, 7.0)

		self.assertEqual("  area\n    CONSTANT       7.00000000\n", disu.area.get_file_entry())

	def test__mf_tdis(self):
		sim = flopy_adaptor._mf_simulation()
		tdis = flopy_adaptor._mf_tdis(sim, 3)

		self.assertEqual(sim, tdis.model_or_sim)
		self.assertEqual("  NPER  3\n", tdis.nper.get_file_entry())
		self.assertFalse(tdis.loading_package)
		self.assertEqual("", tdis.start_date_time.get_file_entry())
		self.assertEqual("sim.tdis", tdis.filename)
		# self.assertEqual("", tdis.pname.get_file_entry())
		self.assertIsNone(tdis.parent_file)

	def test__mf_tdis_adds_disu_to_simulation(self):
		sim = flopy_adaptor._mf_simulation()
		tdis = flopy_adaptor._mf_tdis(sim, 3)

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
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		flopy_adaptor._mf_gwf_disv(model)
		actual = flopy_adaptor._make_empty_modflow_gwf_rch_stress_period_data(model, 3, 5)

		self.assert_empty_stress_period_data(5, 3, actual)

	def test_mf_gwf_rch(self):
		node_count = 3
		stress_period_count = 5
		njag = node_count + 2

		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		flopy_adaptor._mf_gwf_disu(model, node_count, njag, area=1.0)
		flopy_adaptor._mf_tdis(sim, stress_period_count)
		spd = flopy_adaptor._make_empty_modflow_gwf_rch_stress_period_data(model, node_count, stress_period_count)

		rch = flopy_adaptor.mf_gwf_rch(model, node_count, spd)

		self.assertEqual(model, rch.model_or_sim)
		self.assertEqual('  MAXBOUND  3\n', rch.maxbound.get_file_entry())
		self.assert_empty_stress_period_data(5, 3, rch.stress_period_data.get_data())

	def test_write_mf_gwf_rch(self):
		with tempfile.TemporaryDirectory() as temp_directory:
			path = temp_directory + "/some_temp_file"
			node_count = 3
			stress_period_count = 5
			njag = node_count + 2

			sim = flopy_adaptor._mf_simulation()
			model = flopy_adaptor._mf_model(sim, path)
			flopy_adaptor._mf_gwf_disu(model, node_count, njag, area=1.0)
			flopy_adaptor._mf_tdis(sim, stress_period_count)
			spd = flopy_adaptor._make_empty_modflow_gwf_rch_stress_period_data(model, node_count, stress_period_count)

			value = 10
			for sp in range(stress_period_count):
				for n in range(node_count):
					spd[sp][n] = value
					value = value + 1
				value = value + 100

			rch = flopy_adaptor.mf_gwf_rch(model, node_count, spd)

			flopy_adaptor.write_mf_gwf_rch(rch)
			actual = file_test_helpers.slurp_without_first_line(path + ".rch")
			expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  11      10.00000000
  12      11.00000000
  13      12.00000000
END period  1

BEGIN period  2
  114     113.00000000
  115     114.00000000
  116     115.00000000
END period  2

BEGIN period  3
  217     216.00000000
  218     217.00000000
  219     218.00000000
END period  3

BEGIN period  4
  320     319.00000000
  321     320.00000000
  322     321.00000000
END period  4

BEGIN period  5
  423     422.00000000
  424     423.00000000
  425     424.00000000
END period  5

"""
			self.assertEqual(expected, actual)

	def assert_empty_stress_period_data(self, expected_stress_period_count, expected_node_count, actual):
		self.assertEqual(expected_stress_period_count, len(actual))

		for per in range(expected_stress_period_count):
			self.assertEqual(expected_node_count, len(actual[per]))

		expected_field_count = 2
		for per in range(expected_stress_period_count):
			for node in range(expected_node_count):
				self.assertEqual(expected_field_count, len(actual[per][node]))

		for per in range(expected_stress_period_count):
			for node in range(expected_node_count):
				self.assertIsNone(None, actual[per][node][0])
				self.assertTrue(math.isnan(actual[per][node][1]))

	def test_modflow_sfr2_get_empty_segment_data(self):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			actual = flopy_adaptor.modflow_sfr2_get_empty_segment_data(3)

		self.assert_zeros(3, 34, actual)

	def assert_zeros(self, expected_row_count, expected_column_count, actual):
		self.assertEqual(expected_row_count, len(actual))
		for i in range(expected_row_count):
			self.assertEqual(expected_column_count, len(actual[i]))

		for i in range(expected_row_count):
			for j in range(expected_column_count):
				self.assertEqual(0.0, actual[i][j])

	def test_modflow_sfr2_get_empty_reach_data(self):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			actual = flopy_adaptor.modflow_sfr2_get_empty_reach_data(3)
		actual_list = actual.tolist()
		expected = [
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
		]

		np.testing.assert_array_almost_equal(expected, actual_list)

	def test_mf_str2(self):
		path = "aardvark"
		nstrm = 2
		nss = 2
		istcb1 = 0
		istcb2 = 0

		model = flopy_adaptor.modflow_model(path, "mfusg", False)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			rd = flopy_adaptor.modflow_sfr2_get_empty_reach_data(nstrm)
			seg_data = flopy_adaptor.modflow_sfr2_get_empty_segment_data(nss)
			sfr = flopy_adaptor.mf_str2(model, nstrm, nss, istcb1, istcb2, rd, seg_data)

		self.assertEqual(nstrm, sfr.nstrm)
		self.assertEqual(1, sfr.nss) # The constructor overrides the supplied value.
		self.assertEqual(0, sfr.nsfrpar)
		self.assertEqual(0, sfr.nparseg)
		self.assertEqual(0.0001, sfr.dleak)
		self.assertEqual(istcb1, sfr.ipakcb)
		self.assertEqual(istcb2, sfr.istcb2)
		self.assertEqual(1, sfr.isfropt)
		self.assertEqual(10, sfr.nstrail)
		self.assertEqual(1, sfr.isuzn)
		self.assertEqual(30, sfr.nsfrsets)
		self.assertEqual(0, sfr.irtflg)
		self.assertEqual(2, sfr.numtim)
		self.assertEqual(0.75, sfr.weight)
		self.assertEqual(0.0001, sfr.flwtol)

		expected_reach_data = [
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
		]

		actual_reach_data = sfr.reach_data.tolist()
		np.testing.assert_array_almost_equal(expected_reach_data, actual_reach_data)

		self.assertEqual(1, len(sfr.segment_data))
		self.assert_zeros(2, 34, sfr.segment_data[0])

		self.assertEqual(0, sfr.irdflag)
		self.assertEqual(0, sfr.iptflag)
		self.assertEqual(True, sfr.reachinput)
		self.assertEqual(False, sfr.transroute)
		self.assertEqual(False, sfr.tabfiles)
		self.assertEqual(['sfr'], sfr.extension)

	def test_mf_gwf_sfr(self):
		path = "aardvark"
		nss = 2
		nper = 3
		rd = [
			[2, 0, 3, 5, 0.0001, 7, 11, 13, 0.0001, 1, 1.0, 0],
			[2, 1, 3, 5, 0.0001, 7, 11, 13, 0.0001, 1, 1.0, 0],
		]
		cd = [1, 1]
		sd = {0: [(0, "STAGE", 7)]}

		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, path)
		flopy_adaptor._mf_gwf_disv(model)
		flopy_adaptor._mf_tdis(sim, nper)
		sfr = flopy_adaptor.mf_gwf_sfr(model, nss, rd, cd, sd)

		self.assertEqual(model, sfr.model_or_sim)
		self.assertEqual(False, sfr.loading_package)
		self.assertEqual('  UNIT_CONVERSION   86400.00000000\n', sfr.unit_conversion.get_file_entry())
		self.assertEqual(nss, sfr.nreaches.get_data((0, 'STAGE', 7)))

		expected_rd = [
			(2, (0, 3), 5., 0.0001, 7., 11., 13., 0.0001, 1, 1.0, 0., None),
			(2, (1, 3), 5., 0.0001, 7., 11., 13., 0.0001, 1, 1.0, 0., None)]
		self.assertTrue(expected_rd == sfr.packagedata.get_data().tolist())

		self.assertTrue([(1, 1.0)] == sfr.connectiondata.get_data().tolist())

		actual_period_data = {k: v.tolist() for k, v in sfr.perioddata.get_data().items()}
		self.assertEqual(sd, actual_period_data)

	def test_modflow_bas(self):
		model = flopy_adaptor.modflow_model("aardvark", "mf2005", True)
		bas = flopy_adaptor.modflow_bas(model)

		self.assertEqual(bas, model.get_package("bas6"))
		self.assertFalse(bas.ifrefm)

	def test_modflow_str_get_empty(self):
		ncells = 2,
		nss = 3
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			rd, sd = flopy_adaptor.modflow_str_get_empty(ncells, nss)
		
		expected_rd = [
			[0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0],
		]
		np.testing.assert_array_almost_equal(expected_rd, rd.tolist())

		expected_sd = [
			[0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0],
		]
		np.testing.assert_array_almost_equal(expected_sd, sd.tolist())

	def test_modflow_dis(self):
		model, dis = self.make_modflow_dis()

		self.assertEqual(dis, model.get_package("dis"))
		self.assertEqual(2, model.nlay)
		self.assertEqual(3, model.nrow)
		self.assertEqual(5, model.ncol)
		self.assertEqual(7, model.nper)

	def make_modflow_dis(self):
		nlay = 2
		nrow = 3
		ncol = 5
		nper = 7

		model = flopy_adaptor.modflow_model("aardvark", "mf2005", True)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			dis = flopy_adaptor.modflow_dis(model, nlay, nrow, ncol, nper)
		return model, dis

	def test_modflow_dis_get_lrc_when_node_number_is_zero(self):
		# The documetation says that the input node number should be zero-based but the output makes no sense.
		_, dis = self.make_modflow_dis()
		self.assertEqual([(-1, 2, 5)], dis.get_lrc([0]))

	def test_modflow_dis_get_lrc_when_the_input_is_within_the_model_bounds(self):
		# The documentation says that the input node number should be zero-based but the above test indicates
		# this is not the case. Assuming that the input node number is 1-based:
		#  - The output layer is zero-based.
		#  - The output row is zero-based.
		#  - The output column is one-based.
		_, dis = self.make_modflow_dis()

		# Numbers 1-5. Layer 0, Row 0, Columns 1-5.
		self.assertEqual([(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)], dis.get_lrc([1, 2, 3, 4, 5]))

		# Numbers 6-10. Layer 0, Row 1, Columns 1-5.
		self.assertEqual([(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)], dis.get_lrc([6, 7, 8, 9, 10]))

		# Numbers 11-15. Layer 0, Row 2, Columns 1-5.
		self.assertEqual([(0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), (0, 2, 5)], dis.get_lrc([11, 12, 13, 14, 15]))

		# Numbers 16-20. Layer 1, Row 0, Columns 1-5
		self.assertEqual([(1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 0, 4), (1, 0, 5)], dis.get_lrc([16, 17, 18, 19, 20]))

		# Numbers 21-25. Layer 1, Row 1, Columns 1-5
		self.assertEqual([(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5)], dis.get_lrc([21, 22, 23, 24, 25]))

		# Numbers 26-30. Layer 1, Row 2, Columns 1-5
		self.assertEqual([(1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5)], dis.get_lrc([26, 27, 28, 29, 30]))

	def test_modflow_dis_get_lrc_when_the_input_is_beyond_the_model_bounds(self):
		# The pattern above continues when numbers are out of bounds for the model.
		_, dis = self.make_modflow_dis()

		# Numbers 31-35. Layer 2, Row 0, Columns 1-5.
		self.assertEqual([(2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 0, 4), (2, 0, 5)], dis.get_lrc([31, 32, 33, 34, 35]))

		# Numbers 36-30. Layer 2, Row 1, Columns 1-5.
		self.assertEqual([(2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 1, 4), (2, 1, 5)], dis.get_lrc([36, 37, 38, 39, 40]))

		# Numbers 41-45. Layer 1, Row 1, Columns 1-5
		self.assertEqual([(3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 1, 4), (3, 1, 5)], dis.get_lrc([51, 52, 53, 54, 55]))

	def test_modflow_str(self):
		nlay = 1
		nrow = 1
		ncol = 1
		nper = 1
		nstrm = 1
		istcb1 = 13
		istcb2 = 17
		reach_data = {
			0: [
				(1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
			],
		}
		segment_data = {}
		model = flopy_adaptor.modflow_model("aardvark", "mf2005", True)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			flopy_adaptor.modflow_dis(model, nlay, nrow, ncol, nper)
			str = flopy_adaptor.modflow_str(model, nstrm, istcb1, istcb2, reach_data, segment_data)

		self.assertEqual(str, model.get_package("str"))
		self.assertEqual(nstrm, str.mxacts)
		self.assertEqual(nstrm, str.nss)
		self.assertEqual(8, str.ntrib)
		self.assertEqual(istcb1, str.ipakcb)
		self.assertEqual(istcb2, str.istcb2)

		actual_reach_data = {k: v.tolist() for k, v in str.stress_period_data.data.items()}
		self.assertEqual(reach_data, actual_reach_data)

		self.assertEqual(segment_data, str.segment_data)
		self.assertEqual({0:2}, str.irdflg)

	def test_make_empty_modflow_gwf_evt_stress_period_data(self):
		nodes = 2
		nper = 3
		njag = 5
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, "aardvark")
		flopy_adaptor._mf_gwf_disu(model, nodes, njag)
		flopy_adaptor._mf_tdis(sim, nper)

		spd = flopy_adaptor.make_empty_modflow_gwf_evt_stress_period_data(model, nodes, nper)

		actual_spd = {k: v.tolist() for k, v in spd.items()}
		expected = {
			0: [(None, np.nan, np.nan, np.nan, np.nan, None), (None, np.nan, np.nan, np.nan, np.nan, None)],
			1: [(None, np.nan, np.nan, np.nan, np.nan, None), (None, np.nan, np.nan, np.nan, np.nan, None)],
			2: [(None, np.nan, np.nan, np.nan, np.nan, None), (None, np.nan, np.nan, np.nan, np.nan, None)],
		}
		self.assertEqual(str(expected), str(actual_spd))

	def test_modflow_evt(self):
		nevtopt = 2
		ievtcb = 1
		evt_dic = 3
		surf = 5
		exdp = 7
		ievt = 11
		model = flopy_adaptor.modflow_model("aardvark", "mfusg", True)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			flopy_adaptor.modflow_dis(model, 1, 3, 1, 5)
			evt = flopy_adaptor.modflow_evt(model, nevtopt, ievtcb, evt_dic, surf, exdp, ievt)
		
		self.assertEqual(evt, model.get_package("evt"))
		self.assertEqual(nevtopt, evt.nevtop)
		self.assertEqual(ievtcb, evt.ipakcb)

	def test_modflow_gwf_evt(self):
		path = "aardvark"
		nodes = 3
		nper = 7
		njag = nodes + 2
		sim = flopy_adaptor._mf_simulation()
		model = flopy_adaptor._mf_model(sim, path)
		flopy_adaptor._mf_gwf_disu(model, nodes, njag)
		flopy_adaptor._mf_tdis(sim, nper)
		spd = flopy_adaptor.make_empty_modflow_gwf_evt_stress_period_data(model, nodes, nper)
		evt = flopy_adaptor.modflow_gwf_evt(model, nodes, spd)

		self.assertEqual(evt, model.get_package("evt"))
		self.assertEqual(False, evt.fixed_cell.get_data())
		self.assertEqual('  MAXBOUND  3\n', evt.maxbound.get_file_entry())
		self.assertEqual('  NSEG  1\n', evt.nseg.get_file_entry())
		self.assertEqual(spd, evt.stress_period_data.get_data())
		self.assertEqual(False, evt.surf_rate_specified.get_data())
