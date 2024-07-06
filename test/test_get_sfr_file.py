import unittest
import swacmod.model as m
import warnings
import swacmod.input_output as input_output
import test.file_test_helpers as file_test_helpers
import datetime
from test.test_get_attenuated_sfr_flows import make_routing_parameters

class Test_Get_Sfr_File(unittest.TestCase):
	def test_get_sfr_file_mfusg(self):
		run_name = "sfr-mfusg-aardvark"
		gwmodel_type = "mfusg"
		filename = "output_files/sfr-mfusg-aardvark.sfr"

		sfr = get_sfr_adaptor(run_name, gwmodel_type, True, None)
		input_output.dump_sfr_output(sfr)
		actual = file_test_helpers.slurp(filename)
		self.assertEqual(expected_sfr_mfusg_aardvark, actual)

	def test_get_sfr_file_mf6(self):
		run_name = "sfr-mf6-aardvark"
		gwmodel_type = "mf6"
		filename = "output_files/sfr-mf6-aardvark.sfr"
		disv = True
		sfr_obs = []

		sfr = get_sfr_adaptor(run_name, gwmodel_type, disv, sfr_obs)
		sfr.write()
		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(expected_sfr_mf6_aardvark, actual)

	def test_get_sfr_file_mf6_disu(self):
		run_name = "sfr-mf6-disu-aardvark"
		gwmodel_type = "mf6"
		filename = "output_files/sfr-mf6-disu-aardvark.sfr"
		disv = False
		sfr_obs = []

		sfr = get_sfr_adaptor(run_name, gwmodel_type, disv, sfr_obs)
		sfr.write()
		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(expected_sfr_mf6_disu_aardvark, actual)

	def test_get_sfr_file_mf6_obs(self):
		run_name = "sfr-mf6-obs-aardvark"
		gwmodel_type = "mf6"
		filename = "output_files/sfr-mf6-obs-aardvark.sfr"
		disv = True
		sfr_obs = "some-obs-filename"

		sfr = get_sfr_adaptor(run_name, gwmodel_type, disv, sfr_obs)
		sfr.write()
		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(expected_sfr_mf6_obs_aardvark, actual)

	def test_get_sfr_file_mf6_disu_with_release_proportion(self):
		run_name = "sfr-mf6_disu_with_release_proportion"
		gwmodel_type = "mf6"
		filename = "output_files/sfr-mf6_disu_with_release_proportion.sfr"
		disv = False
		sfr_obs = []

		sfr = get_sfr_with_release_proportion(run_name, gwmodel_type, disv, sfr_obs)
		sfr.write()
		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(sfr_mf6_disu_with_release_proportion, actual)

def get_sfr_adaptor(run_name, gwmodel_type, disv, sfr_obs):
		data = {
			"params" : {
				"attenuate_sfr_flows": False,
				"node_areas" : [-1, 100, 200, 300],
				"run_name" : run_name,
				"time_periods" : [1, 2],
				"num_nodes" : 3,
				"routing_topology" : {
					1 : [1, 1, 30, 40, 50, 60, 70, 80, 90, 100],
					2 : [2, 1, 30, 40, 50, 60, 70, 80, 90, 100],
					3 : [3, 1, 30, 40, 50, 60, 70, 80, 90, 100],
				},
				"istcb1" : None,
				"istcb2" : None,
				"gwmodel_type" : gwmodel_type,
				"disv" : disv,
				"sfr_obs" : sfr_obs,
			}
		}
		runoff = [-1, 5, 7, 11, 13, 17, 19]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			sfr = m.get_sfr_file(data, runoff)
		return sfr

def get_sfr_with_release_proportion(run_name, gwmodel_type, disv, sfr_obs):
		nodes = 16
		data = {
			"params" : {
				"attenuate_sfr_flows": True,
				"node_areas" : [-1, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400],
				"run_name" : run_name,
				"time_periods" : [[1, 2], [2, 3]],
				"num_nodes" : 16,
				"routing_topology" : {
					1 : make_routing_parameters(node_mf = 1, ca = 1, downstr = 3),
					2 : make_routing_parameters(node_mf = 2, ca = 2, downstr = 3),
					3 : make_routing_parameters(node_mf = 3, ca = 3, downstr = 4),
					4 : make_routing_parameters(node_mf = 4,  ca = 4, downstr = 12, str_flag = 1, RCHLEN=1, WIDTH2=2, STRTOP=3, STRTHICK=4, STRHC1=5),
					5 : make_routing_parameters(node_mf = 5, ca = 5, downstr = 6),
					6 : make_routing_parameters(node_mf = 6, ca = 6, downstr = 7),
					7 : make_routing_parameters(node_mf = 7, ca = 7, downstr = 8),
					8 : make_routing_parameters(node_mf = 8, ca = 8, downstr = 12, str_flag = 1, RCHLEN=1, WIDTH2=2, STRTOP=3, STRTHICK=4, STRHC1=5),
					9 : make_routing_parameters(node_mf = 9, ca = 9, downstr = 12),
					10 : make_routing_parameters(node_mf = 10, ca = 10, downstr = 12),
					11 : make_routing_parameters(node_mf = 11, ca = 11, downstr = 12),
					12 : make_routing_parameters(node_mf = 12, ca = 12, downstr = 16, str_flag = 1, RCHLEN=1, WIDTH2=2, STRTOP=3, STRTHICK=4, STRHC1=5),
					13 : make_routing_parameters(node_mf = 13, ca = 13, downstr = 16),
					14 : make_routing_parameters(node_mf = 14, ca = 14, downstr = 15),
					15 : make_routing_parameters(node_mf = 15, ca = 15, downstr = 16),
					16 : make_routing_parameters(node_mf = 16, ca = 16, downstr = 0, str_flag = 1, RCHLEN=1, WIDTH2=2, STRTOP=3, STRTHICK=4, STRHC1=5),
				},
				"istcb1" : None,
				"istcb2" : None,
				"gwmodel_type" : gwmodel_type,
				"disv" : disv,
				"sfr_obs" : sfr_obs,
				"sfr_flow_monthly_proportions": {
					1: [0.2, 0.4, 0.6, 0.8],
					2: [0.3, 0.5, 0.7, 0.9],
				},
				"sfr_flow_zones": {
					4: [1],
					8: [2],
					12: [3],
					16: [4],
				},
			},
			"series": {
				"date" : [datetime.datetime(1980, 1, 1), datetime.datetime(1980, 2, 1),]
			},
		}
		time_period_count = 2
		runoff_count = nodes * time_period_count
		runoff = long_list_for_source = [pow(2, x) for x in range(runoff_count)]
		runoff = [-1000, -2000, -3000] + long_list_for_source[:runoff_count]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			sfr = m.get_sfr_file(data, runoff)
		return sfr

expected_sfr_mfusg_aardvark = """# SFR package for  MODFLOW-USG, generated by SWAcMod.
reachinput 
3 3 0 0 86400.00000000 0.00010000 0 0 1 
30 1 1 40.0 60.0 1e-04 70.0 80.0
30 2 1 40.0 60.0 1e-04 70.0 80.0
30 3 1 40.0 60.0 1e-04 70.0 80.0
3 0 0
1 0 1 0 0.0 0.0 0.0 0.0 
100.0 90.0 
100.0 90.0 
2 0 2 0 0.0 1.4 0.0 0.0 
100.0 90.0 
100.0 90.0 
3 0 3 0 0.0 3.3 0.0 0.0 
100.0 90.0 
100.0 90.0 
3 0 0
1 0 1 0 0.0 0.0 0.0 0.0 
100.0 90.0 
100.0 90.0 
2 0 2 0 0.0 3.4 0.0 0.0 
100.0 90.0 
100.0 90.0 
3 0 3 0 0.0 5.7 0.0 0.0 
100.0 90.0 
100.0 90.0 
"""

expected_sfr_mf6_aardvark = """BEGIN options
  UNIT_CONVERSION   86400.00000000
END options

BEGIN dimensions
  NREACHES  3
END dimensions

BEGIN packagedata
  1  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  2  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  3  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
END packagedata

BEGIN connectiondata
  1  -1
  2  -2
  3  -3
END connectiondata

BEGIN period  1
  1  STAGE  150
  1  STATUS  SIMPLE
  2  STAGE  150
  2  STATUS  SIMPLE
  3  STAGE  150
  3  STATUS  SIMPLE
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  1.4000000000000001
  2  INFLOW  0.0
  3  RUNOFF  3.3000000000000003
  3  INFLOW  0.0
END period  1

BEGIN period  2
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  3.4
  2  INFLOW  0.0
  3  RUNOFF  5.7
  3  INFLOW  0.0
END period  2

"""

expected_sfr_mf6_disu_aardvark = """BEGIN options
  UNIT_CONVERSION   86400.00000000
END options

BEGIN dimensions
  NREACHES  3
END dimensions

BEGIN packagedata
  1  30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  2  30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  3  30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
END packagedata

BEGIN connectiondata
  1  -1
  2  -2
  3  -3
END connectiondata

BEGIN period  1
  1  STAGE  150
  1  STATUS  SIMPLE
  2  STAGE  150
  2  STATUS  SIMPLE
  3  STAGE  150
  3  STATUS  SIMPLE
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  1.4000000000000001
  2  INFLOW  0.0
  3  RUNOFF  3.3000000000000003
  3  INFLOW  0.0
END period  1

BEGIN period  2
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  3.4
  2  INFLOW  0.0
  3  RUNOFF  5.7
  3  INFLOW  0.0
END period  2

"""

expected_sfr_mf6_obs_aardvark = """BEGIN options
  OBS6  FILEIN  some-obs-filename
  UNIT_CONVERSION   86400.00000000
END options

BEGIN dimensions
  NREACHES  3
END dimensions

BEGIN packagedata
  1  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  2  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
  3  1 30      40.00000000     100.00000000  1.00000000E-04      60.00000000      70.00000000      80.00000000  0.0001  1       1.00000000  0
END packagedata

BEGIN connectiondata
  1  -1
  2  -2
  3  -3
END connectiondata

BEGIN period  1
  1  STAGE  150
  1  STATUS  SIMPLE
  2  STAGE  150
  2  STATUS  SIMPLE
  3  STAGE  150
  3  STATUS  SIMPLE
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  1.4000000000000001
  2  INFLOW  0.0
  3  RUNOFF  3.3000000000000003
  3  INFLOW  0.0
END period  1

BEGIN period  2
  1  RUNOFF  0.0
  1  INFLOW  0.0
  2  RUNOFF  3.4
  2  INFLOW  0.0
  3  RUNOFF  5.7
  3  INFLOW  0.0
END period  2

"""

sfr_mf6_disu_with_release_proportion = """BEGIN options
  UNIT_CONVERSION   86400.00000000
END options

BEGIN dimensions
  NREACHES  4
END dimensions

BEGIN packagedata
  1  4       1.00000000       2.00000000  1.00000000E-04       3.00000000       4.00000000       5.00000000  0.0001  1       1.00000000  0
  2  8       1.00000000       2.00000000  1.00000000E-04       3.00000000       4.00000000       5.00000000  0.0001  1       1.00000000  0
  3  12       1.00000000       2.00000000  1.00000000E-04       3.00000000       4.00000000       5.00000000  0.0001  3       1.00000000  0
  4  16       1.00000000       2.00000000  1.00000000E-04       3.00000000       4.00000000       5.00000000  0.0001  1       1.00000000  0
END packagedata

BEGIN connectiondata
  1  -3
  2  -3
  3  1  2  -4
  4  3
END connectiondata

BEGIN period  1
  1  STAGE  2
  1  STATUS  SIMPLE
  2  STAGE  2
  2  STATUS  SIMPLE
  3  STAGE  2
  3  STATUS  SIMPLE
  4  STAGE  2
  4  STATUS  SIMPLE
  1  RUNOFF  0.0
  1  INFLOW  -159.78000000000003
  2  RUNOFF  0.0
  2  INFLOW  7.840000000000001
  3  RUNOFF  0.0
  3  INFLOW  248.936
  4  RUNOFF  0.0
  4  INFLOW  3994.6808000000005
END period  1

BEGIN period  2
  1  RUNOFF  0.0
  1  INFLOW  23892.744000000002
  2  RUNOFF  0.0
  2  INFLOW  642258.68
  3  RUNOFF  0.0
  3  INFLOW  14186662.557599999
  4  RUNOFF  0.0
  4  INFLOW  294465729.46912
END period  2

"""