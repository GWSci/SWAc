import unittest
import swacmod.model as m
import swacmod.input_output as input_output
import warnings
import test.file_test_helpers as file_test_helpers

class Test_Get_Evt_File(unittest.TestCase):
	def test_get_evt_file_mfusg_nevopt_2(self):
		run_name = "run-evt-aardvark"
		gwmodel_type = "mfusg"
		ievtcb = 0
		nevtopt = 2
		filename = "output_files/run-evt-aardvark.evt"
		expected = """# EVT package for  MODFLOW-USG, generated by SWAcMod.
         2         0
         3
         1         1         1         3 # Evapotranspiration  dataset 5 for stress period 1
INTERNAL               1   (1E15.6) -1 #surf1                         
   7.000000E+00
   1.700000E+01
   2.900000E+01
INTERNAL               1   (1E15.6) -1 #evtr1                         
   3.700000E-02
   4.100000E-02
   4.300000E-02
INTERNAL               1   (1E15.6) -1 #exdp1                         
   1.100000E+01
   1.900000E+01
   3.100000E+01
INTERNAL               1     (1I10) -1 #ievt1                         
         5
        13
        23
        -1         1        -1        -1 # Evapotranspiration  dataset 5 for stress period 2
INTERNAL               1   (1E15.6) -1 #evtr2                         
   4.700000E-02
   5.300000E-02
   5.900000E-02
"""
		data = {
			"params" : {
				"run_name" : run_name,
				"time_periods" : [1, 2], # Only used for nper
				"num_nodes" : 3,
				"gwmodel_type" : gwmodel_type,
				"ievtcb" : ievtcb,
				"nevtopt" : nevtopt,
				"evt_parameters" : {
					1 : [5, 7, 11],
					2 : [13, 17, 19],
					3 : [23, 29, 31],
				},
			}
		}
		evtrate = [-1, 37, 41, 43, 47, 53, 59]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			warnings.filterwarnings("ignore", category=UserWarning)

			evt_out = m.get_evt_file(data, evtrate)
		input_output.dump_evt_output(evt_out)
		actual = file_test_helpers.slurp(filename)
		self.assertEqual(expected, actual)

	def test_get_evt_file_mf6_nevopt_2(self):
		run_name = "run-evt-mf6-aardvark"
		gwmodel_type = "mf6"
		ievtcb = 0
		nevtopt = 2
		filename = "output_files/run-evt-mf6-aardvark.evt"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
  NSEG  1
END dimensions

BEGIN period  1
  5       7.00000000       0.03700000      11.00000000    -999.00000000
  13      17.00000000       0.04100000      19.00000000    -999.00000000
  23      29.00000000       0.04300000      31.00000000    -999.00000000
END period  1

BEGIN period  2
  5       7.00000000       0.04700000      11.00000000    -999.00000000
  13      17.00000000       0.05300000      19.00000000    -999.00000000
  23      29.00000000       0.05900000      31.00000000    -999.00000000
END period  2

"""
		data = {
			"params" : {
				"run_name" : run_name,
				"time_periods" : [1, 2], # Only used for nper
				"num_nodes" : 3,
				"gwmodel_type" : gwmodel_type,
				"ievtcb" : ievtcb,
				"nevtopt" : nevtopt,
				"evt_parameters" : {
					1 : [5, 7, 11],
					2 : [13, 17, 19],
					3 : [23, 29, 31],
				},
			}
		}
		evtrate = [-1, 37, 41, 43, 47, 53, 59]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			warnings.filterwarnings("ignore", category=UserWarning)

			evt_out = m.get_evt_file(data, evtrate)
		evt_out.write()
		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(expected, actual)
