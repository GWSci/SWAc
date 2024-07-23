import unittest
import swacmod.model as m
import swacmod.flopy_adaptor as flopy_adaptor
import test.file_test_helpers as file_test_helpers

class Test_Get_Mf6Rch_File(unittest.TestCase):
	def test_write_rch_with_disv_and_no_node_mapping(self):
		run_name = "run-aardvark"
		is_disv = True
		recharge_node_mapping = None
		filename = "output_files/run-aardvark.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  1 1       0.00500000
  1 2       0.00700000
  1 3       0.01100000
END period  1

BEGIN period  2
  1 1       0.01300000
  1 2       0.01700000
  1 3       0.01900000
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def test_write_rch_with_disu_and_no_node_mapping(self):
		run_name = "run-bat"
		is_disv = False
		recharge_node_mapping = None
		filename = "output_files/run-bat.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  1       0.00500000
  2       0.00700000
  3       0.01100000
END period  1

BEGIN period  2
  1       0.01300000
  2       0.01700000
  3       0.01900000
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def test_write_rch_with_disu_and_node_mapping_for_all_nodes(self):
		run_name = "run-cat"
		is_disv = False
		recharge_node_mapping = {
			1: [7],
			2: [23],
			3: [29],
		}
		filename = "output_files/run-cat.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  7       0.00500000
  23       0.00700000
  29       0.01100000
END period  1

BEGIN period  2
  7       0.01300000
  23       0.01700000
  29       0.01900000
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def test_write_rch_with_disv_and_node_mapping_for_only_some_nodes(self):
		run_name = "run-dog"
		is_disv = True
		recharge_node_mapping = {
			1: [0],
			2: [23],
		}
		filename = "output_files/run-dog.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  1 23       0.00700000
  
  
END period  1

BEGIN period  2
  1 23       0.01700000
  
  
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def test_write_rch_with_disv_and_node_mapping_for_all_nodes(self):
		run_name = "run-elephant"
		is_disv = True
		recharge_node_mapping = {
			1: [7],
			2: [23],
			3: [29],
		}
		filename = "output_files/run-elephant.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  1 7       0.00500000
  1 23       0.00700000
  1 29       0.01100000
END period  1

BEGIN period  2
  1 7       0.01300000
  1 23       0.01700000
  1 29       0.01900000
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def test_write_rch_with_disu_and_node_mapping_for_only_some_nodes(self):
		run_name = "run-goat"
		is_disv = False
		recharge_node_mapping = {
			1: [0],
			2: [23],
		}
		filename = "output_files/run-goat.rch"
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  23       0.00700000
  
  
END period  1

BEGIN period  2
  23       0.01700000
  
  
END period  2

"""
		self.assert_rch_file(run_name, is_disv, recharge_node_mapping, filename, expected)

	def assert_rch_file(self, run_name, is_disv, recharge_node_mapping, filename, expected):
		data = {
			"params" : {
				"run_name" : run_name,
				"recharge_node_mapping" : recharge_node_mapping,
				"time_periods" : [-1, -1], # Only the length is used as nper
				"num_nodes" : 3,
				"disv" : is_disv,
			}
		}
		# rchrate is 1-based, accessed as rchrate[(nodes * per) + i + 1]
		rchrate = [-1, 5, 7, 11, 13, 17, 19]
		rch_out = m.get_mf6rch_file(data, rchrate)
		flopy_adaptor.write_mf_gwf_rch(rch_out)

		actual = file_test_helpers.slurp_without_first_line(filename)
		self.assertEqual(expected, actual)
