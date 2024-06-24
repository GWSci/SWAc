import unittest
import swacmod.model as m
import swacmod.flopy_adaptor as flopy_adaptor

class Test_Get_Mf6Rch_File(unittest.TestCase):
	def test_write_rch_with_disv_and_no_node_mapping(self):
		data = {
			"params" : {
				"run_name" : "run-aardvark",
				"recharge_node_mapping" : None,
				"time_periods" : [-1, -1], # Only the length is used as nper
				"num_nodes" : 3,
				"disv" : True,
			}
		}
		# rchrate is 1-based, accessed as rchrate[(nodes * per) + i + 1]
		rchrate = [-1, 5, 7, 11, 13, 17, 19]
		rch_out = m.get_mf6rch_file(data, rchrate)
		flopy_adaptor.write_mf_gwf_rch(rch_out)

		with open("output_files/run-aardvark.rch") as file:
			contents = file.read()
		
		contents_without_first_line = contents.split("\n", 1)[1]
		expected = """BEGIN options
END options

BEGIN dimensions
  MAXBOUND  3
END dimensions

BEGIN period  1
  1 1       0.00500000
  1 2       0.00700000
  
END period  1

BEGIN period  2
  1 1       0.01300000
  1 2       0.01700000
  
END period  2

"""
		self.assertEqual(expected, contents_without_first_line)
