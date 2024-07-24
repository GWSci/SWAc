import unittest
import numpy as np
import swacmod.h5py_adaptor as h5py_adaptor
import tempfile

class Test_H5py_Adaptor(unittest.TestCase):
	def test_write_and_read(self):
		path = tempfile.mkstemp()[1]
		input_array = np.array([2, 3, 5, 7, 11])
		h5py_adaptor.write_h5py(path, "aardvark", input_array)

		actual = h5py_adaptor.read_h5py(path)["aardvark"]

		np.testing.assert_array_equal(input_array, actual)
