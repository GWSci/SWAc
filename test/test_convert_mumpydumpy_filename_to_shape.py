from swacmod.time_series_data import convert_numpydumpy_filename_to_shape
import unittest

class Test_Convert_Numpydumpy_Filename_To_Shape(unittest.TestCase):
    def test_y(self):
        expected = (123, 456)
        actual = convert_numpydumpy_filename_to_shape("/aardvark/bat/cat/dog.csv.(123, 456).numpydumpy")
        self.assertEqual(expected, actual)
