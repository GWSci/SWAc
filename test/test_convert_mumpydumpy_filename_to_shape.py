import unittest
import ast

class Test_Convert_Numpydumpy_Filename_To_Shape(unittest.TestCase):
    def test_y(self):
        expected = (123, 456)
        actual = convert_numpydumpy_filename_to_shape("/aardvark/bat/cat/dog.csv.(123, 456).numpydumpy")
        self.assertEqual(expected, actual)

def convert_numpydumpy_filename_to_shape(filename):
    parts = filename.split(".")
    shape_string = parts[-2]
    shape_tuple = ast.literal_eval(shape_string)
    return shape_tuple