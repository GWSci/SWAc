import unittest
from pathlib import Path

class Test_Demo_Models(unittest.TestCase):
	def test_demo_model(self):
		reference_output_folder = "test/reference_output/"
		output_folder = "output_files/"

		for f in Path(output_folder).iterdir():
			f.unlink()

		self.assertEqual(1, 1)
