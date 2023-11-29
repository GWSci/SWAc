import swacmod.csv_resource as csv_resource
import unittest

class Test_Csv_Resource(unittest.TestCase):
	def test_parse_utf8_file(self):
		csv_file="./test/resources/csv_file_utf8/example_utf8_csv_file.csv"
		with(csv_resource.reader_for(csv_file)) as testee:
			actual = [[int(cell) for cell in row] for row in testee]
		expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
		self.assertEqual(expected, actual)
