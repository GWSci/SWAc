import unittest

class Test_Model_Number_to_Lrc(unittest.TestCase):
	def test_modflow_dis_get_lrc_when_node_number_is_out_of_range(self):
		self.assert_dis_get_lrc_exception_with_message("The node number 0 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", 0)
		self.assert_dis_get_lrc_exception_with_message("The node number -1 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", -1)
		self.assert_dis_get_lrc_exception_with_message("The node number 31 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", 31)
		self.assert_dis_get_lrc_exception_with_message("The node number 32 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", 32)

		self.assert_dis_get_lrc_exception_with_message("The node number 0 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", [0])
		self.assert_dis_get_lrc_exception_with_message("The node number 0 is out of bounds. Node numbers muse be in the range 1--30. Layer, row and column counts are 2, 3, 5 respectively.", [1, 0])

	def assert_dis_get_lrc_exception_with_message(self, expected_message, input_node_numbers):
		with self.assertRaises(Exception) as ex:
			dis_get_lrc(2, 3, 5, input_node_numbers)
		self.assertTrue(expected_message in str(ex.exception))

	def test_modflow_dis_get_lrc_when_the_input_is_within_the_model_bounds(self):
		#  - The output layer is zero-based.
		#  - The output row is zero-based.
		#  - The output column is one-based.

		# Numbers 1-5. Layer 0, Row 0, Columns 1-5.
		self.assertEqual([(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)], dis_get_lrc(2, 3, 5, [1, 2, 3, 4, 5]))

		# Numbers 6-10. Layer 0, Row 1, Columns 1-5.
		self.assertEqual([(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)], dis_get_lrc(2, 3, 5, [6, 7, 8, 9, 10]))

		# Numbers 11-15. Layer 0, Row 2, Columns 1-5.
		self.assertEqual([(0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), (0, 2, 5)], dis_get_lrc(2, 3, 5, [11, 12, 13, 14, 15]))

		# Numbers 16-20. Layer 1, Row 0, Columns 1-5
		self.assertEqual([(1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 0, 4), (1, 0, 5)], dis_get_lrc(2, 3, 5, [16, 17, 18, 19, 20]))

		# Numbers 21-25. Layer 1, Row 1, Columns 1-5
		self.assertEqual([(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5)], dis_get_lrc(2, 3, 5, [21, 22, 23, 24, 25]))

		# Numbers 26-30. Layer 1, Row 2, Columns 1-5
		self.assertEqual([(1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5)], dis_get_lrc(2, 3, 5, [26, 27, 28, 29, 30]))

def dis_get_lrc(nlay, nrow, ncol, node_numbers):
	if isinstance(node_numbers, list):
		node_number_list = node_numbers
	else:
		node_number_list = [node_numbers]
	_validate_node_numbers(nlay, nrow, ncol, node_number_list)
	lrc_list = []
	for node_number in node_number_list:
		node_index = node_number - 1
		l = node_index // ncol // nrow
		r = (node_index // ncol) % nrow
		c = 1 + (node_index % ncol) # Flopy 3.3.2 had a quirk in get_lrc. The layer and row returned were 0-based but the column was 1-based.
		lrc_list.append((l, r, c))
	return lrc_list

def _validate_node_numbers(nlay, nrow, ncol, node_numbers):
	max_node_number = (nlay * nrow * ncol)
	for node_number in node_numbers:
		if node_number <= 0 or node_number > max_node_number:
			message = f"The node number {node_number} is out of bounds. Node numbers muse be in the range 1--{max_node_number}. Layer, row and column counts are {nlay}, {nrow}, {ncol} respectively."
			raise Exception(message)
