import unittest
import swacmod.timer as timer

class test_timer_print_table(unittest.TestCase):
	def test_x(self):
		input_tokens = []
		actual = timer.make_time_table(input_tokens)
		expected = []
		self.assertEqual(expected, actual)
		