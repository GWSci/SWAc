import unittest
import swacmod.timer as timer

class test_time_switcher(unittest.TestCase):
	def test_time_switcher_initially_has_an_empty_report(self):
		time_switcher = timer.make_time_switcher()
		actual = timer._time_switcher_report(time_switcher)
		expected = []
		self.assertEqual(expected, actual)

	def test_time_switcher_calling_switch_off_when_not_started_makes_no_change(self):
		time_switcher = timer.make_time_switcher()
		timer.switch_off(time_switcher)
		actual = timer._time_switcher_report(time_switcher)
		expected = []
		self.assertEqual(expected, actual)

	def test_time_switcher_calling_switch_to_does_not_immediately_update_the_report(self):
		time = mock_time([2, 3])
		time_switcher = timer.make_time_switcher()
		timer.switch_to(time_switcher, "aardvark")
		actual = timer._time_switcher_report(time_switcher)
		expected = []
		self.assertEqual(expected, actual)

	def test_time_switcher_has_one_row_after_starting_and_stopping(self):
		time = mock_time([2, 3])
		time_switcher = timer.make_time_switcher()
		timer.switch_to(time_switcher, "aardvark", time)
		timer.switch_off(time_switcher, time)
		actual = timer._time_switcher_report(time_switcher)
		expected = [{"message": "aardvark", "elapsed_seconds": 1}]
		self.assertEqual(expected, actual)

	def test_time_switcher_has_several_rows_after_switching(self):
		time = mock_time([2, 3, 5, 7, 11, 13])
		time_switcher = timer.make_time_switcher()
		timer.switch_to(time_switcher, "aardvark", time)
		timer.switch_to(time_switcher, "bat", time)
		timer.switch_to(time_switcher, "cat", time)
		timer.switch_off(time_switcher, time)
		actual = timer._time_switcher_report(time_switcher)
		expected = [
			{"message": "aardvark", "elapsed_seconds": 1},
			{"message": "bat", "elapsed_seconds": 2},
			{"message": "cat", "elapsed_seconds": 2},
		]
		self.assertEqual(expected, actual)

	def test_time_switcher_updates_times_when_revisiting(self):
		time = mock_time([2, 3, 5, 7, 11, 13, 17, 19])
		time_switcher = timer.make_time_switcher()
		timer.switch_to(time_switcher, "aardvark", time)
		timer.switch_to(time_switcher, "bat", time)
		timer.switch_to(time_switcher, "cat", time)
		timer.switch_to(time_switcher, "aardvark", time)
		timer.switch_off(time_switcher, time)
		actual = timer._time_switcher_report(time_switcher)
		expected = [
			{"message": "aardvark", "elapsed_seconds": 3},
			{"message": "bat", "elapsed_seconds": 2},
			{"message": "cat", "elapsed_seconds": 2},
		]
		self.assertEqual(expected, actual)

	def assert_report(self, time_switcher, expected_report):
		actual = timer._time_switcher_report(time_switcher)
		self.assertEqual(expected_report, actual)

class mock_time():
	def __init__(self, times):
		self.times = list(times)

	def time(self):
         return self.times.pop(0)