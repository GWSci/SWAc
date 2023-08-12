import unittest
import swacmod.timer as timer

class test_time_switcher(unittest.TestCase):
    def test_time_switcher_initially_has_an_empty_report(self):
        time_switcher = timer.make_time_switcher()
        actual = timer.time_switcher_report(time_switcher)
        expected = []
        self.assertEqual(expected, actual)

    def test_time_switcher_calling_switch_off_when_not_started_makes_no_change(self):
        time_switcher = timer.make_time_switcher()
        timer.switch_off(time_switcher)
        actual = timer.time_switcher_report(time_switcher)
        expected = []
        self.assertEqual(expected, actual)

    def test_time_switcher_has_one_row_after_starting_and_stopping(self):
        time = mock_time([2, 3])
        time_switcher = timer.make_time_switcher()
        timer.switch_to(time_switcher, "aardvark")
        timer.switch_off(time_switcher)
        actual = timer.time_switcher_report(time_switcher)
        expected = [{"message": "aardvark", "elapsed_seconds": 1}]
        self.assertEqual(expected, actual)

class mock_time():
	def __init__(self, times):
		self.times = list(times)

	def time(self):
         return self.times.pop(0)