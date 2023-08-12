import unittest
import swacmod.timer as timer

class test_accumulation_timer(unittest.TestCase):
    def test_accumulation_timer_is_initially_zero(self):
        actual = timer.make_accumulation_timer("aardvark")["elapsed_seconds"]
        expected = 0
        self.assertEqual(expected, actual)
        
    def test_accumulation_timer_has_message_set(self):
        actual = timer.make_accumulation_timer("aardvark")["message"]
        expected = "aardvark"
        self.assertEqual(expected, actual)

    def test_accumulation_timer_starting_and_stopping_updates_the_elapsed_time(self):
        time = mock_time([2, 3, 5, 7])
        timer_token = timer.make_accumulation_timer("aardvark")
        
        timer.continue_timing(timer_token, time=time)
        timer.stop_timing(timer_token, time=time)
        timer.continue_timing(timer_token, time=time)
        timer.stop_timing(timer_token, time=time)
        
        actual = timer_token["elapsed_seconds"]
        expected = 3
        self.assertEqual(expected, actual)

    def test_accumulation_timer_continually_stopping_the_timer_does_not_repeatedly_update_the_elapsed_time(self):
        time = mock_time([2, 3, 5, 7])
        timer_token = timer.make_accumulation_timer("aardvark")
        
        timer.continue_timing(timer_token, time=time)
        timer.stop_timing(timer_token, time=time)
        timer.stop_timing(timer_token, time=time)
        timer.stop_timing(timer_token, time=time)
        
        actual = timer_token["elapsed_seconds"]
        expected = 1
        self.assertEqual(expected, actual)

class mock_time():
	def __init__(self, times):
		self.times = list(times)

	def time(self):
         return self.times.pop(0)