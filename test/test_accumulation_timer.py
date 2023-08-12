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
