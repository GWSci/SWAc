import unittest
import os
import swacmod.psutil_adaptor as psutil_adaptor

class Test_Psutil_Adaptor(unittest.TestCase):
	def test_get_ram_usage_includes_rss(self):
		pid = os.getpid()
		memory_info = psutil_adaptor.memory_info_for_pid(pid)
		actual = memory_info.rss
		message = f"Actual RAM usage was {actual}."
		self.assertTrue(actual > 0, msg = message)