import unittest
import swacmod.utils as utils
from datetime import date, timedelta

class test_month_delta(unittest.TestCase):
	def test_characterise_monthdelta(self):
		self.assertEqual(0, utils.monthdelta_old(date(1995, 1, 1), date(1995, 1, 1)))
		self.assertEqual(0, utils.monthdelta_old(date(1995, 1, 1), date(1995, 1, 2)))
		self.assertEqual(1, utils.monthdelta_old(date(1995, 1, 1), date(1995, 2, 1)))
		self.assertEqual(2, utils.monthdelta_old(date(1995, 1, 1), date(1995, 3, 1)))
		self.assertEqual(0, utils.monthdelta_old(date(1995, 1, 10), date(1995, 2, 1)))
		self.assertEqual(1, utils.monthdelta_old(date(1995, 1, 27), date(1995, 2, 27)))
		self.assertEqual(0, utils.monthdelta_old(date(1995, 1, 28), date(1995, 2, 27)))
		self.assertEqual(12, utils.monthdelta_old(date(1995, 1, 1), date(1996, 1, 1)))
	
	def test_monthdelta2(self):
		start_date = date(1995, 1, 1)
		end_date = date(1997, 1, 1)
		one_day = timedelta(days=1)
		d = date(1995, 1, 1)
		
		while d < end_date:
			expected = utils.monthdelta_old(start_date, d)
			actual = utils.monthdelta(start_date, d)
			self.assertEqual(expected, actual)
			d += one_day
