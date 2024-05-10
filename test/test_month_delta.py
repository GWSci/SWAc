import unittest
import swacmod.utils as utils
from datetime import date, timedelta
import datetime

class test_month_delta(unittest.TestCase):
	def test_characterise_monthdelta(self):
		self.assertEqual(0, monthdelta_old(date(1995, 1, 1), date(1995, 1, 1)))
		self.assertEqual(0, monthdelta_old(date(1995, 1, 1), date(1995, 1, 2)))
		self.assertEqual(1, monthdelta_old(date(1995, 1, 1), date(1995, 2, 1)))
		self.assertEqual(2, monthdelta_old(date(1995, 1, 1), date(1995, 3, 1)))
		self.assertEqual(0, monthdelta_old(date(1995, 1, 10), date(1995, 2, 1)))
		self.assertEqual(1, monthdelta_old(date(1995, 1, 27), date(1995, 2, 27)))
		self.assertEqual(0, monthdelta_old(date(1995, 1, 28), date(1995, 2, 27)))
		self.assertEqual(12, monthdelta_old(date(1995, 1, 1), date(1996, 1, 1)))
	
	def test_monthdelta2(self):
		start_date = date(1995, 1, 1)
		end_date = date(1997, 1, 1)
		one_day = timedelta(days=1)
		d = date(1995, 1, 1)
		
		while d < end_date:
			expected = monthdelta_old(start_date, d)
			actual = utils.monthdelta(start_date, d)
			self.assertEqual(expected, actual)
			d += one_day

def monthdelta_old(d1, d2):
    " difference in months between two dates"

    from calendar import monthrange

    delta = 0
    while True:
        mdays = monthrange(d1.year, d1.month)[1]
        d1 += datetime.timedelta(days=mdays)
        if d1 <= d2:
            delta += 1
        else:
            break
    return delta
