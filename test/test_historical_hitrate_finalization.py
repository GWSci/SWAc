import datetime
import unittest
import swacmod.finalization as finalization

class Test_Historical_Nitrate_Finalization(unittest.TestCase):
	def test_fin_historical_start_date_when_valid(self):
		data = {
			"params" : {
				"historical_start_date" : "2024-01-16",
			}
		}
		name = "historical_start_date"
		finalization.fin_historical_start_date(data, name)

		actual = data["params"]["historical_start_date"]
		expected = datetime.datetime(2024, 1, 16)
		self.assertEqual(expected, actual)
