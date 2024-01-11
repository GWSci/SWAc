import numpy as np

class HistoricalNitrateBlackboard():
	def __init__(self):
		self.date = None
		self.historical_nitrate_date = None

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}

def _calculate_truncated_historical_nitrate_date(blackboard):
	historical_nitrate_date = blackboard.historical_nitrate_date
	new_dates = blackboard.date
	if len(new_dates) == 0:
		return historical_nitrate_date
	first_new_date = new_dates[0]
	truncated_historical_nitrate_dates = [date for date in historical_nitrate_date if date < first_new_date]
	return truncated_historical_nitrate_dates
