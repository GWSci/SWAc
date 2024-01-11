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
	date = blackboard.date
	if len(date) == 0:
		return historical_nitrate_date
	first_new_date = date[0]
	truncated_historical_nitrate_dates = [d for d in historical_nitrate_date if d < first_new_date]
	return truncated_historical_nitrate_dates
