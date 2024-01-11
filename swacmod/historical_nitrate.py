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

def _calculate_historical_nitrate_dates(blackboard):
	pass
