import numpy as np

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}
