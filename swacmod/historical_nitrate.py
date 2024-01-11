import numpy as np
import math

class HistoricalNitrateBlackboard():
	def __init__(self):
		self.a = None
		self.date = None
		self.historical_mi_array_kg_per_day = None
		self.historical_nitrate_date = None
		self.mean_hydraulic_conductivity = None
		self.mean_velocity_of_unsaturated_transport = None
		self.nitrate_depth_to_water = None
		self.time_switcher = None
		self.truncated_historical_mi_array_kg_per_day = None
		self.truncated_historical_nitrate_dates = None
		self.μ = None
		self.σ = None

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}

def _calculate_historical_nitrate(blackboard):
	blackboard.truncated_historical_nitrate_dates = _calculate_truncated_historical_nitrate_date(blackboard)
	blackboard.truncated_historical_mi_array_kg_per_day = _calculate_truncated_historical_mi_array_kg_per_day(blackboard)
	return blackboard

def _calculate_truncated_historical_nitrate_date(blackboard):
	historical_nitrate_date = blackboard.historical_nitrate_date
	date = blackboard.date
	if len(date) == 0:
		return historical_nitrate_date
	first_new_date = date[0]
	truncated_historical_nitrate_dates = [d for d in historical_nitrate_date if d < first_new_date]
	return truncated_historical_nitrate_dates

def _calculate_truncated_historical_mi_array_kg_per_day(blackboard):
	truncated_length = len(blackboard.truncated_historical_nitrate_dates)
	return blackboard.historical_mi_array_kg_per_day[:truncated_length]

def _calculate_historic_proportion_reaching_water_table_array_per_day(blackboard):	
	time_switcher = blackboard.time_switcher
	length = len(blackboard.date)
	days_offset = len(blackboard.historical_nitrate_date)
	depth_to_water_m = blackboard.nitrate_depth_to_water[0]
	if depth_to_water_m == 0.0:
		return blackboard.proportion_0
	elif depth_to_water_m == 100.0:
		return blackboard.proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(days_offset, length, blackboard.a, blackboard.μ, blackboard.σ, blackboard.mean_hydraulic_conductivity, blackboard.mean_velocity_of_unsaturated_transport, depth_to_water_m, time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(days_offset, length, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, time_switcher):
	result = np.zeros(length)
	for i in range(length):
		t = days_offset + i
		result[i] = _calculate_daily_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, t)
	return result

def _calculate_daily_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	f_t = _calculate_cumulative_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
	f_t_prev = _calculate_cumulative_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t - 1)
	return f_t - f_t_prev

def _calculate_cumulative_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	if (t <= 0):
		return 0

	numerator = math.log((mean_hydraulic_conductivity/mean_velocity_of_unsaturated_transport) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * (1 + math.erf(- numerator / denominator))
	return result
