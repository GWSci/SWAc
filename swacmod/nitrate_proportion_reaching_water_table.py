import math
import numpy as np

def calculate_proportion_reaching_water_table_array_per_day(blackboard):
	historical_days_count = 0
	return _calculate_proportion_reaching_water_table_array_per_day(blackboard, historical_days_count)

def calculate_historic_proportion_reaching_water_table_array_per_day(blackboard):	
	historical_days_count = len(blackboard.truncated_historical_nitrate_days)
	return _calculate_proportion_reaching_water_table_array_per_day(blackboard, historical_days_count)

def _calculate_proportion_reaching_water_table_array_per_day(blackboard, historical_days_count):
	time_switcher = blackboard.time_switcher
	length = historical_days_count + len(blackboard.days)
	depth_to_water_m = blackboard.nitrate_depth_to_water[0]
	if depth_to_water_m == 0.0:
		return blackboard.proportion_0[:length]
	elif depth_to_water_m == 100.0:
		return blackboard.proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(
			length,
			blackboard.a,
			blackboard.μ,
			blackboard.σ,
			blackboard.mean_hydraulic_conductivity,
			blackboard.mean_velocity_of_unsaturated_transport,
			depth_to_water_m,
			time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(
		length,
		a,
		μ,
		σ,
		mean_hydraulic_conductivity,
		mean_velocity_of_unsaturated_transport,
		depth_to_water_m,
		time_switcher):
	result = np.zeros(length)
	for i in range(length):
		t = i
		result[i] = _calculate_daily_proportion_reaching_water_table(
			a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, t)
	return result

def _calculate_daily_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	f_t = _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
	f_t_prev = _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t - 1)
	return f_t - f_t_prev

def _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	if (t <= 0):
		return 0

	numerator = math.log((mean_hydraulic_conductivity/mean_velocity_of_unsaturated_transport) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * (1 + math.erf(- numerator / denominator))
	return result
