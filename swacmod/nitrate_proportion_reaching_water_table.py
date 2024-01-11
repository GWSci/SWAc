import math
import numpy as np

def n_calculate_proportion_reaching_water_table_array_per_day(blackboard):
	time_switcher = blackboard.time_switcher
	length = len(blackboard.days)
	depth_to_water_m = blackboard.nitrate_depth_to_water[0]
	if depth_to_water_m == 0.0:
		return blackboard.proportion_0
	elif depth_to_water_m == 100.0:
		return blackboard.proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(
			0,
			length,
			blackboard.a,
			blackboard.μ,
			blackboard.σ,
			blackboard.mean_hydraulic_conductivity,
			blackboard.mean_velocity_of_unsaturated_transport,
			depth_to_water_m,
			time_switcher)

def h_calculate_historic_proportion_reaching_water_table_array_per_day(blackboard):	
	time_switcher = blackboard.time_switcher
	length = len(blackboard.date)
	days_offset = len(blackboard.historical_nitrate_date)
	depth_to_water_m = blackboard.nitrate_depth_to_water[0]
	if depth_to_water_m == 0.0:
		return blackboard.proportion_0
	elif depth_to_water_m == 100.0:
		return blackboard.proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(
			days_offset,
			length,
			blackboard.a,
			blackboard.μ,
			blackboard.σ,
			blackboard.mean_hydraulic_conductivity,
			blackboard.mean_velocity_of_unsaturated_transport,
			depth_to_water_m,
			time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(
		days_offset,
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
		t = days_offset + i
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
