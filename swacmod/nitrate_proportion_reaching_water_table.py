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
		return n__calculate_proportion_reaching_water_table_array_per_day(
			length,
			blackboard.a,
			blackboard.μ,
			blackboard.σ,
			blackboard.mean_hydraulic_conductivity,
			blackboard.mean_velocity_of_unsaturated_transport,
			depth_to_water_m,
			time_switcher)

def n__calculate_proportion_reaching_water_table_array_per_day(
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
		result[i] = n_calculate_daily_proportion_reaching_water_table(
			a,
			μ,
			σ,
			mean_hydraulic_conductivity,
			mean_velocity_of_unsaturated_transport,
			depth_to_water_m,
			i)
	return result

def n_calculate_daily_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	f_t = n_calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
	f_t_prev = n_calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t - 1)
	return f_t - f_t_prev

def n_calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t):
	if (t <= 0):
		return 0

	numerator = math.log((mean_hydraulic_conductivity/mean_velocity_of_unsaturated_transport) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * (1 + math.erf(- numerator / denominator))
	return result
