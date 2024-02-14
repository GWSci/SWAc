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
		result = np.zeros(length)
		result[0] = 1.0
		return result
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(
			length,
			blackboard.a,
			blackboard.μ,
			blackboard.σ,
			blackboard.alpha,
			blackboard.effective_porosity,
			depth_to_water_m,
			time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(
		length,
		a,
		μ,
		σ,
		alpha,
		effective_porosity,
		depth_to_water_m,
		time_switcher):
	result = np.zeros(length)
	for i in range(length):
		t = i
		result[i] = _calculate_daily_proportion_reaching_water_table(
			a, μ, σ, alpha, effective_porosity, depth_to_water_m, t)
	return result

def _calculate_daily_proportion_reaching_water_table(
		a, μ, σ, alpha, effective_porosity, DTW, t):
	
	first_day = 0
	if (DTW == 0 and t == first_day):
		return 1.0

	if (DTW == 0 and t > first_day):
		return 0.0

	f_t = _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, alpha, effective_porosity, DTW, t)
	f_t_prev = _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, alpha, effective_porosity, DTW, t - 1)
	return f_t - f_t_prev

def _calculate_cumulative_proportion_reaching_water_table(
		a, μ, σ, alpha, effective_porosity, DTW, t):
	if (t <= 0):
		return 0

	numerator = math.log((alpha * effective_porosity) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * (1 + math.erf(- numerator / denominator))
	return result
