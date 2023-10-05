import math
import numpy as np

def calculate_nitrate(data, output, node):
	her_array_mm_per_day = _calculate_her_mm_per_day(data, output, node)
	m0_array_kg_per_day = _calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day)
	m1_arr_kg_per_day = _calculate_m1_arr_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
	m1a_arr_kg_per_day = _calculate_m1a_arr_kg_per_day(data, output, node, m1_arr_kg_per_day)
	m2_arr_kg_per_day = _calculate_m2_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
	m3_kg_per_day = _calculate_m3_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
	mi_kg_per_day = _calculate_mi_kg_per_day(m1a_arr_kg_per_day, m2_arr_kg_per_day)

def _calculate_her_mm_per_day(data, output, node):
	return output["rainfall_ts"] - output["ae"]

def _calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day):
	params = data["params"]
	cell_area_m_sq = params["node_areas"][node][0]
	days = data["series"]["date"]

	nitrate_leaching = params["nitrate_leaching"][node]
	max_load_per_year_kg_per_hectare = nitrate_leaching[3]
	her_at_5_percent = nitrate_leaching[4]
	her_at_50_percent = nitrate_leaching[5]
	her_at_95_percent = nitrate_leaching[6]

	hectare_area_m_sq = 10000
	max_load_per_year_kg_per_cell = max_load_per_year_kg_per_hectare * cell_area_m_sq / hectare_area_m_sq

	m0_array_kg_per_day = _calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_array_mm_per_day)
	return m0_array_kg_per_day

def _calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_per_day):
	length = len(days)
	result = np.zeros(length)
	remaining_for_year = max_load_per_year_kg_per_cell
	for i in range(length):
		day = days[i]
		her = her_per_day[i]
		if (day.month == 10) and (day.day == 1):
			remaining_for_year = max_load_per_year_kg_per_cell
		fraction_leached = _cumulative_fraction_leaked_per_day(her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			her)
		mass_leached_for_day = min(remaining_for_year, max_load_per_year_kg_per_cell * fraction_leached)
		remaining_for_year -= mass_leached_for_day
		result[i] = mass_leached_for_day
	return result

def _cumulative_fraction_leaked_per_day(her_at_5_percent, her_at_50_percent, her_at_95_percent, her_per_day):
	days_in_year = 365.25
	her_per_year = days_in_year * her_per_day
	y = _cumulative_fraction_leaked_per_year(her_at_5_percent, her_at_50_percent, her_at_95_percent, her_per_year)
	return y / days_in_year

def _cumulative_fraction_leaked_per_year(her_at_5_percent, her_at_50_percent, her_at_95_percent, her_per_year):
	x = her_per_year
	is_below_50_percent = her_per_year < her_at_50_percent
	upper = her_at_50_percent if is_below_50_percent else her_at_95_percent
	lower = her_at_5_percent if is_below_50_percent else her_at_50_percent
	# y = mx + c
	m = 0.45 / (upper - lower)
	c = 0.5 - (her_at_50_percent * m)
	y = (m * x) + c
	return y

def _calculate_m1_arr_kg_per_day(data, output, node, her_array_mm_per_day, m0_kg_per_day):
	perc_through_root_mm_per_day = output["perc_through_root"]
	pp = perc_through_root_mm_per_day / her_array_mm_per_day
	m1_kg_per_day = pp * m0_kg_per_day
	return m1_kg_per_day

def _calculate_m1a_arr_kg_per_day(data, output, node, m1_arr_kg_per_day):
	interflow_volume_mm = output["interflow_volume"]
	infiltration_recharge_mm_per_day = output["infiltration_recharge"]
	interflow_to_rivers_mm_per_day = output["interflow_to_rivers"]

	soil_percolation_mm_per_day = interflow_volume_mm + infiltration_recharge_mm_per_day + interflow_to_rivers_mm_per_day
	proportion = interflow_volume_mm / soil_percolation_mm_per_day

	length = m1_arr_kg_per_day.size
	m1a_arr_kg_per_day = np.zeros(length)
	mit_kg = 0

	for i in range(length):
		mit_kg += m1_arr_kg_per_day[i]
		m1a_kg_per_day = mit_kg * proportion[i]
		mit_kg -= m1a_kg_per_day
		m1a_arr_kg_per_day[i] = m1a_kg_per_day
	
	return m1a_arr_kg_per_day

def _calculate_m2_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day):
	runoff_recharge_mm_per_day = output["runoff_recharge"]
	macropore_att_mm_per_day = output["macropore_att"]
	macropore_dir_mm_per_day = output["macropore_dir"]
	macropore_mm_per_day = macropore_att_mm_per_day + macropore_dir_mm_per_day
	p_non = (runoff_recharge_mm_per_day + macropore_mm_per_day) / her_array_mm_per_day
	m2_kg_per_day = m0_array_kg_per_day * p_non
	return m2_kg_per_day

def _calculate_m3_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day):
	runoff_mm_per_day = output["rapid_runoff"]
	runoff_recharge_mm_per_day = output["runoff_recharge"]
	m3_kg_per_day = (m0_array_kg_per_day
		* (runoff_mm_per_day - runoff_recharge_mm_per_day)
		/ her_array_mm_per_day)
	return m3_kg_per_day

def _calculate_mi_kg_per_day(m1a_arr_kg_per_day, m2_arr_kg_per_day):
	return m1a_arr_kg_per_day + m2_arr_kg_per_day

def _calculate_daily_proportion_reaching_water_table_arr(data, output, node):
	length = data["series"]["date"].size
	depth_to_water_m = data["params"]["nitrate_depth_to_water"][node][0]
	result = np.zeros(length)
	for i in range(length):
		result[i] = _calculate_daily_proportion_reaching_water_table(depth_to_water_m, i)
	return result

def _calculate_daily_proportion_reaching_water_table(DTW, t):
	f_t = _calculate_cumulative_proportion_reaching_water_table(DTW, t)
	f_t_prev = _calculate_cumulative_proportion_reaching_water_table(DTW, t - 1)
	return -(f_t - f_t_prev)

def _calculate_cumulative_proportion_reaching_water_table(DTW, t):
	if (t <= 0):
		return 1

	a = 1.38
	μ = 1.58
	σ = 3.96

	numerator = math.log((1.7/0.0029) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * math.erfc(- numerator / denominator)
	return result

def _calculate_total_mass_on_day_kg(daily_proportion_reaching_water_table, mi_kg_per_day):
	length = daily_proportion_reaching_water_table.size
	result_kg = np.zeros(length)
	for day_nitrate_was_leached in range(length):
		mass_leached_on_day_kg = mi_kg_per_day[day_nitrate_was_leached]
		for day_proportion_reaches_water_table in range(length - day_nitrate_was_leached):
			day = day_nitrate_was_leached + day_proportion_reaches_water_table
			proportion = daily_proportion_reaching_water_table[day_proportion_reaches_water_table]
			mass_reaching_water_table_kg = proportion * mass_leached_on_day_kg
			result_kg[day] += mass_reaching_water_table_kg
	return result_kg
