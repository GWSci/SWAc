import csv
import logging
import math
import numpy as np
import os
import swacmod.feature_flags as ff
import swacmod.timer as timer
import swacmod.utils as utils
import swacmod.model as m

def get_nitrate(data, output, node):
	nitrate = calculate_nitrate(data, output, node)
	return {
		"nitrate_reaching_water_table_array_tons_per_day" : nitrate["nitrate_reaching_water_table_array_tons_per_day"],
	}

def calculate_nitrate(data, output, node):
	length = output["rainfall_ts"].size
	time_switcher = data["time_switcher"]
	if "enabled" == data["params"]["nitrate_process"]:
		proportion_0 = np.zeros(length)
		proportion_100 = data["proportion_100"]

		# timer.switch_to(time_switcher, "Nitrate: _calculate_her_array_mm_per_day")
		her_array_mm_per_day = _calculate_her_array_mm_per_day(data, output, node)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_m0_array_kg_per_day")
		m0_array_kg_per_day = _calculate_m0_array_kg_per_day(data, output, node, her_array_mm_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_m1_array_kg_per_day")
		m1_array_kg_per_day = _calculate_m1_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_m1a_array_kg_per_day")
		m1a_array_kg_per_day = _calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_m2_array_kg_per_day")
		m2_array_kg_per_day = _calculate_m2_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_m3_array_kg_per_day")
		m3_array_kg_per_day = _calculate_m3_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_mi_array_kg_per_day")
		mi_array_kg_per_day = _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _check_masses_balance")
		_check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_proportion_reaching_water_table_array_per_day")
		proportion_reaching_water_table_array_per_day = _calculate_proportion_reaching_water_table_array_per_day(data, output, node, proportion_0, proportion_100)

		# timer.switch_to(time_switcher, "Nitrate: _calculate_mass_reaching_water_table_array_kg_per_day")
		nitrate_reaching_water_table_array_kg_per_day = np.array(m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day))

		# timer.switch_to(time_switcher, "Nitrate: _convert_kg_to_tons_array")
		nitrate_reaching_water_table_array_tons_per_day = _convert_kg_to_tons_array(nitrate_reaching_water_table_array_kg_per_day)

		# timer.switch_to(time_switcher, "Nitrate: return")
		return {
			"her_array_mm_per_day" : her_array_mm_per_day,
			"m0_array_kg_per_day" : m0_array_kg_per_day,
			"m1_array_kg_per_day" : m1_array_kg_per_day,
			"m1a_array_kg_per_day" : m1a_array_kg_per_day,
			"m2_array_kg_per_day" : m2_array_kg_per_day,
			"m3_array_kg_per_day" : m3_array_kg_per_day,
			"mi_array_kg_per_day" : mi_array_kg_per_day,
			"proportion_reaching_water_table_array_per_day" : proportion_reaching_water_table_array_per_day,
			"nitrate_reaching_water_table_array_kg_per_day" : nitrate_reaching_water_table_array_kg_per_day,
			"nitrate_reaching_water_table_array_tons_per_day" : nitrate_reaching_water_table_array_tons_per_day,
		}
	else:
		empty_array = np.zeros(length)
		return {
			"her_array_mm_per_day" : empty_array,
			"m0_array_kg_per_day" : empty_array,
			"m1_array_kg_per_day" : empty_array,
			"m1a_array_kg_per_day" : empty_array,
			"m2_array_kg_per_day" : empty_array,
			"m3_array_kg_per_day" : empty_array,
			"mi_array_kg_per_day" : empty_array,
			"proportion_reaching_water_table_array_per_day" : empty_array,
			"nitrate_reaching_water_table_array_kg_per_day" : empty_array,
			"nitrate_reaching_water_table_array_tons_per_day" : empty_array,
		}


def _calculate_her_array_mm_per_day(data, output, node):
	return np.maximum(0.0, output["rainfall_ts"] - output["ae"])

def _calculate_m0_array_kg_per_day(data, output, node, her_array_mm_per_day):
	time_switcher = data["time_switcher"]
	
	params = data["params"]
	cell_area_m_sq = params["node_areas"][node]
	days = data["series"]["date"]

	nitrate_loading = params["nitrate_loading"][node]
	max_load_per_year_kg_per_hectare = nitrate_loading[3]
	her_at_5_percent = nitrate_loading[4]
	her_at_50_percent = nitrate_loading[5]
	her_at_95_percent = nitrate_loading[6]

	hectare_area_m_sq = 10000
	max_load_per_year_kg_per_cell = max_load_per_year_kg_per_hectare * cell_area_m_sq / hectare_area_m_sq

	m0_array_kg_per_day = m._calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_array_mm_per_day,
		time_switcher)
	return m0_array_kg_per_day

def _calculate_m1_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_kg_per_day):
	perc_through_root_mm_per_day = output["perc_through_root"]
	pp = _divide_arrays(perc_through_root_mm_per_day, her_array_mm_per_day)
	m1_kg_per_day = pp * m0_kg_per_day
	return m1_kg_per_day

def _divide_arrays(a, b):
	return np.divide(a, b, out = np.zeros_like(a), where = b != 0)

def _calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day):
	return m._calculate_m1a_array_kg_per_day(output, m1_array_kg_per_day)

def _calculate_m2_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day):
	runoff_recharge_mm_per_day = output["runoff_recharge"]
	macropore_att_mm_per_day = output["macropore_att"]
	macropore_dir_mm_per_day = output["macropore_dir"]
	macropore_mm_per_day = macropore_att_mm_per_day + macropore_dir_mm_per_day
	p_non = _divide_arrays((runoff_recharge_mm_per_day + macropore_mm_per_day), her_array_mm_per_day)
	m2_kg_per_day = m0_array_kg_per_day * p_non
	return m2_kg_per_day

def _calculate_m3_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day):
	runoff_mm_per_day = output["rapid_runoff"]
	runoff_recharge_mm_per_day = output["runoff_recharge"]
	m3_array_kg_per_day = _divide_arrays(
		m0_array_kg_per_day * (runoff_mm_per_day - runoff_recharge_mm_per_day),
		her_array_mm_per_day)
	return m3_array_kg_per_day

def _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day):
	return m1a_array_kg_per_day + m2_array_kg_per_day

def _check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day):
	m0_kg = m1_array_kg_per_day + m2_array_kg_per_day + m3_array_kg_per_day
	is_m0_as_expected = np.allclose(m0_kg, m0_array_kg_per_day)
	if not is_m0_as_expected:
		for i in range(m0_kg.size):
			if not np.isclose(m0_kg[i], m0_array_kg_per_day[i]):
				break
		m0 = m0_array_kg_per_day[i]
		m1 = m1_array_kg_per_day[i]
		m2 = m2_array_kg_per_day[i]
		m3 = m3_array_kg_per_day[i]
		message = f"Nitrate masses do not balance for node {node} using the equation M0 = M1 + M2 + M3. The first day that does not balance is at index {i}. M0 = {m0} kg; M1 = {m1} kg; M2 = {m2} kg and M3 = {m3} kg."
		logging.warning(message)

def _calculate_proportion_reaching_water_table_array_per_day(data, output, node, proportion_0, proportion_100):	
	time_switcher = data["time_switcher"]
	length = len(data["series"]["date"])
	depth_to_water_m = data["params"]["nitrate_depth_to_water"][node][0]
	if depth_to_water_m == 0.0:
		return proportion_0
	elif depth_to_water_m == 100.0:
		return proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(length, depth_to_water_m, time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(length, depth_to_water_m, time_switcher):
	result = np.zeros(length)
	for i in range(length):
		result[i] = _calculate_daily_proportion_reaching_water_table(depth_to_water_m, i)
	return result

def _calculate_daily_proportion_reaching_water_table(DTW, t):
	f_t = _calculate_cumulative_proportion_reaching_water_table(DTW, t)
	f_t_prev = _calculate_cumulative_proportion_reaching_water_table(DTW, t - 1)
	return f_t - f_t_prev

def _calculate_cumulative_proportion_reaching_water_table(DTW, t):
	if (t <= 0):
		return 0

	a = 1.38
	μ = 1.58
	σ = 3.96

	numerator = math.log((1.7/0.0029) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * (1 + math.erf(- numerator / denominator))
	return result

def _convert_kg_to_tons_array(arr_kg):
	return arr_kg / 1000.0

def make_aggregation_array(data):
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	shape = (_len_time_periods(time_periods), len(node_areas))
	aggregation = np.zeros(shape = shape)
	return aggregation

def aggregate_nitrate(aggregation, data, output, node):
	time_periods = data["params"]["time_periods"]
	nitrate_reaching_water_table_array_tons_per_day = output["nitrate_reaching_water_table_array_tons_per_day"]
	combined_recharge_m_cubed = _calculate_combined_recharge_m_cubed(data, output, node)

	len_time_periods = _len_time_periods(time_periods)
	m._aggregate_nitrate(time_periods, len_time_periods, nitrate_reaching_water_table_array_tons_per_day, combined_recharge_m_cubed, aggregation, node)
	return aggregation

def _len_time_periods(time_periods):
	if ff.use_node_count_override:
		len_time_periods = 0
		while time_periods[len_time_periods][1] < ff.max_node_count_override:
			len_time_periods += 1
		return len_time_periods
	else:
		return len(time_periods)
	

def _calculate_combined_recharge_m_cubed(data, output, node):
	node_areas = data["params"]["node_areas"]
	combined_recharge_mm = output["combined_recharge"]
	combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
	combined_recharge_m_cubed = combined_recharge_m * node_areas[node]
	return combined_recharge_m_cubed

def _convert_mm_to_m(arr):
	return arr / 100.0

def write_nitrate_csv(data, nitrate_aggregation, open=open):
	filename = make_output_filename(data)
	with open(filename, "w", newline="") as f:
		writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, dialect='excel')
		writer.writerow(["Stress Period", "Node", "Recharge Concentration (metric tons/m3)"])
		for stress_period_index, node_index in np.ndindex(nitrate_aggregation.shape):
			stress_period = stress_period_index + 1
			node = node_index + 1
			recharge_concentration = nitrate_aggregation[stress_period_index, node_index]
			writer.writerow([stress_period, node, recharge_concentration])

def make_output_filename(data):
	run_name = data["params"]["run_name"]
	file = run_name + "_nitrate.csv"
	folder = utils.CONSTANTS["OUTPUT_DIR"]
	return os.path.join(folder, file)
 