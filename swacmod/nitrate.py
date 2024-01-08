import csv
import logging
import math
import numpy as np
import os
import swacmod.feature_flags as ff
import swacmod.utils as utils
import swacmod.model as m

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}

def get_nitrate(data, output, node):
	nitrate = calculate_nitrate(data, output, node)
	return {
		"nitrate_reaching_water_table_array_tons_per_day" : nitrate["nitrate_reaching_water_table_array_tons_per_day"],
	}

def calculate_nitrate(data, output, node, logging = logging):
	length = output["rainfall_ts"].size
	params = data["params"]
	if "enabled" == params["nitrate_process"]:
		
		a = params["nitrate_calibration_a"]
		μ = params["nitrate_calibration_mu"]
		σ = params["nitrate_calibration_sigma"]
		mean_hydraulic_conductivity = params["nitrate_calibration_mean_hydraulic_conductivity"]
		mean_velocity_of_unsaturated_transport = params["nitrate_calibration_mean_velocity_of_unsaturated_transport"]
		proportion_0 = np.zeros(length)
		proportion_100 = data["proportion_100"]

		her_array_mm_per_day = _calculate_her_array_mm_per_day(data, output, node)
		m0_array_kg_per_day = _calculate_m0_array_kg_per_day(data, output, node, her_array_mm_per_day)
		pp = _calculate_pp(data, output, node, her_array_mm_per_day)
		m1_array_kg_per_day = _calculate_m1_array_kg_per_day(m0_array_kg_per_day, pp)
		m1a_array_kg_per_day = _calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day)
		p_non = _calculate_p_non(data, output, node, her_array_mm_per_day)
		m2_array_kg_per_day = _calculate_m2_array_kg_per_day(m0_array_kg_per_day, p_non)
		dSMD_array_mm_per_day = _calculate_dSMD_array_mm_per_day(data, output, node)
		Psmd = _calculate_Psmd(her_array_mm_per_day, dSMD_array_mm_per_day)
		m3_array_kg_per_day = _calculate_m3_array_kg_per_day(pp, p_non, m0_array_kg_per_day, her_array_mm_per_day, Psmd)
		m4_array_kg_per_day = _calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day, Psmd)
		m4out_array_kg_per_day = _calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, m4_array_kg_per_day)
		mi_array_kg_per_day = _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day)
		_check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day, logging)
		proportion_reaching_water_table_array_per_day = _calculate_proportion_reaching_water_table_array_per_day(data, output, node, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, proportion_0, proportion_100)
		nitrate_reaching_water_table_array_kg_per_day = np.array(m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day))
		nitrate_reaching_water_table_array_tons_per_day = _convert_kg_to_tons_array(nitrate_reaching_water_table_array_kg_per_day)

		return {
			"her_array_mm_per_day" : her_array_mm_per_day,
			"m0_array_kg_per_day" : m0_array_kg_per_day,
			"m1_array_kg_per_day" : m1_array_kg_per_day,
			"m1a_array_kg_per_day" : m1a_array_kg_per_day,
			"m2_array_kg_per_day" : m2_array_kg_per_day,
			"m3_array_kg_per_day" : m3_array_kg_per_day,
			"m4_array_kg_per_day" : m4_array_kg_per_day,
			"m4out_array_kg_per_day" : m4out_array_kg_per_day,
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
			"m4_array_kg_per_day" : empty_array,
			"m4out_array_kg_per_day" : empty_array,
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

def _calculate_pp(data, output, node, her_array_mm_per_day):
	perc_through_root_mm_per_day = output["perc_through_root"]
	pp = _divide_arrays(perc_through_root_mm_per_day, her_array_mm_per_day)
	return pp

def _calculate_m1_array_kg_per_day(m0_kg_per_day, pp):
	m1_kg_per_day = pp * m0_kg_per_day
	return m1_kg_per_day

def _divide_arrays(a, b):
	return np.divide(a, b, out = np.zeros_like(a), where = b != 0)

def _calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day):
	return m._calculate_m1a_array_kg_per_day(output, m1_array_kg_per_day)

def _calculate_p_non(data, output, node, her_array_mm_per_day):
	runoff_recharge_mm_per_day = output["runoff_recharge"]
	macropore_att_mm_per_day = output["macropore_att"]
	macropore_dir_mm_per_day = output["macropore_dir"]
	macropore_mm_per_day = macropore_att_mm_per_day + macropore_dir_mm_per_day
	p_non = _divide_arrays((runoff_recharge_mm_per_day + macropore_mm_per_day), her_array_mm_per_day)
	return p_non

def _calculate_m2_array_kg_per_day(m0_array_kg_per_day, p_non):
	m2_kg_per_day = m0_array_kg_per_day * p_non
	return m2_kg_per_day

def _calculate_m3_array_kg_per_day(pp, p_non, m0_array_kg_per_day, her_array_mm_per_day, Psmd):
	Pro = np.where(
		her_array_mm_per_day <= 0,
		0,
		1 - pp - p_non - Psmd)
	return m0_array_kg_per_day * Pro

def _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day):
	return m1a_array_kg_per_day + m2_array_kg_per_day

def _calculate_dSMD_array_mm_per_day(data, output, node):
	smd = output["smd"]
	next_day_smd = np.roll(smd, -1)
	if (next_day_smd.size > 0):
		next_day_smd[-1] = 0.0
	return smd - next_day_smd

def _calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day, Psmd):
	Psmd = np.divide(
		np.maximum(0.0, dSMD_array_mm_per_day),
		her_array_mm_per_day,
		out = np.zeros_like(dSMD_array_mm_per_day),
		where=(her_array_mm_per_day != 0))
	M4_array_kg = Psmd * m0_array_kg_per_day
	return M4_array_kg

def _calculate_Psmd(her_array_mm_per_day, dSMD_array_mm_per_day):
	Psmd = np.divide(
		np.maximum(0.0, dSMD_array_mm_per_day),
		her_array_mm_per_day,
		out = np.zeros_like(dSMD_array_mm_per_day),
		where=(her_array_mm_per_day != 0))
	return Psmd

def _calculate_Psoilperc(data, output, node):
	perc_through_root_mm_per_day = output["perc_through_root"]
	TAW_array_mm = output["tawtew"]
	numerator_mm = np.maximum(0.0, perc_through_root_mm_per_day)
	denominator_mm = (perc_through_root_mm_per_day + TAW_array_mm)
	return _divide_arrays(numerator_mm, denominator_mm)

def _calculate_Pherperc(data, output, node, her_array_mm_per_day):
	perc_through_root_mm_per_day = output["perc_through_root"]
	return _divide_arrays(
		np.maximum(0, perc_through_root_mm_per_day),
		her_array_mm_per_day)

def _calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg):
	TAW_array_mm = output["tawtew"]
	SMD_array_mm = output["smd"]
	soil_store_array_mm = TAW_array_mm - SMD_array_mm
	prop_soil_store = - np.minimum(0.0, dSMD_array_mm_per_day) / soil_store_array_mm
	M4out_array_kg = np.zeros_like(dSMD_array_mm_per_day)
	M4tot_kg = 0
	for day in range(M4out_array_kg.size):
		M4tot_kg += M4_array_kg[day]
		M4out_kg = prop_soil_store[day] * M4tot_kg
		M4out_array_kg[day] = M4out_kg
		M4tot_kg -= M4out_kg
	return M4out_array_kg

def _check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day, logging):
	m0_kg = _calculate_m0_kg_for_balance(m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day)
	is_m0_as_expected = _is_mass_balanced(m0_kg, m0_array_kg_per_day)
	if not is_m0_as_expected:
		i = _find_unbalanced_day_to_report(m0_kg, m0_array_kg_per_day)
		message = _make_unbalanced_day_log_message(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day, m0_kg, i)
		logging.warning(message)

def _calculate_m0_kg_for_balance(m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day):
	return m1_array_kg_per_day + m2_array_kg_per_day + m3_array_kg_per_day + m4_array_kg_per_day - m4out_array_kg_per_day

def _is_mass_balanced(m0_kg, m0_array_kg_per_day):
	return np.allclose(m0_kg, m0_array_kg_per_day, atol=0.0001)

def _find_unbalanced_day_to_report(m0_kg, m0_array_kg_per_day):
	return np.argmax(np.abs(m0_kg - m0_array_kg_per_day))

def _make_unbalanced_day_log_message(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, m4out_array_kg_per_day, m0_kg, i):
	m0 = m0_array_kg_per_day[i]
	m1 = m1_array_kg_per_day[i]
	m2 = m2_array_kg_per_day[i]
	m3 = m3_array_kg_per_day[i]
	m4 = m4_array_kg_per_day[i]
	m4out = m4out_array_kg_per_day[i]
	diff = abs(m0_kg[i] - m0_array_kg_per_day[i])
	message = f"Nitrate masses do not balance for node {node} using the equation M0 = M1 + M2 + M3 + M4 - M4out. The day with the largest difference is at index {i} with a difference of {diff} kg. M0 = {m0} kg; M1 = {m1} kg; M2 = {m2} kg; M3 = {m3} kg; M4 = {m4} kg; M4out = {m4out} kg."
	return message

def _calculate_proportion_reaching_water_table_array_per_day(data, output, node, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, proportion_0, proportion_100):	
	time_switcher = data["time_switcher"]
	length = len(data["series"]["date"])
	depth_to_water_m = data["params"]["nitrate_depth_to_water"][node][0]
	if depth_to_water_m == 0.0:
		return proportion_0
	elif depth_to_water_m == 100.0:
		return proportion_100
	else:
		return __calculate_proportion_reaching_water_table_array_per_day(length, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, time_switcher)

def __calculate_proportion_reaching_water_table_array_per_day(length, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, time_switcher):
	result = np.zeros(length)
	for i in range(length):
		result[i] = _calculate_daily_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, depth_to_water_m, i)
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
	return len(time_periods)
	

def _calculate_combined_recharge_m_cubed(data, output, node):
	node_areas = data["params"]["node_areas"]
	combined_recharge_mm = output["combined_recharge"]
	combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
	combined_recharge_m_cubed = combined_recharge_m * node_areas[node]
	return combined_recharge_m_cubed

def _convert_mm_to_m(arr):
	return arr / 100.0

def write_nitrate_csv(data, nitrate_aggregation):
	write_nitrate_csv_bytes_cython(data, nitrate_aggregation)

def write_nitrate_csv_bytes_cython(data, nitrate_aggregation):
	filename = make_output_filename(data)
	m.write_nitrate_csv_bytes(filename, nitrate_aggregation)

def make_output_filename(data):
	run_name = data["params"]["run_name"]
	file = run_name + "_nitrate.csv"
	folder = utils.CONSTANTS["OUTPUT_DIR"]
	return os.path.join(folder, file)
 