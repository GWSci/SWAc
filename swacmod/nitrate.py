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
		Psoilperc = _calculate_Psoilperc(data, output, node)
		Pherperc = _calculate_Pherperc(data, output, node, her_array_mm_per_day)
		dSMD_array_mm_per_day = _calculate_dSMD_array_mm_per_day(data, output, node)
		Psmd = _calculate_Psmd(her_array_mm_per_day, dSMD_array_mm_per_day)
		M_soil_in_kg = _calculate_M_soil_in_kg(m0_array_kg_per_day, Psmd, Pherperc)
		M_soil_tot_kg = _calculate_M_soil_tot_kg(M_soil_in_kg, Psoilperc)
		m1_array_kg_per_day = _calculate_m1_array_kg_per_day(Psoilperc, m0_array_kg_per_day, M_soil_tot_kg)
		m1a_array_kg_per_day = _calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day)
		p_non = _calculate_p_non(data, output, node, her_array_mm_per_day)
		m2_array_kg_per_day = _calculate_m2_array_kg_per_day(m0_array_kg_per_day, p_non)
		Pro = _calculate_Pro(her_array_mm_per_day, p_non, Pherperc, Psmd)
		m3_array_kg_per_day = _calculate_m3_array_kg_per_day(m0_array_kg_per_day, Pro)
		m4_array_kg_per_day = _calculate_M4_array_mm_per_day(M_soil_tot_kg, m1_array_kg_per_day)
		mi_array_kg_per_day = _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day)
		total_NO3_to_receptors_kg = _calculate_total_NO3_to_receptors_kg(m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day)
		mass_balance_error_kg = _calculate_mass_balance_error_kg(m0_array_kg_per_day, total_NO3_to_receptors_kg)
		_check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, total_NO3_to_receptors_kg, mass_balance_error_kg, logging)
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

def _calculate_m1_array_kg_per_day(Psoilperc, m0_array_kg_per_day, M_soil_tot_kg):
	M_soil_tot_initial_kg = np.roll(M_soil_tot_kg, 1)
	if (M_soil_tot_initial_kg.size > 0):
		M_soil_tot_initial_kg[0] = 0.0
	return Psoilperc * (m0_array_kg_per_day + M_soil_tot_initial_kg)

def _calculate_M_soil_tot_kg(M_soil_in_kg, Psoilperc):
	result_kg = np.zeros_like(M_soil_in_kg)
	P = 1 - Psoilperc
	M_soil_tot_initial_kg = 0.0
	for i in range(M_soil_in_kg.size):
		M_soil_tot_kg = (M_soil_tot_initial_kg + M_soil_in_kg[i]) * P[i]
		result_kg[i] = M_soil_tot_kg
		M_soil_tot_initial_kg = M_soil_tot_kg
	return result_kg

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

def _calculate_Pro(her_array_mm_per_day, p_non, pherperc, Psmd):
	Pro = np.where(
		her_array_mm_per_day <= 0,
		0,
		1 - p_non - pherperc - Psmd)
	return Pro

def _calculate_m3_array_kg_per_day(m0_array_kg_per_day, Pro):
	return m0_array_kg_per_day * Pro

def _calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day):
	return m1a_array_kg_per_day + m2_array_kg_per_day

def _calculate_dSMD_array_mm_per_day(data, output, node):
	smd = output["smd"]
	p_smd = output["p_smd"]
	return smd - np.maximum(0, p_smd)

def _calculate_M4_array_mm_per_day(M_soil_tot_kg, m1_array_kg_per_day):
	return M_soil_tot_kg - m1_array_kg_per_day

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

def _calculate_M_soil_in_kg(m0_array_kg_per_day, Psmd, Pherperc):
	return m0_array_kg_per_day * (Psmd + Pherperc)

def _calculate_total_NO3_to_receptors_kg(m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day):
	return m1_array_kg_per_day + m2_array_kg_per_day + m3_array_kg_per_day + m4_array_kg_per_day

def _calculate_mass_balance_error_kg(m0_array_kg_per_day, total_NO3_to_receptors_kg):
	return m0_array_kg_per_day - total_NO3_to_receptors_kg

def _check_masses_balance(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, total_NO3_to_receptors_kg, mass_balance_error_kg, logging):
	is_mass_balanced = _is_mass_balanced(mass_balance_error_kg)
	if not is_mass_balanced:
		i = _find_unbalanced_day_to_report(mass_balance_error_kg)
		message = _make_unbalanced_day_log_message(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, i, mass_balance_error_kg, total_NO3_to_receptors_kg)
		logging.warning(message)

def _is_mass_balanced(mass_balance_error_kg):
	return np.allclose(mass_balance_error_kg, 0.0)

def _find_unbalanced_day_to_report(mass_balance_error_kg):
	return np.argmax(np.abs(mass_balance_error_kg))

def _make_unbalanced_day_log_message(node, m0_array_kg_per_day, m1_array_kg_per_day, m2_array_kg_per_day, m3_array_kg_per_day, m4_array_kg_per_day, i, mass_balance_error_kg, total_NO3_to_receptors_kg):
	m0 = m0_array_kg_per_day[i]
	m1 = m1_array_kg_per_day[i]
	m2 = m2_array_kg_per_day[i]
	m3 = m3_array_kg_per_day[i]
	m4 = m4_array_kg_per_day[i]
	mass_balance_error = mass_balance_error_kg[i]
	total_NO3_to_receptors = total_NO3_to_receptors_kg[i]
	message = f"Nitrate masses do not balance for node {node} using the equation M0 = M1 + M2 + M3 + M4. The day with the largest mass balance error is at index {i} with a mass balance error of {mass_balance_error} kg. total_NO3_to_receptors = {total_NO3_to_receptors} kg; M0 = {m0} kg; M1 = {m1} kg; M2 = {m2} kg; M3 = {m3} kg; M4 = {m4} kg."
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
 