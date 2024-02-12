import logging
import numpy as np
import os
import swacmod.feature_flags as ff
from swacmod.nitrate_blackboard import NitrateBlackboard
import swacmod.utils as utils
import swacmod.model as m
import swacmod.nitrate_proportion_reaching_water_table as nitrate_proportion

def get_nitrate(data, output, node):
	nitrate = calculate_nitrate(data, output, node)
	return {
		"mi_array_kg_per_day" : nitrate["mi_array_kg_per_day"],
		"nitrate_reaching_water_table_array_tons_per_day" : nitrate["nitrate_reaching_water_table_array_tons_per_day"],
		"nitrate_to_surface_water_array_tons_per_day" : nitrate["nitrate_to_surface_water_array_tons_per_day"],
	}

def calculate_nitrate(data, output, node, logging = logging):
	if "enabled" == data["params"]["nitrate_process"]:
		blackboard = NitrateBlackboard()
		blackboard.initialise_blackboard(data, output, node, logging)
		blackboard = _do_nitrate_calculations(blackboard)
		return _convert_blackboard_to_result(blackboard)
	else:
		length = output["rainfall_ts"].size
		return _make_empty_result(length)

def _do_nitrate_calculations(blackboard):
	blackboard.her_array_mm_per_day = _calculate_her_array_mm_per_day(blackboard)
	blackboard.m0_array_kg_per_day = _calculate_m0_array_kg_per_day(blackboard)
	blackboard.Psoilperc = _calculate_Psoilperc(blackboard)
	blackboard.Pherperc = _calculate_Pherperc(blackboard)
	blackboard.dSMD_array_mm_per_day = _calculate_dSMD_array_mm_per_day(blackboard)
	blackboard.Psmd = _calculate_Psmd(blackboard)
	blackboard.M_soil_in_kg = _calculate_M_soil_in_kg(blackboard)
	blackboard.M_soil_tot_kg = _calculate_M_soil_tot_kg(blackboard)
	blackboard.m1_array_kg_per_day = _calculate_m1_array_kg_per_day(blackboard)
	m1a_b_array_kg_per_day = _calculate_m1a_b_array_kg_per_day(blackboard)
	blackboard.m1a_array_kg_per_day = m1a_b_array_kg_per_day[0,:]
	blackboard.m1b_array_kg_per_day = m1a_b_array_kg_per_day[1,:]		
	blackboard.p_non_her = _calculate_p_non_her(blackboard)
	blackboard.p_non = _calculate_p_non(blackboard)
	blackboard.m2_array_kg_per_day = _calculate_m2_array_kg_per_day(blackboard)
	blackboard.Pro = _calculate_Pro(blackboard)
	blackboard.m3_array_kg_per_day = _calculate_m3_array_kg_per_day(blackboard)
	blackboard.m4_array_kg_per_day = _calculate_M4_array_mm_per_day(blackboard)
	blackboard.mi_array_kg_per_day = _calculate_mi_array_kg_per_day(blackboard)
	blackboard.total_NO3_to_receptors_kg = _calculate_total_NO3_to_receptors_kg(blackboard)
	blackboard.mass_balance_error_kg = _calculate_mass_balance_error_kg(blackboard)
	_check_masses_balance(blackboard)
	blackboard.proportion_reaching_water_table_array_per_day = _calculate_proportion_reaching_water_table_array_per_day(blackboard)
	blackboard.nitrate_reaching_water_table_array_from_this_run_kg_per_day = np.array(m.calculate_mass_reaching_water_table_array_kg_per_day(blackboard))
	blackboard.nitrate_reaching_water_table_array_from_this_run_tons_per_day = _convert_kg_to_tons_array(blackboard)
	blackboard.nitrate_reaching_water_table_array_tons_per_day = _combine_nitrate_reaching_water_table_array_from_this_run_and_historical_run_tons_per_day(blackboard)
	blackboard.nitrate_to_surface_water_kg_per_day = _calculate_nitrate_to_surface_water_array_kg_per_day(blackboard)
	blackboard.nitrate_to_surface_water_array_tons_per_day = _convert_nitrate_to_surface_water_kg_to_tons_array(blackboard)
	return blackboard

def _calculate_her_array_mm_per_day(blackboard):
	return np.maximum(0.0, blackboard.rainfall_ts - blackboard.ae)

def _calculate_m0_array_kg_per_day(blackboard):
	max_load_per_year_kg_per_hectare = blackboard.nitrate_loading[3]
	her_at_5_percent = blackboard.nitrate_loading[4]
	her_at_50_percent = blackboard.nitrate_loading[5]
	her_at_95_percent = blackboard.nitrate_loading[6]

	hectare_area_m_sq = 10000
	max_load_per_year_kg_per_cell = max_load_per_year_kg_per_hectare * blackboard.cell_area_m_sq / hectare_area_m_sq

	m0_array_kg_per_day = m._calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		blackboard.days,
		blackboard.her_array_mm_per_day,
		blackboard.time_switcher)
	return m0_array_kg_per_day

def _calculate_m1_array_kg_per_day(blackboard):
	M_soil_tot_initial_kg = np.roll(blackboard.M_soil_tot_kg, 1)
	if (M_soil_tot_initial_kg.size > 0):
		M_soil_tot_initial_kg[0] = 0.0
	return blackboard.Psoilperc * (blackboard.M_soil_in_kg + M_soil_tot_initial_kg)

def _calculate_M_soil_tot_kg(blackboard):
	result_kg = np.zeros_like(blackboard.M_soil_in_kg)
	P = 1 - blackboard.Psoilperc
	M_soil_tot_initial_kg = 0.0
	for i in range(blackboard.M_soil_in_kg.size):
		M_soil_tot_kg = (M_soil_tot_initial_kg + blackboard.M_soil_in_kg[i]) * P[i]
		result_kg[i] = M_soil_tot_kg
		M_soil_tot_initial_kg = M_soil_tot_kg
	return result_kg

def _divide_arrays(a, b):
	return np.divide(a, b, out = np.zeros_like(a), where = b != 0)

def _calculate_m1a_b_array_kg_per_day(blackboard):
	return m._calculate_m1a_b_array_kg_per_day(blackboard)

def _calculate_p_non_her(blackboard):
	non_her_mm_day = blackboard.runoff_mm_per_day + blackboard.Pherperc * blackboard.her_array_mm_per_day + blackboard.Psmd * blackboard.her_array_mm_per_day + blackboard.macropore_att_mm_per_day + blackboard.macropore_dir_mm_per_day-blackboard.her_array_mm_per_day
	p_non_her = np.where(
		blackboard.her_array_mm_per_day <= 0,
		blackboard.runoff_mm_per_day + blackboard.macropore_att_mm_per_day + blackboard.macropore_dir_mm_per_day,
		_divide_arrays(non_her_mm_day, (blackboard.runoff_mm_per_day + blackboard.macropore_att_mm_per_day + blackboard.macropore_dir_mm_per_day)))
	return p_non_her

def _calculate_p_non(blackboard):
	macropore_mm_per_day = blackboard.macropore_att_mm_per_day + blackboard.macropore_dir_mm_per_day
	p_her = 1 - blackboard.p_non_her
	runoff_recharge_her_mm_per_day = blackboard.runoff_recharge_mm_per_day * p_her
	macropore_her_mm_per_day = macropore_mm_per_day * p_her
	p_non = np.where(
		blackboard.her_array_mm_per_day <= 0,
		0,
		_divide_arrays((runoff_recharge_her_mm_per_day + macropore_her_mm_per_day), blackboard.her_array_mm_per_day))
	return p_non

def _calculate_m2_array_kg_per_day(blackboard):
	m2_kg_per_day = blackboard.m0_array_kg_per_day * blackboard.p_non
	return m2_kg_per_day

def _calculate_Pro(blackboard):
	Pro_init = np.where(
		blackboard.her_array_mm_per_day <= 0,
		0,
		1 - blackboard.p_non - blackboard.Pherperc - blackboard.Psmd)

	if np.max(blackboard.p_non + blackboard.Pherperc + blackboard.Psmd) > 2:
		i = np.argmax(blackboard.p_non + blackboard.Pherperc + blackboard.Psmd)
		blackboard.logging.warning(f"node = {blackboard.node}; day = {i}; p_non = {blackboard.p_non[i]}; Pherperc = {blackboard.Pherperc[i]}; Psmd = {blackboard.Psmd[i]}")
		blackboard.logging.warning(f"p_non inputs: macropore_att_mm_per_day = {blackboard.macropore_att_mm_per_day[i]}; macropore_dir_mm_per_day = {blackboard.macropore_dir_mm_per_day[i]}; her_array_mm_per_day = {blackboard.her_array_mm_per_day[i]}; runoff_recharge_mm_per_day = {blackboard.runoff_recharge_mm_per_day[i]}")
		blackboard.logging.warning(f"Pherperc inputs: perc_through_root_mm_per_day = {blackboard.perc_through_root_mm_per_day[i]}")
		blackboard.logging.warning(f"Psmd inputs: dSMD_array_mm_per_day = {blackboard.dSMD_array_mm_per_day[i]}")

	Pro = np.where(
		Pro_init <= 0,
		0,
		Pro_init)
	return Pro

def _calculate_m3_array_kg_per_day(blackboard):
	return blackboard.m0_array_kg_per_day * blackboard.Pro

def _calculate_mi_array_kg_per_day(blackboard):
	return blackboard.m1a_array_kg_per_day + blackboard.m2_array_kg_per_day

def _calculate_dSMD_array_mm_per_day(blackboard):
	return blackboard.smd - np.maximum(0, blackboard.p_smd)

def _calculate_M4_array_mm_per_day(blackboard):
	return blackboard.M_soil_in_kg - blackboard.m1_array_kg_per_day

def _calculate_Psmd(blackboard):
	Psmd_init = _divide_arrays(
		np.maximum(0.0, blackboard.dSMD_array_mm_per_day),
		blackboard.her_array_mm_per_day)
	Psmd = np.minimum(1.0, Psmd_init)
	return Psmd

def _calculate_Psoilperc(blackboard):
	numerator_mm = np.maximum(0.0, blackboard.perc_through_root_mm_per_day)
	denominator_mm = (blackboard.perc_through_root_mm_per_day + blackboard.TAW_array_mm)
	return _divide_arrays(numerator_mm, denominator_mm)

def _calculate_Pherperc(blackboard):
	return _divide_arrays(
		np.maximum(0, blackboard.perc_through_root_mm_per_day),
		blackboard.her_array_mm_per_day)

def _calculate_M_soil_in_kg(blackboard):
	return blackboard.m0_array_kg_per_day * (blackboard.Psmd + blackboard.Pherperc)

def _calculate_total_NO3_to_receptors_kg(blackboard):
	return blackboard.m1_array_kg_per_day + blackboard.m2_array_kg_per_day + blackboard.m3_array_kg_per_day + blackboard.m4_array_kg_per_day

def _calculate_mass_balance_error_kg(blackboard):
	return blackboard.m0_array_kg_per_day - blackboard.total_NO3_to_receptors_kg

def _check_masses_balance(blackboard):
	is_mass_balanced = _is_mass_balanced(blackboard.mass_balance_error_kg)
	if not is_mass_balanced:
		i = _find_unbalanced_day_to_report(blackboard.mass_balance_error_kg)
		message = _make_unbalanced_day_log_message(i, blackboard)
		blackboard.logging.warning(message)

def _is_mass_balanced(mass_balance_error_kg):
	return np.allclose(mass_balance_error_kg, 0.0)

def _find_unbalanced_day_to_report(mass_balance_error_kg):
	return np.argmax(np.abs(mass_balance_error_kg))

def _make_unbalanced_day_log_message(i, blackboard):
	m0 = blackboard.m0_array_kg_per_day[i]
	m1 = blackboard.m1_array_kg_per_day[i]
	m2 = blackboard.m2_array_kg_per_day[i]
	m3 = blackboard.m3_array_kg_per_day[i]
	m4 = blackboard.m4_array_kg_per_day[i]
	mass_balance_error = blackboard.mass_balance_error_kg[i]
	total_NO3_to_receptors = blackboard.total_NO3_to_receptors_kg[i]
	message = f"Nitrate masses do not balance for node {blackboard.node} using the equation M0 = M1 + M2 + M3 + M4. The day with the largest mass balance error is at index {i} with a mass balance error of {mass_balance_error} kg. total_NO3_to_receptors = {total_NO3_to_receptors} kg; M0 = {m0} kg; M1 = {m1} kg; M2 = {m2} kg; M3 = {m3} kg; M4 = {m4} kg."
	return message

def _calculate_proportion_reaching_water_table_array_per_day(blackboard):
	return nitrate_proportion.calculate_proportion_reaching_water_table_array_per_day(blackboard)

def _calculate_nitrate_to_surface_water_array_kg_per_day(blackboard):
	return blackboard.m1b_array_kg_per_day + blackboard.m3_array_kg_per_day

def _convert_kg_to_tons_array(blackboard):
	return blackboard.nitrate_reaching_water_table_array_from_this_run_kg_per_day / 1000.0

def _convert_nitrate_to_surface_water_kg_to_tons_array(blackboard):
	return blackboard.nitrate_to_surface_water_kg_per_day / 1000.0

def _combine_nitrate_reaching_water_table_array_from_this_run_and_historical_run_tons_per_day(blackboard):
	return (blackboard.nitrate_reaching_water_table_array_from_this_run_tons_per_day 
		 + blackboard.historical_nitrate_reaching_water_table_array_tons_per_day)

def make_aggregation_array(data):
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	shape = (_len_time_periods(time_periods), len(node_areas))
	aggregation = np.zeros(shape = shape)
	return aggregation
	
def make_stream_aggregation_array(data):
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	shape = (_len_time_periods(time_periods), len(node_areas))
	stream_aggregation = np.zeros(shape = shape)
	return stream_aggregation

def make_mi_aggregation_array(data):
	time_periods = data["params"]["time_periods"]
	len_time_periods = _len_time_periods(time_periods)
	node_count = data['params']['num_nodes']
	shape = (node_count, len_time_periods)
	return np.zeros(shape = shape)

def aggregate_nitrate(aggregation, data, output, node):
	time_periods = data["params"]["time_periods"]
	nitrate_reaching_water_table_array_tons_per_day = output["nitrate_reaching_water_table_array_tons_per_day"]
	combined_recharge_m_cubed = _calculate_combined_recharge_m_cubed(data, output, node)

	node_areas = data["params"]["node_areas"]
	len_time_periods = _len_time_periods(time_periods)
	m._aggregate_nitrate(time_periods, len_time_periods, nitrate_reaching_water_table_array_tons_per_day, combined_recharge_m_cubed, aggregation, node, node_areas)
	return aggregation
	
def aggregate_surface_water_nitrate(stream_nitrate_aggregation, data, output, node):
	time_periods = data["params"]["time_periods"]
	nitrate_to_surface_water_array_tons_per_day = output["nitrate_to_surface_water_array_tons_per_day"]
	combined_surface_water_m_cubed = _calculate_combined_surface_water_m_cubed(data, output, node)

	len_time_periods = _len_time_periods(time_periods)
	#fills stream_aggregation array with total nitrate mass for all days in a stress period for this node
	m._aggregate_surface_water_nitrate(time_periods, len_time_periods, nitrate_to_surface_water_array_tons_per_day, combined_surface_water_m_cubed, stream_nitrate_aggregation, node)
	return stream_nitrate_aggregation

def aggregate_mi(aggregation, data, output, node):
	time_periods = data["params"]["time_periods"]
	mi_array_kg_per_day = output["mi_array_kg_per_day"]
	len_time_periods = _len_time_periods(time_periods)
	return m.aggregate_mi(aggregation, time_periods, len_time_periods, mi_array_kg_per_day, node)

def _len_time_periods(time_periods):
	return len(time_periods)
	
def _calculate_combined_recharge_m_cubed(data, output, node):
	node_areas = data["params"]["node_areas"]
	combined_recharge_mm = output["combined_recharge"]
	combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
	combined_recharge_m_cubed = combined_recharge_m * node_areas[node]
	return combined_recharge_m_cubed
	
def _calculate_combined_surface_water_m_cubed(data, output, node):
	node_areas = data["params"]["node_areas"]
	combined_surface_water_mm = output["combined_str"]
	combined_surface_water_m = _convert_mm_to_m(combined_surface_water_mm)
	combined_surface_water_m_cubed = combined_surface_water_m * node_areas[node]
	return combined_surface_water_m_cubed

def _convert_mm_to_m(arr):
	return arr / 100.0

def write_nitrate_csv(data, nitrate_aggregation):
	write_nitrate_csv_bytes_cython(data, nitrate_aggregation)

def write_nitrate_csv_bytes_cython(data, nitrate_aggregation):
	filename = make_output_filename(data)
	m.write_nitrate_csv_bytes(filename, nitrate_aggregation)
	
def write_stream_nitrate_csv(data, stream_conc):
	write_stream_nitrate_csv_bytes_cython(data, stream_conc)

def write_stream_nitrate_csv_bytes_cython(data, stream_conc):
	filename = make_nitrate_surface_flow_filename(data)
	m.write_stream_nitrate_csv_bytes(filename, stream_conc)

def write_mi_csv(data, nitrate_mi_aggregation):
	node_count = nitrate_mi_aggregation.shape[0]
	time_period_count = nitrate_mi_aggregation.shape[1]

	filename = make_mi_output_filename(data)
	with open(filename, "wb") as f:
		for node in range(node_count):
			cell = b"%g" % (nitrate_mi_aggregation[node, 0])
			f.write(cell)
			for time_period in range(1, time_period_count):
				cell = b",%g" % (nitrate_mi_aggregation[node, time_period])
				f.write(cell)
			f.write(b"\r\n")
	return filename

def make_output_filename(data):
	return _make_run_filename_with_suffix(data, "nitrate")

def make_mi_output_filename(data):
	return _make_run_filename_with_suffix(data, "mi")

def make_nitrate_surface_flow_filename(data):
	return _make_run_filename_with_suffix(data, "stream_nitrate")

def _make_run_filename_with_suffix(data, suffix):
	run_name = data["params"]["run_name"]
	file = run_name + "_" + suffix + ".csv"
	folder = utils.CONSTANTS["OUTPUT_DIR"]
	return os.path.join(folder, file)
	
def _convert_blackboard_to_result(blackboard):
	return {
		"her_array_mm_per_day" : blackboard.her_array_mm_per_day,
		"m0_array_kg_per_day" : blackboard.m0_array_kg_per_day,
		"m1_array_kg_per_day" : blackboard.m1_array_kg_per_day,
		"m1a_array_kg_per_day" : blackboard.m1a_array_kg_per_day,
		"m1b_array_kg_per_day" : blackboard.m1b_array_kg_per_day,
		"m2_array_kg_per_day" : blackboard.m2_array_kg_per_day,
		"m3_array_kg_per_day" : blackboard.m3_array_kg_per_day,
		"m4_array_kg_per_day" : blackboard.m4_array_kg_per_day,
		"mi_array_kg_per_day" : blackboard.mi_array_kg_per_day,
		"proportion_reaching_water_table_array_per_day" : blackboard.proportion_reaching_water_table_array_per_day,
		"nitrate_reaching_water_table_array_from_this_run_kg_per_day" : blackboard.nitrate_reaching_water_table_array_from_this_run_kg_per_day,
		"nitrate_reaching_water_table_array_tons_per_day" : blackboard.nitrate_reaching_water_table_array_tons_per_day,
		"nitrate_to_surface_water_kg_per_day" : blackboard.nitrate_to_surface_water_kg_per_day,
		"nitrate_to_surface_water_array_tons_per_day" : blackboard.nitrate_to_surface_water_array_tons_per_day,
	}

def _make_empty_result(length):
	empty_array = np.zeros(length)
	return {
		"mi_array_kg_per_day" : empty_array,
		"nitrate_reaching_water_table_array_tons_per_day" : empty_array,
		"nitrate_to_surface_water_array_tons_per_day" : empty_array,
	}
