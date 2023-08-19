import numpy
import numpy as np
import swacmod.timer as t
import sys
from . import utils as u
from swacmod.utils import monthdelta2, weekdelta


def numpy_get_precipitation(data, nodes):
	token = t.start_timing("series and params")
	series, params = data['series'], data['params']
	t.stop_timing(token)

	token = t.start_timing("rainfall_zone_mapping = params['rainfall_zone_mapping']")
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	t.stop_timing(token)

	token = t.start_timing("zone_rf")
	zone_rf = numpy.array([rainfall_zone_mapping[node][0] - 1 for node in nodes])
	t.stop_timing(token)

	token = t.start_timing("coef_rf")
	coef_rf = numpy.array([rainfall_zone_mapping[node][1] for node in nodes])
	t.stop_timing(token)

	token = t.start_timing("rainfall_ts")
	rainfall_ts = numpy.transpose(series['rainfall_ts'][:, zone_rf]) * coef_rf[:, None]
	t.stop_timing(token)
	return rainfall_ts

def lazy_get_precipitation(data, nodes):
	token = t.start_timing("lazy_get_precipitation")
	series, params = data['series'], data['params']
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	zone_rf = numpy.array([rainfall_zone_mapping[node][0] - 1 for node in nodes])
	coef_rf = numpy.array([rainfall_zone_mapping[node][1] for node in nodes])
	result = Lazy_Precipitation(zone_rf, coef_rf, series['rainfall_ts'])
	t.stop_timing(token)
	return result

def get_precipitation(data, output, node):
	"""C) Precipitation [mm/d]."""
	rainfall_ts = data["precalculated_precipitation"][node - 1]
	return {'rainfall_ts': rainfall_ts}

def get_pefac(data, output, node):
	"""E) Vegetation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
	series, params = data['series'], data['params']
	days = len(series['date'])

	fao = params['fao_process']
	canopy = params['canopy_process']
	calculate_pefac = fao == 'enabled' or canopy == 'enabled'
	if not calculate_pefac:
		return {'pefac': np.zeros(days)}

	pe = output['pe_ts']
	kc = params['kc_list'][series['months']]
	zone_lu = np.array(params['lu_spatial'][node], dtype=np.float64)
	len_lu = len(params['lu_spatial'][node])

	pefac = pe * np.sum(kc[:, 0:len_lu] * zone_lu[0:len_lu], axis=1)

	return {'pefac': np.array(pefac)}

def get_ae(data, output, node):
	"""Multicolumn function.

	N) Rapid Runoff Class [%]
	O) Rapid Runoff [mm/d]
	P) Runoff Recharge [mm/d]
	Q) MacroPore: (attenuated) Bypass of Root Zone and Interflow,
	   subject to recharge attenuation [mm/d]
	R) MacroPore: (direct) Bypass of Root Zone, Interflow and
	   Recharge Attenuation [mm/d]
	S) Percolation into Root Zone [mm/d]
	V) Potential Soil Moisture Defecit (pSMD) [mm]
	W) Soil Moisture Defecit (SMD) [mm]
	X) Ks (slope factor) [-]
	Y) AE (actual evapotranspiration) [mm/d]
	"""
	series, params = data['series'], data['params']
	rrp = params['rapid_runoff_params']
	s_smd = params['smd']
	mac_opt = params['macropore_activation_option']

	col_rapid_runoff_c = np.zeros(len(series['date']))
	col_rapid_runoff = np.zeros(len(series['date']))
	col_runoff_recharge = np.zeros(len(series['date']))
	col_macropore_att = np.zeros(len(series['date']))
	col_macropore_dir = np.zeros(len(series['date']))
	col_percol_in_root = np.zeros(len(series['date']))
	col_p_smd = np.zeros(len(series['date']))
	col_smd = np.zeros(len(series['date']))
	col_k_slope = np.zeros(len(series['date']))
	col_ae = np.zeros(len(series['date']))
	zone_mac = params['macropore_zone_mapping'][node] - 1
	zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
	ssmd = u.weighted_sum(params['soil_spatial'][node],
									s_smd['starting_SMD'])
	class_smd = np.array(rrp[zone_rro]['class_smd'],
										dtype=np.int64)
	class_ri = np.array(rrp[zone_rro]['class_ri'],
										dtype=np.int64)
	macro_prop = params['macro_prop']
	macro_limit = params['macro_limit']
	macro_act = params['macro_act']
	macro_rec = params['macro_rec']
	values = np.array(rrp[zone_rro]['values'])
	len_class_smd = len(class_smd)
	len_class_ri = len(class_ri)
	last_smd = class_smd[-1] - 1
	last_ri = class_ri[-1] - 1
	value = values[-1][0]
	p_smd = ssmd
	length = len(series['date'])
	net_rainfall = output['net_rainfall']
	net_pefac_a = output['net_pefac']
	tawtew_a = output['tawtew']
	rawrew_a = output['rawrew']
	months = np.array(series['months'], dtype=np.int64)

	if params['swrecharge_process'] == 'enabled':
		col_runoff_recharge[:] = 0.0

	use_rapid_runoff_process = params['rapid_runoff_process'] == 'enabled'
	use_macropore_process = params['macropore_process'] == 'enabled'
	use_fao_process = params['fao_process'] == 'enabled'
	macro_act_factor_A = 0 if mac_opt == 'SMD' else 1
	macro_act_factor_B = 1 - macro_act_factor_A
	tawtew_a_minus_rawrew_a = tawtew_a - rawrew_a
	var_12_denominator_is_zero = tawtew_a_minus_rawrew_a == 0.0
	is_net_rainfall_greater_than_last_ri = net_rainfall > last_ri

	var10a_arr = macro_rec[months, zone_mac]	
	macro_act_for_month_and_zone = macro_act[months, zone_mac]
	net_rainfall_minus_macro_act_with_factor = net_rainfall - (macro_act_factor_A * macro_act_for_month_and_zone)
	ma_arr = (macro_act_factor_A * sys.float_info.max) + (macro_act_factor_B * macro_act_for_month_and_zone)
	macro_prop_arr = macro_prop[months, zone_mac]
	var10_arr = macro_limit[months, zone_mac]

	range_len_class_smd = range(len_class_smd)
	range_len_class_ri = range(len_class_ri)
		
	previous_smd = ssmd

	for num in range(length):
		net_rainfall_num = net_rainfall[num]

		if use_rapid_runoff_process:
			if is_net_rainfall_greater_than_last_ri[num] or previous_smd > last_smd:
				col_rapid_runoff_c[num] = value
			else:
				var3 = 0
				for i in range_len_class_ri:
					if class_ri[i] >= net_rainfall_num:
						var3 = i
						break
				for i in range_len_class_smd:
					if class_smd[i] >= previous_smd:
						col_rapid_runoff_c[num] = values[var3, i]
						break

			col_rapid_runoff_num = (0.0 if net_rainfall_num < 0.0 else (net_rainfall_num * col_rapid_runoff_c[num]))
			col_rapid_runoff[num] = col_rapid_runoff_num

		if use_macropore_process:
			var8a = net_rainfall_minus_macro_act_with_factor[num] - col_rapid_runoff_num
			if var8a > 0.0 and p_smd < ma_arr[num]:
				var9 = macro_prop_arr[num] * var8a
				macropore = min(var10_arr[num], var9)
				col_macropore_att[num] = macropore * (1 - var10a_arr[num])
				col_macropore_dir[num] = macropore * var10a_arr[num]

		col_percol_in_root_num = (net_rainfall_num - col_rapid_runoff_num
						  - col_macropore_att[num]
						  - col_macropore_dir[num])
		
		col_percol_in_root[num] = col_percol_in_root_num

		if use_fao_process:
			net_pefac_a_num = net_pefac_a[num]
			col_smd_num = max(p_smd, 0.0)
			col_smd[num] = col_smd_num
			tawtew_a_num = tawtew_a[num]

			if col_percol_in_root_num > net_pefac_a_num:
				col_k_slope[num] = -1.0
			elif var_12_denominator_is_zero[num]:
				col_k_slope[num] = 1.0
			else:
				var12 = (tawtew_a_num - col_smd_num) / tawtew_a_minus_rawrew_a[num]
				col_k_slope[num] = min(max(var12, 0.0), 1.0)

			if col_smd_num < rawrew_a[num] or col_percol_in_root_num > net_pefac_a_num:
				col_ae_num = net_pefac_a_num
			elif col_smd_num <= tawtew_a_num:
				col_ae_num = col_k_slope[num] * (net_pefac_a_num - col_percol_in_root_num)
			else:
				col_ae_num = 0.0

			p_smd = col_smd_num + col_ae_num - col_percol_in_root_num
			col_p_smd[num] = p_smd
			col_ae[num] = col_ae_num
			previous_smd = col_smd_num

	col = {}
	col['rapid_runoff_c'] = col_rapid_runoff_c
	col['rapid_runoff'] = col_rapid_runoff
	col['runoff_recharge'] = col_runoff_recharge
	col['macropore_att'] = col_macropore_att
	col['macropore_dir'] = col_macropore_dir
	col['percol_in_root'] = col_percol_in_root
	col['p_smd'] = col_p_smd
	col['smd'] = col_smd
	col['k_slope'] = col_k_slope
	col['ae'] = col_ae
	return col

def get_interflow(data, output, node):
	"""Multicolumn function.

	AF) Interflow Store Volume [mm]
	AG) Infiltration Recharge [mm/d]
	AH) Interflow to Surface Water Courses [mm/d]
	"""
	series, params = data['series'], data['params']

	length = len(series['date'])
	day_indexes = np.arange(length)
	col_interflow_volume = np.zeros(length)
	col_infiltration_recharge = np.zeros(length)
	col_interflow_to_rivers = np.zeros(length)
	interflow_store_input = output['interflow_store_input']
	interflow_zone = params['interflow_zone_mapping'][node]
	volume = params['init_interflow_store'][interflow_zone]

	if params['interflow_process'] == 'enabled':
		if params["infiltration_limit_use_timeseries"]:
			var5 = series['infiltration_limit_ts'][day_indexes, interflow_zone-1]
		else:
			var5 = np.full([length], params['infiltration_limit'][interflow_zone])

		if params["interflow_decay_use_timeseries"]:
			var8 = series['interflow_decay_ts'][day_indexes, interflow_zone-1]
		else:
			var8 = np.full([length], params['interflow_decay'][interflow_zone])
		
		one_minus_var8 = 1 - var8

		for num in range(length):
			var1 = volume - min(var5[num], volume)
			volume = interflow_store_input[num-1] + var1 * one_minus_var8[num]
			col_interflow_volume[num] = volume
		
		col_infiltration_recharge = np.minimum(var5, col_interflow_volume)
		var6_arr = col_interflow_volume - col_infiltration_recharge
		col_interflow_to_rivers = var6_arr * var8

	col = {}
	col['interflow_volume'] = col_interflow_volume
	col['infiltration_recharge'] = col_infiltration_recharge
	col['interflow_to_rivers'] = col_interflow_to_rivers

	return col

class Lazy_Precipitation:
	def __init__(self, zone_rf, coef_rf, rainfall_ts):
		self.zone_rf = zone_rf
		self.coef_rf = coef_rf
		self.rainfall_ts = rainfall_ts
		self.last_zone_rf = None
		self.last_coef_rf = None
		self.last_rainfall_ts = None
	
	def __getitem__(self, subscript):
		zone_rf = self.zone_rf[subscript]
		coef_rf = self.coef_rf[subscript]
				
		is_same_zone_rf = self.last_zone_rf == zone_rf
		is_same_coef_rf = self.last_coef_rf == coef_rf
		is_same_both = is_same_zone_rf and is_same_coef_rf
				
		if is_same_both:
			return self.last_rainfall_ts
		self.last_zone_rf = zone_rf
		self.last_coef_rf = coef_rf
		self.last_rainfall_ts = self.rainfall_ts[:, zone_rf] * coef_rf
		return self.last_rainfall_ts

def get_recharge(data, output, node):
	"""Multicolumn function.

	AJ) Recharge Store Volume [mm]
	AK) RCH: Combined Recharge [mm/d]
	"""
	series, params = data['series'], data['params']

	length = len(series['date'])
	col_recharge_store = np.zeros(length)
	col_combined_recharge = np.zeros(length)
	irs = params['recharge_attenuation_params'][node][0]
	rlp = params['recharge_attenuation_params'][node][1]
	rll = params['recharge_attenuation_params'][node][2]
	recharge_store_input = output['recharge_store_input']
	macropore_dir = output['macropore_dir']

	previous_recharge_store_and_macropore_dir = np.roll(recharge_store_input + macropore_dir, 1)

	if params['recharge_attenuation_process'] == 'enabled':
		recharge = irs
		combined_recharge = (min((irs * rlp), rll) + macropore_dir[0])
		col_recharge_store[0] = recharge
		col_combined_recharge[0] = combined_recharge
		for num in range(1, length):
			recharge = previous_recharge_store_and_macropore_dir[num] + recharge - combined_recharge
			combined_recharge = min((recharge * rlp), rll) + macropore_dir[num]

			col_recharge_store[num] = recharge
			col_combined_recharge[num] = combined_recharge

	else:
		col_recharge_store[0] = irs
		for num in range(1, length):
			col_combined_recharge[num] = (recharge_store_input[num] + macropore_dir[num])

	col = {}
	col['recharge_store'] = col_recharge_store
	col['combined_recharge'] = col_combined_recharge
	return col

def get_swabs(data, output, node):
	"""Surface water abtractions"""

	series, params = data['series'], data['params']
	dates = series['date']
	swabs_ts = np.zeros(len(dates))

	if node in params['swabs_locs']:
		areas = data['params']['node_areas']
		fac = 1000.0
		freq_flag = data['params']['swabs_f']
		start_date = dates[0]

		series_swabs_ts = series['swabs_ts']
		area = areas[node]
		zone_swabs = params['swabs_locs'][node] - 1

		if freq_flag == 0:
			# daily input just populate
			swabs_ts = series_swabs_ts[:, zone_swabs] / area * fac

		elif freq_flag == 1:
			# if weeks convert to days
			for iday, day in enumerate(dates):
				week = weekdelta(start_date, day)
				swabs_ts[iday] = (series_swabs_ts[week, zone_swabs] / area * fac)

		elif freq_flag == 2:
			global _months
			# if months convert to days
			months = np.zeros(len(dates), dtype=np.int64)
			for iday, day in enumerate(dates):
				months[iday] = monthdelta2(start_date, day)
			swabs_ts = (series_swabs_ts[months, zone_swabs] / area * fac)

	return {'swabs_ts': swabs_ts}

def get_change(data, output, node):
	"""AR) TOTAL STORAGE CHANGE [mm]."""
	p_smd = output['p_smd']
	sw_attenuation = output['sw_attenuation']
	sw_attenuation_rolled = np.roll(sw_attenuation, 1)
	sw_attenuation_difference = sw_attenuation - sw_attenuation_rolled
	sw_attenuation_difference[0] = 0
	p_smd_clipped = np.clip(p_smd, a_min=None, a_max=0)
	p_smd_clipped[0] = 0

	col_change = (output['recharge_store_input'] -
			(output['combined_recharge'] - output['macropore_dir']) +
			(output['interflow_store_input'] - output['interflow_to_rivers']) -
			output['infiltration_recharge'] +
			(output['percol_in_root'] - output['ae'])
			+ p_smd_clipped + sw_attenuation_difference)

	return {'total_storage_change': col_change}
