import numpy
import numpy as np
import swacmod.timer as t
import sys
from . import utils as u


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

    # Fully calculated
    use_rapid_runoff_process = params['rapid_runoff_process'] == 'enabled'
    use_macropore_process = params['macropore_process'] == 'enabled'
    use_fao_process = params['fao_process'] == 'enabled'
    macro_act_factor_A = 0 if mac_opt == 'SMD' else 1
    macro_act_factor_B = 1 - macro_act_factor_A
    tawtew_a_minus_rawrew_a = tawtew_a - rawrew_a
    var3_arr = _make_var3_arr(length, len_class_ri, class_ri, net_rainfall)

    indexes = np.arange(length)
    f = lambda num: macro_rec[months[num]][zone_mac]
    fv = np.vectorize(f)
    var10a_arr = fv(indexes)

    # calculated in loop
    previous_smd_arr = np.zeros(length + 1)
    previous_smd_arr[0] = ssmd
    
    for num in range(length):
        if use_rapid_runoff_process:
            if previous_smd_arr[num] > last_smd or net_rainfall[num] > last_ri:
                col_rapid_runoff_c[num] = value
            else:
                var3 = var3_arr[num]
                col_rapid_runoff_c[num] = _calc_col_rapid_runoff_c(num, len_class_smd, class_smd, previous_smd_arr, values, var3)
            col_rapid_runoff[num] = (0.0 if net_rainfall[num] < 0.0 else (net_rainfall[num] * col_rapid_runoff_c[num]))

        if use_macropore_process:
            macropore = _calc_macropore(net_rainfall, num, col_rapid_runoff, macro_act_factor_A, macro_act, months, zone_mac, macro_act_factor_B, p_smd, macro_prop, macro_limit)

            col_macropore_att[num] = macropore * (1 - var10a_arr[num])
            col_macropore_dir[num] = macropore * var10a_arr[num]

        col_percol_in_root[num] = (net_rainfall[num] - col_rapid_runoff[num]
                          - col_macropore_att[num]
                          - col_macropore_dir[num])

        if use_fao_process:
            col_smd[num] = max(p_smd, 0.0)
            previous_smd_arr[num + 1] = col_smd[num]

            if col_percol_in_root[num] > net_pefac_a[num]:
                col_k_slope[num] = -1.0
            else:
                col_k_slope[num] = _calc_col_k_slope(tawtew_a_minus_rawrew_a, num, tawtew_a, col_smd)

            col_ae[num] = _calc_var13_arr(col_smd, num, rawrew_a, col_percol_in_root, net_pefac_a, tawtew_a, col_k_slope)

            p_smd = col_smd[num] + col_ae[num] - col_percol_in_root[num]
            col_p_smd[num] = p_smd

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

def _calc_col_rapid_runoff_c(num, len_class_smd, class_smd, previous_smd_arr, values, var3):
    var4 = 0
    for i in range(len_class_smd):
        if class_smd[i] >= previous_smd_arr[num]:
            var4 = i
            break
    result = values[var3][var4]
    return result

def _make_var3_arr(length, len_class_ri, class_ri, net_rainfall):
    var3_arr = np.zeros(length, dtype=np.int32)
    for num in range(length):
        for i in range(len_class_ri):
            if class_ri[i] >= net_rainfall[num]:
                var3_arr[num] = i
                break
    return var3_arr

def _calc_macropore(net_rainfall, num, col_rapid_runoff, macro_act_factor_A, macro_act, months, zone_mac, macro_act_factor_B, p_smd, macro_prop, macro_limit):
    var8a = net_rainfall[num] - col_rapid_runoff[num] - (macro_act_factor_A * macro_act[months[num]][zone_mac])
    if var8a > 0.0:
        ma = (macro_act_factor_A * sys.float_info.max) + (macro_act_factor_B * macro_act[months[num]][zone_mac])
        if p_smd < ma:
            var9 = macro_prop[months[num]][zone_mac] * var8a
            var10 = macro_limit[months[num]][zone_mac]
            macropore = min(var10, var9)
        else:
            macropore = 0.0
    else:
        macropore = 0.0
    return macropore

def _calc_col_k_slope(tawtew_a_minus_rawrew_a, num, tawtew_a, col_smd):
    if tawtew_a_minus_rawrew_a[num] == 0.0:
        var12 = 1.0
    else:
        var12 = (tawtew_a[num] - col_smd[num]) / tawtew_a_minus_rawrew_a[num]

    if var12 >= 1.0:
        result = 1.0
    else:
        result = max(var12, 0.0)
    return result

def _calc_var13_arr(col_smd, num, rawrew_a, col_percol_in_root, net_pefac_a, tawtew_a, col_k_slope):
    if col_smd[num] < rawrew_a[num] or col_percol_in_root[num] > net_pefac_a[num]:
        var13 = net_pefac_a[num]
    elif col_smd[num] >= rawrew_a[num] and col_smd[num] <= tawtew_a[num]:
        var13 = col_k_slope[num] * (net_pefac_a[num] - col_percol_in_root[num])
    else:
        var13 = 0.0
    return var13

     
class Lazy_Precipitation:
	def __init__(self, zone_rf, coef_rf, rainfall_ts):
		self.zone_rf = zone_rf
		self.coef_rf = coef_rf
		self.rainfall_ts = rainfall_ts
	
	def __getitem__(self, subscript):
		zone_rf = self.zone_rf[subscript]
		coef_rf = self.coef_rf[subscript]
		rainfall_ts = self.rainfall_ts[:, zone_rf] * coef_rf
		return rainfall_ts
