# -*- coding: utf-8 -*-
"""SWAcMod model functions in Cython."""

# Third Party Libraries
import numpy as np
cimport numpy as np

# Internal modules
from . import utils as u


###############################################################################
def get_precipitation(data, output, node):
    """C) Precipitation [mm/d]."""
    series, params = data['series'], data['params']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]
    rainfall_ts = series['rainfall_ts'][:, zone_rf] * coef_rf
    return {'rainfall_ts': rainfall_ts}


###############################################################################
def get_pe(data, output, node):
    """D) Potential Evapotranspiration (PE) [mm/d]."""
    series, params = data['series'], data['params']

    fao = params['fao_process']
    canopy = params['canopy_process']

    if fao == 'enabled' or canopy == 'enabled':
        zone_pe = params['pe_zone_mapping'][node][0] - 1
        coef_pe = params['pe_zone_mapping'][node][1]
        pe_ts = series['pe_ts'][:, zone_pe] * coef_pe
    else:
        pe_ts = np.zeros(len(series['date']))

    return {'pe_ts': pe_ts}


###############################################################################
def get_pefac(data, output, node):
    """E) Vegetation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
    series, params = data['series'], data['params']

    fao = params['fao_process']
    canopy = params['canopy_process']

    if fao == 'enabled' or canopy == 'enabled':
        zone_lu = params['lu_spatial'][node]
        var1 = (params['kc_list'][series['months']] * zone_lu).sum(axis=1)
        pefac = output['pe_ts'] * var1
    else:
        pefac = np.zeros(len(series['date']))

    return {'pefac': pefac}


###############################################################################
def get_canopy_storage(data, output, node):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params = data['series'], data['params']

    if params['canopy_process'] == 'enabled':
        ftf = params['free_throughfall'][node]
        mcs = params['max_canopy_storage'][node]
        canopy_storage = output['rainfall_ts'] * (1 - ftf)
        canopy_storage[canopy_storage > mcs] = mcs
        #indexes = np.where(canopy_storage > output['pefac'])
        #canopy_storage[indexes] = output['pefac']
        canopy_storage = np.where(canopy_storage > output['pefac'],
                                  output['pefac'], canopy_storage)
    else:
        canopy_storage = np.zeros(len(series['date']))

    return {'canopy_storage': canopy_storage}


###############################################################################
def get_net_pefac(data, output, node):
    """G) Vegetation-factored PE less Canopy Evaporation [mm/d]."""
    net_pefac = output['pefac'] - output['canopy_storage']
    return {'net_pefac': net_pefac}


###############################################################################
def get_precip_to_ground(data, output, node):
    """H) Precipitation at Groundlevel [mm/d]."""
    series, params = data['series'], data['params']
    precip_to_ground = output['rainfall_ts'] - output['canopy_storage']
    return {'precip_to_ground': precip_to_ground}


###############################################################################
def get_snowfall_o(data, output, node):
    """I) Snowfall [mm/d]."""
    series, params = data['series'], data['params']

    if params['snow_process'] == 'enabled':
        zone_tm = params['temperature_zone_mapping'][node] - 1
        snow_fall_temp = params['snow_params'][node][1]
        snow_melt_temp = params['snow_params'][node][2]
        diff = snow_fall_temp - snow_melt_temp
        var1 = series['temperature_ts'][:, zone_tm] - snow_fall_temp
        var3 = 1 - (np.exp(- var1 / diff))**2
        var3[var3 > 0] = 0
        var3 = - var3
        var3[var3 > 1] = 1
        snowfall_o = var3 * output['precip_to_ground']
    else:
        snowfall_o = np.zeros(len(series['date']))

    return {'snowfall_o': snowfall_o}


###############################################################################
def get_rainfall_o(data, output, node):
    """J) Precipitation as Rainfall [mm/d]."""
    rainfall_o = output['precip_to_ground'] - output['snowfall_o']
    return {'rainfall_o': rainfall_o}


###############################################################################
def get_snow(data, output, node):
    """"Multicolumn function.

    K) SnowPack [mm]
    L) SnowMelt [mm/d].
    """
    series, params = data['series'], data['params']

    cdef:
        size_t num
        size_t length = len(series['date'])
        double [:] col_snowpack = np.zeros(length)
        double [:] col_snowmelt = np.zeros(len(series['date']))

    if params['snow_process'] == 'disabled':
        col = {}
        col['snowpack'] = col_snowpack.base
        col['snowmelt'] = col_snowmelt.base
        return col

    cdef:
        double start_snow_pack = params['snow_params'][node][0]
        double snow_fall_temp = params['snow_params'][node][1]
        double snow_melt_temp = params['snow_params'][node][2]
        double diff = snow_fall_temp - snow_melt_temp
        size_t zone_tm = params['temperature_zone_mapping'][node] - 1
        double [:] var3 = (snow_melt_temp -
                           series['temperature_ts'][:, zone_tm])/diff
        double [:] var5 = 1 - (np.exp(var3))**2
        double [:] snowfall_o = output['snowfall_o']
        double var6 = (var5[0] if var5[0] > 0 else 0)
        double snowpack = (1 - var6) * start_snow_pack + snowfall_o[0]

    col_snowmelt[0] = start_snow_pack * var6
    col_snowpack[0] = snowpack
    for num in xrange(1, length):
        if var5[num] < 0:
            var5[num] = 0
        col_snowmelt[num] = snowpack * var5[num]
        snowpack = (1 - var5[num]) * snowpack + snowfall_o[num]
        col_snowpack[num] = snowpack

    return {'snowpack': col_snowpack.base, 'snowmelt': col_snowmelt.base}


###############################################################################
def get_net_rainfall(data, output, node):
    """M) Net Rainfall and Snow Melt [mm/d]."""
    net_rainfall = output['snowmelt'] + output['rainfall_o']
    return {'net_rainfall': net_rainfall}


###############################################################################
def get_rawrew(data, output, node):
    """S) RAWREW (Readily Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']
    if params['fao_process'] == 'enabled':
        rawrew = params['raw'][node][series['months']]
    else:
        rawrew = np.zeros(len(series['date']))
    return {'rawrew': rawrew}


###############################################################################
def get_tawtew(data, output, node):
    """T) TAWTEW (Total Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        tawtew = params['taw'][node][series['months']]
    else:
        tawtew = np.zeros(len(series['date']))

    return {'tawtew': tawtew}


###############################################################################
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
    ssp = params['soil_static_params']
    rrp = params['rapid_runoff_params']
    s_smd = params['smd']

    cdef:
        double [:] col_rapid_runoff_c = np.zeros(len(series['date']))
        double [:] col_rapid_runoff = np.zeros(len(series['date']))
        double [:] col_runoff_recharge = np.zeros(len(series['date']))
        double [:] col_macropore_att = np.zeros(len(series['date']))
        double [:] col_macropore_dir = np.zeros(len(series['date']))
        double [:] col_percol_in_root = np.zeros(len(series['date']))
        double [:] col_p_smd = np.zeros(len(series['date']))
        double [:] col_smd = np.zeros(len(series['date']))
        double [:] col_k_slope = np.zeros(len(series['date']))
        double [:] col_ae = np.zeros(len(series['date']))
        size_t zone_mac = params['macropore_zone_mapping'][node] - 1
        size_t zone_ror = params['rorecharge_zone_mapping'][node] - 1
        size_t zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
        double ssmd = u.weighted_sum(params['soil_spatial'][node],
                                     s_smd['starting_SMD'])
        long long [:] class_smd = np.array(rrp[zone_rro]['class_smd'],
                                           dtype=np.int64)
        long long [:] class_ri = np.array(rrp[zone_rro]['class_ri'],
                                          dtype=np.int64)
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
        double [:, :] ror_act = params['ror_act']
        double [:, :] macro_prop = params['macro_prop']
        double [:, :] macro_limit = params['macro_limit']
        double [:, :] macro_act = params['macro_act']
        double [:, :] macro_rec = params['macro_rec']
        double [:, :] values = np.array(rrp[zone_rro]['values'])
        size_t len_class_smd = len(class_smd)
        size_t len_class_ri = len(class_ri)
        double last_smd = class_smd[-1] - 1
        double last_ri = class_ri[-1] - 1
        double value = values[-1][0]
        double p_smd = ssmd
        double smd = ssmd
        double var2, var5, var7, var8, var8a, var9, var10, var11, var12, var13
        double rapid_runoff_c, rapid_runoff, macropore, percol_in_root
        double net_pefac, tawtew, rawrew
        size_t num, i, var3, var4, var6
        size_t length = len(series['date'])
        double [:] net_rainfall = output['net_rainfall']
        double [:] net_pefac_a = output['net_pefac']
        double [:] tawtew_a = output['tawtew']
        double [:] rawrew_a = output['rawrew']
        long long [:] months = np.array(series['months'], dtype=np.int64)

    for num in xrange(length):
        var2 = net_rainfall[num]

        if params['rapid_runoff_process'] == 'enabled':
            if smd > last_smd or var2 > last_ri:
                rapid_runoff_c = value
            else:
                var3 = 0
                for i in xrange(len_class_ri):
                    if class_ri[i] < var2:
                        var3 += 1
                var4 = 0
                for i in xrange(len_class_smd):
                    if class_smd[i] < smd:
                        var4 += 1
                rapid_runoff_c = values[var3][var4]
            col_rapid_runoff_c[num] = rapid_runoff_c
            var5 = var2 * rapid_runoff_c
            rapid_runoff = (0 if var2 < 0 else var5)
            col_rapid_runoff[num] = rapid_runoff

        var6 = months[num]

        if params['rorecharge_process'] == 'enabled':
            var6a = rapid_runoff - ror_act[var6][zone_ror]
            if var6a > 0:
                var7 = ror_prop[var6][zone_ror] * var6a
                var8 = ror_limit[var6][zone_ror]
                col_runoff_recharge[num] = (var8 if var7 > var8 else var7)
            else:
                col_runoff_recharge[num] = 0.0

        if params['macropore_process'] == 'enabled':
            var8a = var2 - rapid_runoff - macro_act[var6][zone_mac]
            if var8a > 0:
                var9 = macro_prop[var6][zone_mac] * var8a
                var10 = macro_limit[var6][zone_mac]
                macropore = (var10 if var9 > var10 else var9)
            else:
                macropore = 0.0

            var10a = macro_rec[var6][zone_mac]
            col_macropore_att[num] = macropore * (1 - var10a)
            col_macropore_dir[num] = macropore * var10a

        percol_in_root = (var2 - rapid_runoff - col_macropore_att[num])
        col_percol_in_root[num] = percol_in_root
        
        if params['fao_process'] == 'enabled':

            smd = (p_smd if p_smd > 0 else 0.0)
            col_smd[num] = smd
            net_pefac = net_pefac_a[num]
            tawtew = tawtew_a[num]
            rawrew = rawrew_a[num]

            if percol_in_root > net_pefac:
                var11 = -1
            else:
                # tmp div zero
                if (tawtew - rawrew) == 0.0:
                    var12 = 1.0
                else:
                    var12 = (tawtew - smd) / (tawtew - rawrew)
                    
                if var12 >= 1:
                    var11 = 1
                else:
                    var11 = (var12 if var12 >= 0 else 0.0)
            col_k_slope[num] = var11

            var13 = percol_in_root
            if smd < rawrew or percol_in_root > net_pefac:
                var13 = net_pefac
            elif smd >= rawrew and smd <= tawtew:
                var13 = var11 * (net_pefac - percol_in_root)
            else:
                var13 = 0.0
            col_ae[num] = var13
            p_smd = smd + var13 - percol_in_root
            col_p_smd[num] = p_smd

    col = {}
    col['rapid_runoff_c'] = col_rapid_runoff_c.base
    col['rapid_runoff'] = col_rapid_runoff.base
    col['runoff_recharge'] = col_runoff_recharge.base
    col['macropore_att'] = col_macropore_att.base
    col['macropore_dir'] = col_macropore_dir.base
    col['percol_in_root'] = col_percol_in_root.base
    col['p_smd'] = col_p_smd.base
    col['smd'] = col_smd.base
    col['k_slope'] = col_k_slope.base
    col['ae'] = col_ae.base

    return col



###############################################################################
def get_unutilised_pe(data, output, node):
    """Z) Unutilised PE [mm/d]."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        unutilised_pe = output['net_pefac'] - output['ae']
        unutilised_pe[unutilised_pe < 0] = 0
    else:
        unutilised_pe = np.zeros(len(series['date']))

    return {'unutilised_pe': unutilised_pe}


###############################################################################
def get_rejected_recharge(data, output, node):
    """AA) Rejected Recharge."""
    series, params = data['series'], data['params']
    rej = params['percolation_rejection']

    rejected_recharge = np.zeros(len(series['date']))
    if params['fao_process'] == 'enabled':
        zone = params['lu_spatial'][node]
        rej = (np.array(rej['percolation_rejection']) * zone).sum(axis=0)
        perc = np.copy(output['p_smd'])
        perc[perc > 0] = 0
        perc = - perc
        rejected_recharge[perc > rej] = perc[perc > rej] - rej

    return {'rejected_recharge': rejected_recharge}


###############################################################################
def get_perc_through_root(data, output, node):
    """AB) Percolation Through the Root Zone [mm/d]."""
    params = data['params']

    if params['fao_process'] == 'enabled':
        perc = np.copy(output['p_smd'])
        perc[perc > 0] = 0
        perc = - perc
    else:
        perc = np.copy(output['percol_in_root'])

    return {'perc_through_root': perc - output['rejected_recharge']}


###############################################################################
def get_subroot_leak(data, output, node):
    """AC) Sub Root Zone Leakege / Inputs [mm/d]."""
    series, params = data['series'], data['params']

    if params['leakage_process'] == 'enabled':
        zone_sr = params['subroot_zone_mapping'][node][0] - 1
        coef_sr = params['subroot_zone_mapping'][node][1]
        slf = params['subsoilzone_leakage_fraction'][node]
        subroot_leak = series['subroot_leakage_ts'][:, zone_sr] * coef_sr * slf
    else:
        subroot_leak = np.zeros(len(series['date']))

    return {'subroot_leak': subroot_leak}


###############################################################################
def get_interflow_bypass(data, output, node):
    """AD) Bypassing the Interflow Store [mm/d]."""
    params = data['params']
    if params['interflow_process'] == 'enabled':
        coef = params['interflow_params'][node][1]
    else:
        coef = 1.0

    interflow_bypass = coef * (output['perc_through_root'] +
                               output['subroot_leak'])

    return {'interflow_bypass': interflow_bypass}


###############################################################################
def get_interflow_store_input(data, output, node):
    """AE) Input to Interflow Store [mm/d]."""
    interflow_store_input = (output['perc_through_root'] +
                             output['subroot_leak'] -
                             output['interflow_bypass'])

    return {'interflow_store_input': interflow_store_input}


###############################################################################
def get_interflow(data, output, node):
    """Multicolumn function.

    AF) Interflow Store Volume [mm]
    AG) Infiltration Recharge [mm/d]
    AH) Interflow to Surface Water Courses [mm/d]
    """
    series, params = data['series'], data['params']

    cdef:
        size_t length = len(series['date'])
        double [:] col_interflow_volume = np.zeros(length)
        double [:] col_infiltration_recharge = np.zeros(length)
        double [:] col_interflow_to_rivers = np.zeros(length)
        double [:] interflow_store_input = output['interflow_store_input']
        double var0 = params['interflow_params'][node][0]
        double var5 = params['interflow_params'][node][2]
        double var8 = params['interflow_params'][node][3]
        double volume = var0
        double recharge = (var5 if volume >= var5 else volume)
        double rivers = (volume - recharge) * var8
        size_t num
        

    if params['interflow_process'] == 'enabled':
        col_interflow_volume[0] = volume
        col_infiltration_recharge[0] = recharge
        col_interflow_to_rivers[0] = rivers

        for num in xrange(1, length):
            var1 = volume - (var5 if var5 < volume else volume)
            volume = interflow_store_input[num-1] + var1 * (1 - var8)
            col_interflow_volume[num] = volume
            if volume >= var5:
                col_infiltration_recharge[num] = var5
            else:
                col_infiltration_recharge[num] = volume
            var6 = (col_interflow_volume[num] - col_infiltration_recharge[num])
            col_interflow_to_rivers[num] = var6 * var8

    col = {}
    col['interflow_volume'] = col_interflow_volume.base
    col['infiltration_recharge'] = col_infiltration_recharge.base
    col['interflow_to_rivers'] = col_interflow_to_rivers.base

    return col


###############################################################################
def get_recharge_store_input(data, output, node):
    """AI) Input to Recharge Store [mm/d]."""
    recharge_store_input = (output['infiltration_recharge'] +
                            output['interflow_bypass'] +
                            output['macropore_att'] +
                            output['runoff_recharge'])

    return {'recharge_store_input': recharge_store_input}


###############################################################################
def get_recharge(data, output, node):
    """Multicolumn function.

    AJ) Recharge Store Volume [mm]
    AK) RCH: Combined Recharge [mm/d]
    """
    series, params = data['series'], data['params']

    cdef:
        size_t length = len(series['date'])
        double [:] col_recharge_store = np.zeros(length)
        double [:] col_combined_recharge = np.zeros(length)
        double irs = params['recharge_attenuation_params'][node][0]
        double rlp = params['recharge_attenuation_params'][node][1]
        double rll = params['recharge_attenuation_params'][node][2]
        double recharge
        double [:] recharge_store_input = output['recharge_store_input']
        rep_zone = data['params']['reporting_zone_mapping'][node]
        size_t num
        double var1, var2

    for num in xrange(length):
        if params['recharge_attenuation_process'] == 'enabled':
            if num == 0:
                recharge = irs
            else:
                recharge = (recharge_store_input[num - 1] +
                            col_recharge_store[num - 1] -
                            col_combined_recharge[num - 1])
            col_recharge_store[num] = recharge
            var1 = recharge * rlp
            col_combined_recharge[num] = (rll if var1 > rll else var1)
        else:
            col_combined_recharge[num] = recharge_store_input[num]
        col_combined_recharge[num] += output['macropore_dir'][num]

    col = {}
    col['recharge_store'] = col_recharge_store.base
    col['combined_recharge'] = col_combined_recharge.base
    return col


###############################################################################
def get_combined_str(data, output, node):
    """Multicolumn function.

    AL) SW Attenuation Store [mm].
    AM) STR: Combined Surface Flow To Surface Water Courses [mm/d].
    """
    series, params = data['series'], data['params']

    cdef:
        size_t length = len(series['date'])
        double [:] col_attenuation = np.zeros(length)
        double [:] col_combined_str = np.zeros(length)
        double rlp = params['sw_params'][node][1]
        double base = (params['sw_params'][node][0] +
                       output['interflow_to_rivers'][0] +
                       output['rapid_runoff'][0] -
                       output['runoff_recharge'][0])
        size_t num

    if params['sw_process'] == 'enabled':
        col_combined_str[0] = rlp * base
        col_attenuation[0] = base - col_combined_str[0]
        for num in xrange(1, length):
            base = (col_attenuation[num-1] +
                    output['interflow_to_rivers'][num] +
                    output['rapid_runoff'][num] -
                    output['runoff_recharge'][num] +
                    output['rejected_recharge'][num])
            col_combined_str[num] = rlp * base
            col_attenuation[num] = base - col_combined_str[num]
    else:
        col_combined_str = (output['interflow_to_rivers'] +
                            output['rapid_runoff'] -
                            output['runoff_recharge'])

    col = {}
    col['sw_attenuation'] = col_attenuation.base
    col['combined_str'] = col_combined_str.base

    return col


###############################################################################
def get_combined_ae(data, output, node):
    """AN) AE: Combined AE [mm/d]."""
    combined_ae = output['canopy_storage'] + output['ae']
    return {'combined_ae': combined_ae}


###############################################################################
def get_evt(data, output, node):
    """AO) EVT: Unitilised PE [mm/d]."""
    return {'evt': output['unutilised_pe']}


###############################################################################
def get_average_in(data, output, node):
    """AP) AVERAGE IN [mm]."""
    series, params = data['series'], data['params']
    average_in = output['rainfall_ts'] + output['subroot_leak']
    return {'average_in': average_in}


###############################################################################
def get_average_out(data, output, node):
    """AQ) AVERAGE OUT [mm]."""
    average_out = (output['combined_str'] +
                   output['combined_recharge'] +
                   output['ae'] +
                   output['canopy_storage'])

    return {'average_out': average_out}


###############################################################################
def get_change(data, output, node):
    """AR) TOTAL STORAGE CHANGE [mm]."""
    series = data['series']

    cdef:
        size_t length = len(series['date'])
        double [:] col_change = np.zeros(length)
        size_t num
    for num in xrange(1, length):
        col_change[num] = output['recharge_store'][num] - \
                          output['recharge_store'][num - 1] + \
                          output['interflow_volume'][num] - \
                          output['interflow_volume'][num - 1] + \
                          output['smd'][num] - \
                          output['smd'][num - 1] + \
                          output['snowpack'][num] - \
                          output['snowpack'][num - 1]

    return {'total_storage_change': col_change.base}


###############################################################################
def get_balance(data, output, node):
    """AS) BALANCE [mm]."""
    balance = (output['average_in'] -
               output['average_out'] -
               output['total_storage_change'])

    return {'balance': balance}


###############################################################################
def aggregate(output, area, reporting=None, index=None):
    """Aggregate reporting."""
    new_rep = {}
    
    for key in output:
        if isinstance(index, list):
            new_rep[key] = output[key][index].mean(dtype=np.float64) * area
        elif index is not None:
            new_rep[key] = output[key][index] * area
        else:
            new_rep[key] = output[key] * area
        if reporting:
            new_rep[key] += reporting[key]
    return new_rep

###############################################################################
def get_sfr_flows(sorted_by_ca, idx, runoff, done, areas, swac_seg_dic, ro, flow, nodes, per):
    
    """get flows for one period"""
    
    ro[:] = 0.0
    flow[:] = 0.0
    
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, hcond1,
         depth, width) = line

        acc = 0.0

        # accumulate pre-stream flows into network
        while downstr > 1:

            str_flag = sorted_by_ca[node_swac][idx['str_flag']]
            node_mf = sorted_by_ca[node_swac][idx['node_mf']]

            # not str
            if str_flag < 1 or node_mf < 1:
                # not not done
                if done[node_swac] < 1:
                    acc += (runoff[(nodes * per) + node_swac] * areas[node_swac])
                    done[node_swac] += 1
            else:
                # stream cell
                iseg = swac_seg_dic[node_swac]

                # not done
                if done[node_swac] < 1:
                    ro[iseg - 1] += runoff[(nodes * per) + node_swac] * areas[node_swac]
                    flow[iseg - 1] += acc
                    done[node_swac] += 1
                    acc = 0.0

                # stream cell been done
                elif acc > 0.0:
                    flow[iseg - 1] += acc
                    acc = 0.0

            # new node
            node_swac = downstr
            # get new downstr node
            downstr = sorted_by_ca[node_swac][idx['downstr']]
            
    return ro, flow

###############################################################################

def get_sfr_file(data, runoff):
    """get SFR object."""
    
    import flopy
    import csv
    import numpy as np
    import copy
    from collections import OrderedDict
    import os.path
    from swacmod.input_output import print_progress

    areas = data['params']['node_areas']
    fileout = data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    m = flopy.modflow.Modflow(modelname=path,
                              version='mfusg', structured=False)

    nper = len(data['params']['time_periods'])
    nodes = data['params']['num_nodes']

    njag = nodes + 2
    lenx = (njag/2) - (nodes/2)

    dis = flopy.modflow.ModflowDisU(
        m,
        nodes=nodes,
        nper=nper, iac=[njag] + (nodes - 1) * [0],
        ja=np.zeros((njag), dtype=int),
        njag=njag,
        idsymrd=1,
        cl1=np.zeros((lenx)),
        cl2=np.zeros((lenx)),
        fahl=np.zeros((lenx)))

    m.dis = m.disu

    sorted_by_ca = OrderedDict(sorted(data['params']['routing_toplogy'].items(),
                                      key=lambda x: x[1][4]))

    names = ['downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z',
             'bed_thk', 'str_k', 'hcond1', 'depth', 'width']

    idx = dict((y, x) for (x, y) in enumerate(names))
    
    nstrm = nss = sum(value[idx['str_flag']] > 0 for value in sorted_by_ca.values()) 

    istcb1, istcb2 = data['params']['istcb1'], data['params']['istcb2']
    
    sd = flopy.modflow.ModflowSfr2.get_empty_segment_data(nss)
    rd = flopy.modflow.ModflowSfr2.get_empty_reach_data(nstrm,
                                                        structured=False)

    seg_data = {}
    swac_seg_dic = {}
    seg_swac_dic = {}
    done = np.zeros((nodes), dtype=int)

    # initialise reach & segment data
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, hcond1,
         depth, width) = line
        
        str_k = hcond1 = 0.0
        
        if str_flag > 0 and node_mf > 0:
            swac_seg_dic[node_swac] = str_count + 1
            seg_swac_dic[str_count + 1] = node_swac
            rd[str_count]['node'] = node_mf - 1 # external
            rd[str_count]['iseg'] = str_count + 1 # serial
            rd[str_count]['ireach'] = 1 # str_count + 1 # serial
            rd[str_count]['rchlen'] = length # external
            rd[str_count]['strtop'] = z # external
            rd[str_count]['strthick'] = bed_thk # constant (for now)
            rd[str_count]['strhc1'] = str_k # constant (for now)

            # segment data
            sd[str_count]['nseg'] = str_count + 1 # serial
            sd[str_count]['icalc'] = 0 # constant
            sd[str_count]['outseg'] = 0
            sd[str_count]['iupseg'] = 0 # constant (0)
            sd[str_count]['flow'] = 0.0  # constant (for now - swac)
            sd[str_count]['runoff'] = 0.0  # constant (for now - swac)
            sd[str_count]['etsw'] = 0.0 # # cotnstant (0)
            sd[str_count]['pptsw'] = 0.0 # constant (0)
            sd[str_count]['hcond1'] = hcond1 # get from lpf
            sd[str_count]['thickm1'] = bed_thk # constant
            sd[str_count]['elevup'] = z # get from mf
            sd[str_count]['width1'] = width # constant
            sd[str_count]['depth1'] = depth # constant
            sd[str_count]['width2'] = width # constant
            sd[str_count]['depth2'] = depth # constant
            
            # inc stream counter
            str_count += 1

    for iseg in xrange(nss):
        node_swac = seg_swac_dic[iseg + 1]
        downstr = sorted_by_ca[node_swac][idx['downstr']]
        if downstr in swac_seg_dic:
            sd[iseg]['outseg'] = swac_seg_dic[downstr]
        else:
            sd[iseg ]['outseg'] = 0

    ro, flow = np.zeros((nss)), np.zeros((nss))
            
    # populate runoff and flow
    for per in xrange(nper):

        print_progress(per + 1, nper, 'SWAcMod Serial     ')

        ro, flow = get_sfr_flows(sorted_by_ca, idx, runoff, done, areas,
                                 swac_seg_dic, ro, flow, nodes, per)

        for iseg in xrange(nss):
            sd[iseg]['runoff'] = ro[iseg]
            sd[iseg]['flow'] = flow[iseg]

        # add segment data for this period
        seg_data[per] = copy.deepcopy(sd)
        done[:] = 0
        # for iseg in xrange(nss):
        #     sd[iseg]['runoff'] = 0.0
        #     sd[iseg]['flow'] = 0.0
            
    isfropt = 1
    sfr = flopy.modflow.mfsfr2.ModflowSfr2(m, nstrm=nstrm, nss=nss, nsfrpar=0,
                                           nparseg=0, const=None, dleak=0.0001,
                                           ipakcb=istcb1, istcb2=istcb2, isfropt=isfropt,
                                           nstrail=10, isuzn=1, nsfrsets=30, irtflg=0,
                                           numtim=2, weight=0.75, flwtol=0.0001,
                                           reach_data=rd, segment_data=seg_data,
                                           channel_geometry_data=None,
                                           channel_flow_data=None, dataset_5=None,
                                           irdflag=0, iptflag=0, reachinput=True,
                                           transroute=False, tabfiles=False,
                                           tabfiles_dict=None, extension='sfr',
                                           unit_number=None, filenames=None)
    # compute the slopes
    m.sfr.get_slopes()

    return sfr

