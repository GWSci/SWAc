# -*- coding: utf-8 -*-
"""SWAcMod model functions in Cython."""

# Third Party Libraries
import numpy as np
cimport numpy as np

# Internal modules
from . import utils as u


###############################################################################
def get_pefac(data, output, node):
    """E) Vegitation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
    series, params = data['series'], data['params']

    fao = params['fao_process']
    canopy = params['canopy_process']

    if fao == 'enabled' or canopy == 'enabled':
        zone_pe = params['pe_zone_mapping'][node][0] - 1
        coef_pe = params['pe_zone_mapping'][node][1]
        zone_lu = params['lu_spatial'][node]
        var1 = (params['kc_list'][series['months']] * zone_lu).sum(axis=1)
        pefac = series['pe_ts'][:, zone_pe] * coef_pe * var1
    else:
        pefac = np.zeros(len(series['date']))

    return {'pefac': pefac}


###############################################################################
def get_canopy_storage(data, output, node):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params = data['series'], data['params']

    if params['canopy_process'] == 'enabled':
        zone_rf = params['rainfall_zone_mapping'][node][0] - 1
        coef = params['rainfall_zone_mapping'][node][1]
        ftf = params['free_throughfall'][node]
        mcs = params['max_canopy_storage'][node]
        canopy_storage = series['rainfall_ts'][:, zone_rf] * coef * (1 - ftf)
        canopy_storage[canopy_storage > mcs] = mcs
        indexes = np.where(canopy_storage > output['pefac'])
        canopy_storage[indexes] = output['pefac']
    else:
        canopy_storage = np.zeros(len(series['date']))

    return {'canopy_storage': canopy_storage}


###############################################################################
def get_net_pefac(data, output, node):
    """G) Vegitation-factored PE less Canopy Evaporation [mm/d]."""
    net_pefac = output['pefac'] - output['canopy_storage']
    return {'net_pefac': net_pefac}


###############################################################################
def get_precip_to_ground(data, output, node):
    """H) Precipitation at Groundlevel [mm/d]."""
    series, params = data['series'], data['params']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    precip_to_ground = (series['rainfall_ts'][:, zone_rf] * coef_rf -
                        output['canopy_storage'])

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

    if params['snow_process'] == 'enabled':
        col_snowmelt[0] = start_snow_pack * var6
        col_snowpack[0] = snowpack
        for num in range(1, length):
            if var5[num] < 0:
                var5[num] = 0
            col_snowmelt[num] = snowpack * var5[num]
            snowpack = (1 - var5[num]) * snowpack + snowfall_o[num]
            col_snowpack[num] = snowpack

    col = {}
    col['snowpack'] = col_snowpack.base
    col['snowmelt'] = col_snowmelt.base

    return col


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
def get_tawrew(data, output, node):
    """T) TAWREW (Total Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        tawrew = params['taw'][node][series['months']]
    else:
        tawrew = np.zeros(len(series['date']))

    return {'tawrew': tawrew}


###############################################################################
def get_ae(data, output, node):
    """Multicolumn function.

    N) Rapid Runoff Class [%]
    O) Rapid Runoff [mm/d]
    P) Runoff Recharge [mm/d]
    Q) MacroPore: Bypass of Root Zone and Interflow [mm/d]
    R) Percolation into Root Zone [mm/d]
    U) Potential Soil Moisture Defecit (pSMD) [mm]
    V) Soil Moisture Defecit (SMD) [mm]
    W) Ks (slope factor) [-]
    X) AE (actual evapotranspiration) [mm/d]
    """
    series, params = data['series'], data['params']
    ssp = params['soil_static_params']
    rrp = params['rapid_runoff_params']

    cdef:
        double [:] col_rapid_runoff_c = np.zeros(len(series['date']))
        double [:] col_rapid_runoff = np.zeros(len(series['date']))
        double [:] col_runoff_recharge = np.zeros(len(series['date']))
        double [:] col_macropore = np.zeros(len(series['date']))
        double [:] col_percol_in_root = np.zeros(len(series['date']))
        double [:] col_p_smd = np.zeros(len(series['date']))
        double [:] col_smd = np.zeros(len(series['date']))
        double [:] col_k_slope = np.zeros(len(series['date']))
        double [:] col_ae = np.zeros(len(series['date']))
        size_t zone_mac = params['macropore_zone_mapping'][node] - 1
        size_t zone_ror = params['rorecharge_zone_mapping'][node] - 1
        size_t zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
        double ssmd = u.weighted_sum(params['soil_spatial'][node],
                                     ssp['starting_SMD'])
        long long [:] class_smd = np.array(rrp[zone_rro]['class_smd'],
                                           dtype=np.int64)
        long long [:] class_ri = np.array(rrp[zone_rro]['class_ri'],
                                          dtype=np.int64)
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
        double [:, :] macro_prop = params['macro_prop']
        double [:, :] macro_limit = params['macro_limit']
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
        double net_pefac, tawrew, rawrew
        size_t num, i, var3, var4, var6
        size_t length = len(series['date'])
        double [:] net_rainfall = output['net_rainfall']
        double [:] net_pefac_a = output['net_pefac']
        double [:] tawrew_a = output['tawrew']
        double [:] rawrew_a = output['rawrew']
        long long [:] months = np.array(series['months'], dtype=np.int64)

    for num in range(length):
        var2 = net_rainfall[num]

        if params['rapid_runoff_process'] == 'enabled':
            if smd > last_smd or var2 > last_ri:
                rapid_runoff_c = value
            else:
                var3 = 0
                for i in range(len_class_ri):
                    if class_ri[i] < var2:
                        var3 += 1
                var4 = 0
                for i in range(len_class_smd):
                    if class_smd[i] < smd:
                        var4 += 1
                rapid_runoff_c = values[var3][var4]
            col_rapid_runoff_c[num] = rapid_runoff_c
            var5 = var2 * rapid_runoff_c
            rapid_runoff = (0 if var2 < 0 else var5)
            col_rapid_runoff[num] = rapid_runoff

        var6 = months[num]

        if params['rorecharge_process'] == 'enabled':
            var7 = ror_prop[var6][zone_ror] * rapid_runoff
            var8 = ror_limit[var6][zone_ror]
            col_runoff_recharge[num] = (var8 if var7 > var8 else var7)

        if params['macropore_process'] == 'enabled':
            var8a = var2 - rapid_runoff
            var9 = macro_prop[var6][zone_mac] * var8a
            var10 = macro_limit[var6][zone_mac]
            macropore = (var10 if var9 > var10 else var9)
            col_macropore[num] = macropore

        percol_in_root = (var2 - rapid_runoff - macropore)
        col_percol_in_root[num] = percol_in_root

        if params['fao_process'] == 'enabled':

            smd = (p_smd if p_smd > 0 else 0.0)
            col_smd[num] = smd
            net_pefac = net_pefac_a[num]
            tawrew = tawrew_a[num]
            rawrew = rawrew_a[num]

            if percol_in_root > net_pefac:
                var11 = -1
            else:
                var12 = (tawrew - smd) / (tawrew - rawrew)
                if var12 >= 1:
                    var11 = 1
                else:
                    var11 = (var12 if var12 >= 0 else 0.0)
            col_k_slope[num] = var11

            var13 = percol_in_root
            if smd < rawrew or percol_in_root > net_pefac:
                var13 = net_pefac
            elif smd >= rawrew and smd <= tawrew:
                var13 = var11 * (net_pefac - percol_in_root) + percol_in_root
            col_ae[num] = var13

            p_smd = smd + var13 - percol_in_root
            col_p_smd[num] = p_smd

    col = {}
    col['rapid_runoff_c'] = col_rapid_runoff_c.base
    col['rapid_runoff'] = col_rapid_runoff.base
    col['runoff_recharge'] = col_runoff_recharge.base
    col['macropore'] = col_macropore.base
    col['percol_in_root'] = col_percol_in_root.base
    col['p_smd'] = col_p_smd.base
    col['smd'] = col_smd.base
    col['k_slope'] = col_k_slope.base
    col['ae'] = col_ae.base

    return col


###############################################################################
def get_unutilised_pe(data, output, node):
    """Y) Unutilised PE [mm/d]."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        unutilised_pe = output['net_pefac'] - output['ae']
        unutilised_pe[unutilised_pe < 0] = 0
    else:
        unutilised_pe = np.zeros(len(series['date']))

    return {'unutilised_pe': unutilised_pe}


###############################################################################
def get_perc_through_root(data, output, node):
    """Z) Percolation Through the Root Zone [mm/d]."""
    params = data['params']

    if params['fao_process'] == 'enabled':
        perc_through_root = np.copy(output['p_smd'])
        perc_through_root[perc_through_root > 0] = 0
        perc_through_root = - perc_through_root
    else:
        perc_through_root = np.copy(output['percol_in_root'])

    return {'perc_through_root': perc_through_root}


###############################################################################
def get_subroot_leak(data, output, node):
    """AA) Sub Root Zone Leakege / Inputs [mm/d]."""
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
    """AB) Bypassing the Interflow Store [mm/d]."""
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
    """AC) Input to Interflow Store [mm/d]."""
    interflow_store_input = (output['perc_through_root'] +
                             output['subroot_leak'] -
                             output['interflow_bypass'])

    return {'interflow_store_input': interflow_store_input}


###############################################################################
def get_interflow(data, output, node):
    """Multicolumn function.

    AD) Interflow Store Volume [mm]
    AE) Infiltration Recharge [mm/d]
    AF) Interflow to Surface Water Courses [mm/d]
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

    for num in range(1, length):
        if params['interflow_process'] == 'enabled':
            var1 = volume - (var5 if var5 < volume else volume)
            volume = interflow_store_input[num-1] + var1 * (1 - var8)
            col_interflow_volume[num] = volume
        if volume >= var5:
            col_infiltration_recharge[num] = var5
        else:
            col_infiltration_recharge[num] = volume
        col_interflow_to_rivers[num] = (col_interflow_volume[num] -
                                        col_infiltration_recharge[num]) * var8

    col = {}
    col['interflow_volume'] = col_interflow_volume.base
    col['infiltration_recharge'] = col_infiltration_recharge.base
    col['interflow_to_rivers'] = col_interflow_to_rivers.base

    return col


###############################################################################
def get_recharge_store_input(data, output, node):
    """AG) Input to Recharge Store [mm/d]."""
    recharge_store_input = (output['infiltration_recharge'] +
                            output['interflow_bypass'] +
                            output['macropore'] +
                            output['runoff_recharge'])

    return {'recharge_store_input': recharge_store_input}


###############################################################################
def get_recharge(data, output, node):
    """Multicolumn function.

    AH) Recharge Store Volume [mm]
    AI) RCH: Combined Recharge [mm/d]
    """
    series, params = data['series'], data['params']

    cdef:
        size_t length = len(series['date'])
        double [:] col_recharge_store = np.zeros(length)
        double [:] col_combined_recharge = np.zeros(length)
        double irs = params['recharge_attenuation_params'][node][0]
        double rlp = params['recharge_attenuation_params'][node][1]
        double rll = params['recharge_attenuation_params'][node][2]
        double recharge = irs
        double combined = recharge * rlp
        double [:] recharge_store_input = output['recharge_store_input']
        size_t num
        double var1, var2

    if params['recharge_attenuation_process'] == 'enabled':
        col_recharge_store[0] = recharge
        col_combined_recharge[0] = combined
        for num in range(1, length):
            var1 = recharge * rlp
            recharge = (recharge + recharge_store_input[num-1] -
                        (var1 if var1 < rll else rll))
            col_recharge_store[num] = recharge
            var2 = recharge * rlp
            if var2 > rll:
                col_combined_recharge[num] = rll
            else:
                col_combined_recharge[num] = var2

    col = {}
    col['recharge_store'] = col_recharge_store.base
    col['combined_recharge'] = col_combined_recharge.base

    return col


###############################################################################
def get_combined_str(data, output, node):
    """AJ) STR: Combined Surface Flow To Surface Water Courses [mm/d]."""
    combined_str = (output['interflow_to_rivers'] +
                    output['rapid_runoff'] -
                    output['runoff_recharge'])

    return {'combined_str': combined_str}


###############################################################################
def get_combined_ae(data, output, node):
    """AK) AE: Combined AE [mm/d]."""
    combined_ae = output['canopy_storage'] + output['ae']
    return {'combined_ae': combined_ae}


###############################################################################
def get_evt(data, output, node):
    """AL) EVT: Unitilised PE [mm/d]."""
    return {'evt': output['unutilised_pe']}


###############################################################################
def get_average_in(data, output, node):
    """AM) AVERAGE IN [mm]."""
    series, params = data['series'], data['params']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    average_in = (series['rainfall_ts'][:, zone_rf] * coef_rf +
                  output['subroot_leak'])

    return {'average_in': average_in}


###############################################################################
def get_average_out(data, output, node):
    """AN) AVERAGE OUT [mm]."""
    average_out = (output['combined_str'] +
                   output['combined_recharge'] +
                   output['ae'] +
                   output['canopy_storage'])

    return {'average_out': average_out}


###############################################################################
def get_balance(data, output, node):
    """AO) BALANCE [mm]."""
    balance = output['average_in'] - output['average_out']
    return {'balance': balance}

