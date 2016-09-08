#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod model functions."""

# Third Party Libraries
import numpy as np

# Internal modules
from . import utils as u


###############################################################################
def get_pefac(data, node):
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
def get_canopy_storage(data, node):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

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
def get_net_pefac(data, node):
    """G) Vegitation-factored PE less Canopy Evaporation [mm/d]."""
    output = data['output']
    net_pefac = output['pefac'] - output['canopy_storage']
    return {'net_pefac': net_pefac}


###############################################################################
def get_precip_to_ground(data, node):
    """H) Precipitation at Groundlevel [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    precip_to_ground = (series['rainfall_ts'][:, zone_rf] * coef_rf -
                        output['canopy_storage'])

    return {'precip_to_ground': precip_to_ground}


###############################################################################
def get_snowfall_o(data, node):
    """I) Snowfall [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

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
def get_rainfall_o(data, node):
    """J) Precipitation as Rainfall [mm/d]."""
    output = data['output']
    rainfall_o = output['precip_to_ground'] - output['snowfall_o']
    return {'rainfall_o': rainfall_o}


###############################################################################
def get_snow(data, node):
    """"Multicolumn function.

    K) SnowPack [mm]
    L) SnowMelt [mm/d].
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['snowpack', 'snowmelt']:
        col[key] = np.zeros(len(series['date']))

    if params['snow_process'] == 'enabled':
        start_snow_pack = params['snow_params'][node][0]
        snow_fall_temp = params['snow_params'][node][1]
        snow_melt_temp = params['snow_params'][node][2]
        diff = snow_fall_temp - snow_melt_temp
        zone_tm = params['temperature_zone_mapping'][node] - 1
        var3 = series['temperature_ts'][:, zone_tm] - snow_melt_temp
        var5 = 1 - (np.exp(- var3 / diff))**2
        var5[var5 < 0] = 0
        col['snowmelt'][0] = start_snow_pack * var5[0]
        snowpack = (start_snow_pack + output['snowfall_o'][0] -
                    start_snow_pack * var5[0])
        col['snowpack'][0] = snowpack
        for num in range(1, len(series['date'])):
            snowmelt = snowpack * var5[num]
            col['snowmelt'][num] = snowmelt
            snowpack += output['snowfall_o'][num] - snowmelt
            col['snowpack'][num] = snowpack

    return col


###############################################################################
def get_net_rainfall(data, node):
    """M) Net Rainfall and Snow Melt [mm/d]."""
    output = data['output']
    net_rainfall = output['snowmelt'] + output['rainfall_o']
    return {'net_rainfall': net_rainfall}


###############################################################################
def get_rawrew(data, node):
    """S) RAWREW (Readily Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']
    if params['fao_process'] == 'enabled':
        rawrew = params['RAW'][node][series['months']]
    else:
        rawrew = np.zeros(len(series['date']))
    return {'rawrew': rawrew}


###############################################################################
def get_tawrew(data, node):
    """T) TAWREW (Total Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        tawrew = params['TAW'][node][series['months']]
    else:
        tawrew = np.zeros(len(series['date']))

    return {'tawrew': tawrew}


###############################################################################
def get_ae(data, node):
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
    series, params, output = data['series'], data['params'], data['output']
    order = ['rapid_runoff_c', 'rapid_runoff', 'runoff_recharge', 'macropore',
             'percol_in_root', 'p_smd', 'smd', 'k_slope', 'ae']
    col = {}
    for key in order:
        col[key] = np.zeros(len(series['date']))

    ssmd = u.weighted_sum(params['soil_spatial'][node],
                          params['soil_static_params']['starting_SMD'])

    zone_mac = params['macropore_zone_mapping'][node] - 1
    zone_ror = params['rorecharge_zone_mapping'][node] - 1
    zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
    prop = params['rorecharge_proportion']
    class_smd = params['rapid_runoff_params'][zone_rro]['class_smd']
    class_ri = params['rapid_runoff_params'][zone_rro]['class_ri']
    values = params['rapid_runoff_params'][zone_rro]['values']
    last_smd = class_smd[-1]
    last_ri = class_ri[-1]
    value = values[-1][0]

    p_smd, smd = ssmd, ssmd
    for num in range(len(series['date'])):
        var2 = output['net_rainfall'][num]

        if params['rapid_runoff_process'] == 'enabled':
            if smd > (last_smd - 1) or var2 > (last_ri - 1):
                rapid_runoff_c = value
            else:
                var3 = len([i for i in class_ri if i < var2])
                var4 = len([i for i in class_smd if i < smd])
                rapid_runoff_c = values[var3][var4]
            col['rapid_runoff_c'][num] = rapid_runoff_c
            var5 = output['net_rainfall'][num] * rapid_runoff_c
            rapid_runoff = (0 if var2 < 0 else var5)
            col['rapid_runoff'][num] = rapid_runoff

        var6 = series['months'][num] + 1

        if params['rorecharge_process'] == 'enabled':
            var7 = prop[var6][zone_ror] * rapid_runoff
            var8 = params['rorecharge_limit'][var6][zone_ror]
            col['runoff_recharge'][num] = (var8 if var7 > var8 else var7)

        if params['macropore_process'] == 'enabled':
            var8a = var2 - rapid_runoff
            var9 = params['macropore_proportion'][var6][zone_mac] * var8a
            var10 = params['macropore_limit'][var6][zone_mac]
            marcopore = (var10 if var9 > var10 else var9)
            col['macropore'][num] = marcopore

        percol_in_root = (var2 - rapid_runoff - marcopore)
        col['percol_in_root'][num] = percol_in_root

        if params['fao_process'] == 'enabled':

            smd = (p_smd if p_smd > 0 else 0.0)
            col['smd'][num] = smd
            net_pefac = output['net_pefac'][num]
            tawrew = output['tawrew'][num]
            rawrew = output['rawrew'][num]

            if percol_in_root > net_pefac:
                var11 = -1
            else:
                var12 = (tawrew - smd) / (tawrew - rawrew)
                if var12 >= 1:
                    var11 = 1
                else:
                    var11 = (var12 if var12 >= 0 else 0.0)
            col['k_slope'][num] = var11

            if smd < rawrew or percol_in_root > net_pefac:
                var13 = net_pefac
            else:
                if smd >= rawrew and smd <= tawrew:
                    var14 = net_pefac - percol_in_root
                    var13 = var11 * var14 + percol_in_root
                else:
                    var13 = percol_in_root
            col['ae'][num] = var13

            p_smd = smd + var13 - percol_in_root
            col['p_smd'][num] = p_smd

    return col


###############################################################################
def get_unutilised_pe(data, node):
    """Y) Unutilised PE [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

    if params['fao_process'] == 'enabled':
        unutilised_pe = output['net_pefac'] - output['ae']
        unutilised_pe[unutilised_pe < 0] = 0
    else:
        unutilised_pe = np.zeros(len(series['date']))

    return {'unutilised_pe': unutilised_pe}


###############################################################################
def get_perc_through_root(data, node):
    """Z) Percolation Through the Root Zone [mm/d]."""
    params, output = data['params'], data['output']

    if params['fao_process'] == 'enabled':
        perc_through_root = np.copy(output['p_smd'])
        perc_through_root[perc_through_root > 0] = 0
        perc_through_root = - perc_through_root
    else:
        perc_through_root = np.copy(output['percol_in_root'])

    return {'perc_through_root': perc_through_root}


###############################################################################
def get_subroot_leak(data, node):
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
def get_interflow_bypass(data, node):
    """AB) Bypassing the Interflow Store [mm/d]."""
    params, output = data['params'], data['output']
    if params['interflow_process'] == 'enabled':
        coef = params['interflow_params'][node][1]
    else:
        coef = 1.0

    interflow_bypass = coef * (output['perc_through_root'] +
                               output['subroot_leak'])

    return {'interflow_bypass': interflow_bypass}


###############################################################################
def get_interflow_store_input(data, node):
    """AC) Input to Interflow Store [mm/d]."""
    output = data['output']

    interflow_store_input = (output['perc_through_root'] +
                             output['subroot_leak'] -
                             output['interflow_bypass'])

    return {'interflow_store_input': interflow_store_input}


###############################################################################
def get_interflow(data, node):
    """Multicolumn function.

    AD) Interflow Store Volume [mm]
    AE) Infiltration Recharge [mm/d]
    AF) Interflow to Surface Water Courses [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['interflow_volume', 'infiltration_recharge',
                'interflow_to_rivers']:
        col[key] = np.zeros(len(series['date']))

    var0 = params['interflow_params'][node][0]
    var5 = params['interflow_params'][node][2]
    var8 = params['interflow_params'][node][3]

    volume = var0
    col['interflow_volume'][0] = volume
    recharge = (var5 if volume >= var5 else volume)
    col['infiltration_recharge'][0] = recharge
    rivers = (volume - recharge) * var8
    col['interflow_to_rivers'][0] = rivers

    for num in range(1, len(series['date'])):
        if params['interflow_process'] == 'enabled':
            var1 = output['interflow_store_input'][num-1]
            volume = var1 + volume - recharge - rivers
            col['interflow_volume'][num] = volume
        else:
            volume = 0
        recharge = (var5 if volume >= var5 else volume)
        col['infiltration_recharge'][num] = recharge
        rivers = (volume - recharge) * var8
        col['interflow_to_rivers'][num] = rivers

    return col


###############################################################################
def get_recharge_store_input(data, node):
    """AG) Input to Recharge Store [mm/d]."""
    output = data['output']

    recharge_store_input = (output['infiltration_recharge'] +
                            output['interflow_bypass'] +
                            output['macropore'] +
                            output['runoff_recharge'])

    return {'recharge_store_input': recharge_store_input}


###############################################################################
def get_recharge(data, node):
    """Multicolumn function.

    AH) Recharge Store Volume [mm]
    AI) RCH: Combined Recharge [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['recharge_store', 'combined_recharge']:
        col[key] = np.zeros(len(series['date']))

    if params['recharge_attenuation_process'] == 'enabled':
        irs = params['recharge_attenuation_params'][node][0]
        rlp = params['recharge_attenuation_params'][node][1]
        rll = params['recharge_attenuation_params'][node][2]
        recharge = irs
        combined = recharge * rlp
        col['recharge_store'][0] = recharge
        col['combined_recharge'][0] = combined
        for num in range(1, len(series['date'])):
            recharge += output['recharge_store_input'][num-1] - combined
            col['recharge_store'][num] = recharge
            var4 = recharge * rlp
            combined = (rll if var4 > rll else var4)
            col['combined_recharge'][num] = combined

    return col


###############################################################################
def get_combined_str(data, node):
    """AJ) STR: Combined Surface Flow To Surface Water Courses [mm/d]."""
    output = data['output']

    combined_str = (output['interflow_to_rivers'] +
                    output['rapid_runoff'] -
                    output['runoff_recharge'])

    return {'combined_str': combined_str}


###############################################################################
def get_combined_ae(data, node):
    """AK) AE: Combined AE [mm/d]."""
    output = data['output']
    combined_ae = output['canopy_storage'] + output['ae']
    return {'combined_ae': combined_ae}


###############################################################################
def get_evt(data, node):
    """AL) EVT: Unitilised PE [mm/d]."""
    output = data['output']
    return {'evt': output['unutilised_pe']}


###############################################################################
def get_average_in(data, node):
    """AM) AVERAGE IN [mm]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    average_in = (series['rainfall_ts'][:, zone_rf] * coef_rf +
                  output['subroot_leak'])

    return {'average_in': average_in}


###############################################################################
def get_average_out(data, node):
    """AN) AVERAGE OUT [mm]."""
    output = data['output']

    average_out = (output['combined_str'] +
                   output['combined_recharge'] +
                   output['ae'] +
                   output['canopy_storage'])

    return {'average_out': average_out}


###############################################################################
def get_balance(data, node):
    """AO) BALANCE [mm]."""
    output = data['output']
    balance = output['average_in'] - output['average_out']
    return {'balance': balance}
