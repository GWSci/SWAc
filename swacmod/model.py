#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod model functions."""

# Standard Library
import math

# Internal modules
from . import utils as u


###############################################################################
def get_pefac(data, node):
    """E) Vegitation Factored PE (PEfac) [mm/d]."""
    series, params = data['series'], data['params']
    zone_pe = params['pe_zone_mapping'][node][0] - 1
    coef_pe = params['pe_zone_mapping'][node][1]
    zone_lu = params['lu_spatial'][node]

    pefac = []
    for num in range(len(series['date'])):
        var1 = series['date'][num].month
        var2 = series['pe_ts'][num][zone_pe] * coef_pe
        var3 = u.weighted_sum(params['kc'][var1], zone_lu)
        pefac.append(var2 * var3)

    return {'pefac': pefac}


###############################################################################
def get_canopy_storage(data, node):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]
    ftf = params['free_throughfall'][node]

    canopy_storage = []
    for num in range(len(series['date'])):
        var1 = series['rainfall_ts'][num][zone_rf] * coef_rf * (1 - ftf)
        var2 = params['max_canopy_storage'][node]
        var3 = output[node]['pefac'][num]
        var4 = (var2 if var1 > var2 else var1)
        canopy_storage.append(var3 if var4 > var3 else var4)

    return {'canopy_storage': canopy_storage}


###############################################################################
def get_veg_diff(data, node):
    """G) Vegitation Factored PE less Canopy Evaporation [mm/d]."""
    series, output = data['series'], data['output']

    veg_diff = []
    for num in range(len(series['date'])):
        var1 = output[node]['pefac'][num] - output[node]['canopy_storage'][num]
        veg_diff.append(var1)

    return {'veg_diff': veg_diff}


###############################################################################
def get_precipitation(data, node):
    """H) Precipitation at Groundlevel [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    precipitation = []
    for num in range(len(series['date'])):
        var1 = series['rainfall_ts'][num][zone_rf] * coef_rf
        var2 = var1 - output[node]['canopy_storage'][num]
        precipitation.append(var2)

    return {'precipitation': precipitation}


###############################################################################
def get_snowfall_o(data, node):
    """I) Snowfall [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_tm = params['temperature_zone_mapping'][node] - 1
    sfd = params['snow_params'][node][1]
    smd = params['snow_params'][node][2]

    snowfall_o = []
    for num in range(len(series['date'])):
        var1 = series['temperature_ts'][num][zone_tm] - sfd
        var2 = sfd - smd
        var3 = 1 - (math.exp(- var1 / var2))**2
        var4 = (0 if var3 > 0 else var3)
        var5 = (1 if -var4 > 1 else -var4) * output[node]['precipitation'][num]
        snowfall_o.append(var5)

    return {'snowfall_o': snowfall_o}


###############################################################################
def get_rainfall_o(data, node):
    """J) Precipitation as Rainfall [mm/d]."""
    series, output = data['series'], data['output']

    rainfall_o = []
    for num in range(len(series['date'])):
        var1 = output[node]['precipitation'][num]
        var2 = output[node]['snowfall_o'][num]
        rainfall_o.append(var1 - var2)

    return {'rainfall_o': rainfall_o}


###############################################################################
def get_snow(data, node):
    """"Multicolumn function.

    K) SnowPack [mm]
    L) SnowMelt [mm/d].
    """
    series, params, output = data['series'], data['params'], data['output']
    ssp = params['snow_params'][node][0]
    sfd = params['snow_params'][node][1]
    smd = params['snow_params'][node][2]
    zone_tm = params['temperature_zone_mapping'][node] - 1

    col = {}
    for key in ['snowpack', 'snowmelt']:
        col[key] = [0 for _ in series['date']]

    for num in range(len(series['date'])):
        var2 = (col['snowpack'][num-1] if num > 0 else ssp)
        var3 = series['temperature_ts'][num][zone_tm] - smd
        var4 = sfd - smd
        var5 = 1 - (math.exp(- var3 / var4))**2
        col['snowmelt'][num] = var2 * (0 if var5 < 0 else var5)
        col['snowpack'][num] = var2
        col['snowpack'][num] += output[node]['snowfall_o'][num]
        col['snowpack'][num] -= col['snowmelt'][num]

    return col


###############################################################################
def get_net_rainfall(data, node):
    """M) Net Rainfall and Snow Melt [mm/d]."""
    series, output = data['series'], data['output']

    net_rainfall = []
    for num in range(len(series['date'])):
        var1 = output[node]['snowmelt'][num] + output[node]['rainfall_o'][num]
        net_rainfall.append(var1)

    return {'net_rainfall': net_rainfall}


###############################################################################
def get_rawrew(data, node):
    """S) RAWREW."""
    series, params = data['series'], data['params']

    rawrew = []
    for num in range(len(series['date'])):
        var1 = series['date'][num].month - 1
        rawrew.append(params['RAW'][node][var1])

    return {'rawrew': rawrew}


###############################################################################
def get_tawrew(data, node):
    """T) TAWREW."""
    series, params = data['series'], data['params']

    tawrew = []
    for num in range(len(series['date'])):
        var1 = series['date'][num].month - 1
        tawrew.append(params['TAW'][node][var1])

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
    X) AE [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']
    order = ['rapid_runoff_c', 'rapid_runoff', 'runoff_recharge', 'macropore',
             'perc_in_root', 'p_smd', 'smd', 'k_s', 'ae']
    col = {}
    for key in order:
        col[key] = [0.0 for _ in series['date']]

    ssmd = u.weighted_sum(params['soil_spatial'][node],
                          params['soil_static_params']['starting_SMD'])

    zone_mac = params['macropore_zone_mapping'][node] - 1
    zone_ror = params['ror_zone_mapping'][node] - 1
    zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
    prop = params['recharge_proportion']
    class_smd = params['rapid_runoff_params'][zone_rro]['class_smd']
    class_ri = params['rapid_runoff_params'][zone_rro]['class_ri']
    values = params['rapid_runoff_params'][zone_rro]['values']

    for num in range(len(series['date'])):
        var0 = (col['p_smd'][num-1] if num > 0 else ssmd)
        var1 = (col['smd'][num-1] if num > 0 else ssmd)
        var2 = output[node]['net_rainfall'][num]
        cond1 = var1 > (class_smd[-1] - 1)
        cond2 = var2 > (class_ri[-1] - 1)
        if cond1 or cond2:
            col['rapid_runoff_c'][num] = values[-1][0]
        else:
            var3 = len([i for i in class_ri if i < var2])
            var4 = len([i for i in class_smd if i < var1])
            col['rapid_runoff_c'][num] = values[var3][var4]
        var5 = output[node]['net_rainfall'][num] * col['rapid_runoff_c'][num]
        col['rapid_runoff'][num] = (0 if var2 < 0 else var5)
        var6 = series['date'][num].month
        var7 = prop[var6][zone_ror] * col['rapid_runoff'][num]
        var8 = params['recharge_limit'][var6][zone_ror]
        col['runoff_recharge'][num] = (var8 if var7 > var8 else var7)
        var8a = var2 - col['rapid_runoff'][num]
        var9 = params['macropore_proportion'][var6][zone_mac] * var8a
        var10 = params['macropore_limit'][var6][zone_mac]
        col['macropore'][num] = (var10 if var9 > var10 else var9)
        col['perc_in_root'][num] = var2
        col['perc_in_root'][num] -= col['rapid_runoff'][num]
        col['perc_in_root'][num] -= col['macropore'][num]
        col['smd'][num] = (var0 if var0 > 0 else 0.0)
        if col['perc_in_root'][num] > output[node]['veg_diff'][num]:
            var11 = -1
        else:
            var12 = (output[node]['tawrew'][num] - col['smd'][num]) / \
                    (output[node]['tawrew'][num] - output[node]['rawrew'][num])
            if var12 >= 1:
                var11 = 1
            else:
                var11 = (var12 if var12 >= 0 else 0.0)
        col['k_s'][num] = var11
        cond3 = col['smd'][num] < output[node]['rawrew'][num]
        cond4 = col['perc_in_root'][num] > output[node]['veg_diff'][num]
        if cond3 or cond4:
            var13 = output[node]['veg_diff'][num]
        else:
            cond5 = col['smd'][num] >= output[node]['rawrew'][num]
            cond6 = col['smd'][num] <= output[node]['tawrew'][num]
            if cond5 and cond6:
                var14 = (output[node]['veg_diff'][num] -
                         col['perc_in_root'][num])
                var13 = col['perc_in_root'][num] + col['k_s'][num] * var14
            else:
                var13 = col['perc_in_root'][num]
        col['ae'][num] = var13
        var14 = col['smd'][num] + col['ae'][num] - col['perc_in_root'][num]
        col['p_smd'][num] = var14

    return col


###############################################################################
def get_unutilized_pe(data, node):
    """Y) Unutilised PE [mm/d]."""
    series, output = data['series'], data['output']

    unutilized_pe = []
    for num in range(len(series['date'])):
        var1 = output[node]['veg_diff'][num] - output[node]['ae'][num]
        var2 = (0 if var1 < 0 else var1)
        unutilized_pe.append(var2)

    return {'unutilized_pe': unutilized_pe}


###############################################################################
def get_perc_through_root(data, node):
    """Z) Percolation Through the Root Zone [mm/d]."""
    series, output = data['series'], data['output']

    perc_through_root = []
    for num in range(len(series['date'])):
        var1 = output[node]['p_smd'][num]
        var2 = (-var1 if var1 < 0 else 0)
        perc_through_root.append(var2)

    return {'perc_through_root': perc_through_root}


###############################################################################
def get_subroot_leak(data, node):
    """AA) Sub Root Zone Leakege / Inputs [mm/d]."""
    series, params = data['series'], data['params']
    zone_sr = params['subroot_zone_mapping'][node][0] - 1
    coef_sr = params['subroot_zone_mapping'][node][1]

    subroot_leak = []
    for num in range(len(series['date'])):
        var1 = series['subroot_leakage_ts'][num][zone_sr] * coef_sr
        var2 = var1 * params['leakage'][node]
        subroot_leak.append(var2)

    return {'subroot_leak': subroot_leak}


###############################################################################
def get_interflow_bypass(data, node):
    """AB) Bypassing the Interflow Store [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

    interflow_bypass = []
    for num in range(len(series['date'])):
        var1 = output[node]['perc_through_root'][num]
        var1 += output[node]['subroot_leak'][num]
        var2 = params['interflow_params'][node][1] * var1
        interflow_bypass.append(var2)

    return {'interflow_bypass': interflow_bypass}


###############################################################################
def get_interflow_input(data, node):
    """AC) Input to Interflow Store [mm/d]."""
    series, output = data['series'], data['output']

    interflow_input = []
    for num in range(len(series['date'])):
        var1 = output[node]['perc_through_root'][num]
        var1 += output[node]['subroot_leak'][num]
        var2 = var1 - output[node]['interflow_bypass'][num]
        interflow_input.append(var2)

    return {'interflow_input': interflow_input}


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
        col[key] = [0.0 for _ in series['date']]

    var0 = params['interflow_params'][node][0]
    var5 = params['interflow_params'][node][2]
    var8 = params['interflow_params'][node][3]

    for num in range(len(series['date'])):
        var1 = (output[node]['interflow_input'][num-1] if num > 0 else 0.0)
        var2 = (col['interflow_volume'][num-1] if num > 0 else var0)
        var3 = (col['infiltration_recharge'][num-1] if num > 0 else 0.0)
        var4 = (col['interflow_to_rivers'][num-1] if num > 0 else 0.0)
        col['interflow_volume'][num] = var1 + var2 - var3 - var4
        var6 = col['interflow_volume'][num]
        col['infiltration_recharge'][num] = (var5 if var6 >= var5 else var6)
        var7 = col['interflow_volume'][num] - col['infiltration_recharge'][num]
        col['interflow_to_rivers'][num] = var7 * var8

    return col


###############################################################################
def get_recharge_input(data, node):
    """AG) Input to Recharge Store [mm/d]."""
    series, output = data['series'], data['output']

    recharge_input = []
    for num in range(len(series['date'])):
        var1 = output[node]['infiltration_recharge'][num]
        var1 += output[node]['interflow_bypass'][num]
        var1 += output[node]['macropore'][num]
        var1 += output[node]['runoff_recharge'][num]
        recharge_input.append(var1)

    return {'recharge_input': recharge_input}


###############################################################################
def get_recharge(data, node):
    """Multicolumn function.

    AH) Recharge Store Volume [mm]
    AI) RCH: Combined Recharge [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['recharge_store', 'combined_recharge']:
        col[key] = [0.0 for _ in series['date']]

    irs = params['release_params'][node][0]
    rlp = params['release_params'][node][1]
    rll = params['release_params'][node][2]

    for num in range(len(series['date'])):
        var1 = (col['recharge_store'][num-1] if num > 0 else irs)
        var2 = (output[node]['recharge_input'][num-1] if num > 0 else 0.0)
        var3 = (col['combined_recharge'][num-1] if num > 0 else 0.0)
        col['recharge_store'][num] = var1 + var2 - var3
        var4 = col['recharge_store'][num] * rlp
        col['combined_recharge'][num] = (rll if var4 > rll else var4)

    return col


###############################################################################
def get_str(data, node):
    """AJ) STR: Combined Surface Flow To Surface Water Courses [mm/d]."""
    series, output = data['series'], data['output']

    out_str = []
    for num in range(len(series['date'])):
        var1 = output[node]['interflow_to_rivers'][num]
        var1 += output[node]['rapid_runoff'][num]
        var1 -= output[node]['runoff_recharge'][num]
        out_str.append(var1)

    return {'str': out_str}


###############################################################################
def get_combined_ae(data, node):
    """AK) AE: Combined AE [mm/d]."""
    series, output = data['series'], data['output']

    combined_ae = []
    for num in range(len(series['date'])):
        var1 = output[node]['canopy_storage'][num] + output[node]['ae'][num]
        combined_ae.append(var1)

    return {'combined_ae': combined_ae}


###############################################################################
def get_evt(data, node):
    """AL) EVT: Unitilised PE [mm/d]."""
    output = data['output']

    return {'evt': output[node]['unutilized_pe']}


###############################################################################
def get_average_in(data, node):
    """AM) AVERAGE IN [mm]."""
    series, params, output = data['series'], data['params'], data['output']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]

    average_in = []
    for num in range(len(series['date'])):
        var1 = series['rainfall_ts'][num][zone_rf] * coef_rf
        var1 += output[node]['subroot_leak'][num]
        average_in.append(var1)

    return {'average_in': average_in}


###############################################################################
def get_average_out(data, node):
    """AN) AVERAGE OUT [mm]."""
    series, output = data['series'], data['output']

    average_out = []
    for num in range(len(series['date'])):
        var1 = output[node]['str'][num]
        var1 += output[node]['combined_recharge'][num]
        var1 += output[node]['ae'][num]
        var1 += output[node]['canopy_storage'][num]
        average_out.append(var1)
    return {'average_out': average_out}


###############################################################################
def get_balance(data, node):
    """AO) BALANCE [mm]."""
    series, output = data['series'], data['output']

    balance = []
    for num in range(len(series['date'])):
        var1 = output[node]['average_in'][num]
        var1 -= output[node]['average_out'][num]
        balance.append(var1)

    return {'balance': balance}
