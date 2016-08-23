#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod model functions."""

# Standard Library
import math


###############################################################################
def get_pefac(data):
    """E) Vegitation Factored PE (PEfac) [mm/d]."""
    series, params = data['series'], data['params']

    pefac = []
    for num in range(len(series['rainfall'])):
        var1 = series['date'][num].month - 1
        pefac.append(params['KC'][var1] * series['PE'][num])

    return {'pefac': pefac}


###############################################################################
def get_canopy_storage(data):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

    canopy_storage = []
    for num in range(len(series['rainfall'])):
        var1 = series['rainfall'][num] * (1 - params['free_throughfall'])
        var2 = params['max_canopy_storage']
        var3 = output['pefac'][num]
        var4 = (var2 if var1 > var2 else var1)
        canopy_storage.append(var3 if var4 > var3 else var4)

    return {'canopy_storage': canopy_storage}


###############################################################################
def get_veg_diff(data):
    """G) Vegitation Factored PE less Canopy Evaporation [mm/d]."""
    series, output = data['series'], data['output']

    veg_diff = []
    for num in range(len(series['rainfall'])):
        var1 = output['pefac'][num] - output['canopy_storage'][num]
        veg_diff.append(var1)

    return {'veg_diff': veg_diff}


###############################################################################
def get_precipitation(data):
    """H) Precipitation at Groundlevel [mm/d]."""
    series, output = data['series'], data['output']

    precipitation = []
    for num in range(len(series['rainfall'])):
        var1 = series['rainfall'][num] - output['canopy_storage'][num]
        precipitation.append(var1)

    return {'precipitation': precipitation}


###############################################################################
def get_snowfall_o(data):
    """I) Snowfall [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

    snowfall_o = []
    for num in range(len(series['rainfall'])):
        var1 = series['T'][num] - params['snowfall_degrees']
        var2 = params['snowfall_degrees'] - params['snowmelt_degrees']
        var3 = 1 - (math.exp(- var1 / var2))**2
        var4 = (0 if var3 > 0 else var3)
        var5 = (1 if -var4 > 1 else -var4) * output['precipitation'][num]
        snowfall_o.append(var5)

    return {'snowfall_o': snowfall_o}


###############################################################################
def get_rainfall_o(data):
    """J) Precipitation as Rainfall [mm/d]."""
    series, output = data['series'], data['output']

    rainfall_o = []
    for num in range(len(series['rainfall'])):
        var1 = output['precipitation'][num] - output['snowfall_o'][num]
        rainfall_o.append(var1)

    return {'rainfall_o': rainfall_o}


###############################################################################
def get_snow(data):
    """"Multicolumn function.

    K) SnowPack [mm]
    L) SnowMelt [mm/d].
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['snowpack', 'snowmelt']:
        col[key] = [0 for _ in series['rainfall']]

    for num in range(len(series['rainfall'])):
        var1 = params['starting_snow_pack']
        var2 = (col['snowpack'][num-1] if num > 0 else var1)
        var3 = series['T'][num] - params['snowmelt_degrees']
        var4 = params['snowfall_degrees'] - params['snowmelt_degrees']
        var5 = 1 - (math.exp(- var3 / var4))**2
        col['snowmelt'][num] = var2 * (0 if var5 < 0 else var5)
        col['snowpack'][num] = var2
        col['snowpack'][num] += output['snowfall_o'][num]
        col['snowpack'][num] -= col['snowmelt'][num]

    return col


###############################################################################
def get_net_rainfall(data):
    """M) Net Rainfall and Snow Melt [mm/d]."""
    series, output = data['series'], data['output']

    net_rainfall = []
    for num in range(len(series['rainfall'])):
        var1 = output['snowmelt'][num] + output['rainfall_o'][num]
        net_rainfall.append(var1)

    return {'net_rainfall': net_rainfall}


###############################################################################
def get_rawrew(data):
    """S) RAWREW."""
    series, params = data['series'], data['params']

    rawrew = []
    for num in range(len(series['rainfall'])):
        var1 = series['date'][num].month - 1
        rawrew.append(params['RAW'][var1])

    return {'rawrew': rawrew}


###############################################################################
def get_tawrew(data):
    """T) TAWREW."""
    series, params = data['series'], data['params']

    tawrew = []
    for num in range(len(series['rainfall'])):
        var1 = series['date'][num].month - 1
        tawrew.append(params['TAW'][var1])

    return {'tawrew': tawrew}


###############################################################################
def get_ae(data):
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
        col[key] = [0.0 for _ in series['rainfall']]

    for num in range(len(series['rainfall'])):
        var0 = (col['p_smd'][num-1] if num > 0 else params['starting_SMD'])
        var1 = (col['smd'][num-1] if num > 0 else params['starting_SMD'])
        var2 = output['net_rainfall'][num]
        class_smd = params['rainfall_to_runoff']['class_smd']
        class_ri = params['rainfall_to_runoff']['class_ri']
        values = params['rainfall_to_runoff']['values']
        cond1 = var1 > (class_smd[-1] - 1)
        cond2 = var2 > (class_ri[-1] - 1)
        if cond1 or cond2:
            col['rapid_runoff_c'][num] = values[-1][0]
        else:
            var3 = len([i for i in class_ri if i < var2])
            var4 = len([i for i in class_smd if i < var1])
            col['rapid_runoff_c'][num] = values[var3][var4]
        var5 = output['net_rainfall'][num] * col['rapid_runoff_c'][num]
        col['rapid_runoff'][num] = (0 if var2 < 0 else var5)
        var6 = series['date'][num].month - 1
        var7 = params['recharge_proportion'][var6] * col['rapid_runoff'][num]
        var8 = params['recharge_limit'][var6]
        col['runoff_recharge'][num] = (var8 if var7 > var8 else var7)
        var8a = var2 - col['rapid_runoff'][num]
        var9 = params['macropore_proportion'][var6] * var8a
        var10 = params['macropore_limit'][var6]
        col['macropore'][num] = (var10 if var9 > var10 else var9)
        col['perc_in_root'][num] = var2
        col['perc_in_root'][num] -= col['rapid_runoff'][num]
        col['perc_in_root'][num] -= col['macropore'][num]
        col['smd'][num] = (var0 if var0 > 0 else 0.0)
        if col['perc_in_root'][num] > output['veg_diff'][num]:
            var11 = -1
        else:
            var12 = (output['tawrew'][num] - col['smd'][num]) / \
                    (output['tawrew'][num] - output['rawrew'][num])
            if var12 >= 1:
                var11 = 1
            else:
                var11 = (var12 if var12 >= 0 else 0.0)
        col['k_s'][num] = var11
        cond3 = col['smd'][num] < output['rawrew'][num]
        cond4 = col['perc_in_root'][num] > output['veg_diff'][num]
        if cond3 or cond4:
            var13 = output['veg_diff'][num]
        else:
            cond5 = col['smd'][num] >= output['rawrew'][num]
            cond6 = col['smd'][num] <= output['tawrew'][num]
            if cond5 and cond6:
                var14 = (output['veg_diff'][num] - col['perc_in_root'][num])
                var13 = col['perc_in_root'][num] + col['k_s'][num] * var14
            else:
                var13 = col['perc_in_root'][num]
        col['ae'][num] = var13
        var14 = col['smd'][num] + col['ae'][num] - col['perc_in_root'][num]
        col['p_smd'][num] = var14

    return col


###############################################################################
def get_unutilized_pe(data):
    """Y) Unutilised PE [mm/d]."""
    series, output = data['series'], data['output']

    unutilized_pe = []
    for num in range(len(series['rainfall'])):
        var1 = output['veg_diff'][num] - output['ae'][num]
        var2 = (0 if var1 < 0 else var1)
        unutilized_pe.append(var2)

    return {'unutilized_pe': unutilized_pe}


###############################################################################
def get_perc_through_root(data):
    """Z) Percolation Through the Root Zone [mm/d]."""
    series, output = data['series'], data['output']

    perc_through_root = []
    for num in range(len(series['rainfall'])):
        var1 = output['p_smd'][num]
        var2 = (-var1 if var1 < 0 else 0)
        perc_through_root.append(var2)

    return {'perc_through_root': perc_through_root}


###############################################################################
def get_subroot_leak(data):
    """AA) Sub Root Zone Leakege / Inputs [mm/d]."""
    series, params = data['series'], data['params']

    subroot_leak = []
    for num in range(len(series['rainfall'])):
        var1 = series['SZL'][num] * params['leakage']
        subroot_leak.append(var1)

    return {'subroot_leak': subroot_leak}


###############################################################################
def get_interflow_bypass(data):
    """AB) Bypassing the Interflow Store [mm/d]."""
    series, params, output = data['series'], data['params'], data['output']

    interflow_bypass = []
    for num in range(len(series['rainfall'])):
        var1 = output['perc_through_root'][num] + output['subroot_leak'][num]
        var2 = params['store_bypass'] * var1
        interflow_bypass.append(var2)

    return {'interflow_bypass': interflow_bypass}


###############################################################################
def get_interflow_input(data):
    """AC) Input to Interflow Store [mm/d]."""
    series, output = data['series'], data['output']

    interflow_input = []
    for num in range(len(series['rainfall'])):
        var1 = output['perc_through_root'][num] + output['subroot_leak'][num]
        var2 = var1 - output['interflow_bypass'][num]
        interflow_input.append(var2)

    return {'interflow_input': interflow_input}


###############################################################################
def get_interflow(data):
    """Multicolumn function.

    AD) Interflow Store Volume [mm]
    AE) Infiltration Recharge [mm/d]
    AF) Interflow to Surface Water Courses [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['interflow_volume', 'infiltration_recharge',
                'interflow_to_rivers']:
        col[key] = [0.0 for _ in series['rainfall']]

    for num in range(len(series['rainfall'])):
        var0 = params['init_interflow_store']
        var1 = (output['interflow_input'][num-1] if num > 0 else 0.0)
        var2 = (col['interflow_volume'][num-1] if num > 0 else var0)
        var3 = (col['infiltration_recharge'][num-1] if num > 0 else 0.0)
        var4 = (col['interflow_to_rivers'][num-1] if num > 0 else 0.0)
        col['interflow_volume'][num] = var1 + var2 - var3 - var4
        var5 = params['infiltration']
        var6 = col['interflow_volume'][num]
        col['infiltration_recharge'][num] = (var5 if var6 >= var5 else var6)
        var7 = col['interflow_volume'][num] - col['infiltration_recharge'][num]
        var8 = params['interflow_to_rivers']
        col['interflow_to_rivers'][num] = var7 * var8

    return col


###############################################################################
def get_recharge_input(data):
    """AG) Input to Recharge Store [mm/d]."""
    series, output = data['series'], data['output']

    recharge_input = []
    for num in range(len(series['rainfall'])):
        var1 = output['infiltration_recharge'][num]
        var1 += output['interflow_bypass'][num]
        var1 += output['macropore'][num]
        var1 += output['runoff_recharge'][num]
        recharge_input.append(var1)

    return {'recharge_input': recharge_input}


###############################################################################
def get_recharge(data):
    """Multicolumn function.

    AH) Recharge Store Volume [mm]
    AI) RCH: Combined Recharge [mm/d]
    """
    series, params, output = data['series'], data['params'], data['output']

    col = {}
    for key in ['recharge_store', 'combined_recharge']:
        col[key] = [0.0 for _ in series['rainfall']]

    for num in range(len(series['rainfall'])):
        var0 = params['init_recharge_store']
        var1 = (col['recharge_store'][num-1] if num > 0 else var0)
        var2 = (output['recharge_input'][num-1] if num > 0 else 0.0)
        var3 = (col['combined_recharge'][num-1] if num > 0 else 0.0)
        col['recharge_store'][num] = var1 + var2 - var3
        var4 = col['recharge_store'][num]
        var5 = params['release_proportion'] * var4
        var6 = params['release_limit']
        col['combined_recharge'][num] = (var6 if var5 > var6 else var5)

    return col


###############################################################################
def get_str(data):
    """AJ) STR: Combined Surface Flow To Surface Water Courses [mm/d]."""
    series, output = data['series'], data['output']

    out_str = []
    for num in range(len(series['rainfall'])):
        var1 = output['interflow_to_rivers'][num]
        var1 += output['rapid_runoff'][num]
        var1 -= output['runoff_recharge'][num]
        out_str.append(var1)

    return {'str': out_str}


###############################################################################
def get_combined_ae(data):
    """AK) AE: Combined AE [mm/d]."""
    series, output = data['series'], data['output']

    combined_ae = []
    for num in range(len(series['rainfall'])):
        var1 = output['canopy_storage'][num] + output['ae'][num]
        combined_ae.append(var1)

    return {'combined_ae': combined_ae}


###############################################################################
def get_evt(data):
    """AL) EVT: Unitilised PE [mm/d]."""
    output = data['output']

    return {'evt': output['unutilized_pe']}


###############################################################################
def get_average_in(data):
    """AM) AVERAGE IN [mm]."""
    series, output = data['series'], data['output']

    average_in = []
    for num in range(len(series['rainfall'])):
        var1 = series['rainfall'][num] + output['subroot_leak'][num]
        average_in.append(var1)

    return {'average_in': average_in}


###############################################################################
def get_average_out(data):
    """AN) AVERAGE OUT [mm]."""
    series, output = data['series'], data['output']

    average_out = []
    for num in range(len(series['rainfall'])):
        var1 = output['str'][num] + output['combined_recharge'][num]
        var1 += output['ae'][num] + output['canopy_storage'][num]
        average_out.append(var1)
    return {'average_out': average_out}


###############################################################################
def get_balance(data):
    """AO) BALANCE [mm]."""
    series, output = data['series'], data['output']

    balance = []
    for num in range(len(series['rainfall'])):
        var1 = output['average_in'][num] - output['average_out'][num]
        balance.append(var1)

    return {'balance': balance}
