# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=True, wraparound=True


# Third Party Libraries
import numpy as np
from collections import OrderedDict

# Internal modules
from . import utils as u
from tqdm import tqdm
import networkx as nx
import sys
import cython
cimport cython
cimport numpy as np
import swacmod.feature_flags as ff

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(u.__file__), ".."))
from swacmod.snow_melt import SnowMelt
import swacmod.timer as timer
import swacmod.flopy_adaptor as flopy_adaptor


###############################################################################

def get_precipitation(data, output, node):
    """C) Precipitation [mm/d]."""
    series, params = data['series'], data['params']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]
    rainfall_ts = series['rainfall_ts'][:, zone_rf] * coef_rf
    return {'rainfall_ts': rainfall_ts}

###############################################################################

def get_precipitation_r(data, output, node):
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


@cython.boundscheck(False)
@cython.wraparound(False)
def get_pefac(data, output, node):
    """E) Vegetation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
    series, params = data['series'], data['params']
    cdef int days = len(series['date'])
    cdef int day, z
    cdef double[:] pefac = np.zeros(days)
    cdef double var1 = 0.0
    cdef double[:] pe = output['pe_ts']
    cdef double[:, :] kc = params['kc_list'][series['months']]
    cdef double[:] zone_lu = np.array(params['lu_spatial'][node],
                                      dtype=np.float64)
    cdef int len_lu = len(params['lu_spatial'][node])

    fao = params['fao_process']
    canopy = params['canopy_process']

    if fao == 'enabled' or canopy == 'enabled':
        for day in range(days):
            var1 = 0.0
            for z in range(len_lu):
                var1 = var1 + (kc[day, z] * zone_lu[z])
            pefac[day] = pe[day] * var1

    return {'pefac': np.array(pefac)}

###############################################################################


def get_canopy_storage(data, output, node):
    """F) Canopy Storage and PEfac Limited Interception [mm/d]."""
    series, params = data['series'], data['params']

    if params['canopy_process'] == 'enabled':
        canopy_zone = params['canopy_zone_mapping'][node]
        ftf = params['free_throughfall'][canopy_zone]
        mcs = params['max_canopy_storage'][canopy_zone]
        canopy_storage = output['rainfall_ts'] * (1 - ftf)
        canopy_storage[canopy_storage > mcs] = mcs
        canopy_storage = np.where(canopy_storage > output['pefac'],
                                  output['pefac'], canopy_storage)
        if ff.use_natproc:
            canopy_storage = np.where(output['pefac'] < 0.0,
                                    0.0, canopy_storage)
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
    precip_to_ground = output['rainfall_ts'] - output['canopy_storage']
    return {'precip_to_ground': precip_to_ground}

###############################################################################

def get_snowfall_o(data, output, node):
    """I) Snowfall [mm/d]."""
    series, params = data['series'], data['params']

    if params['snow_process_simple'] == 'enabled':
        zone_tm = params['temperature_zone_mapping'][node] - 1
        snow_fall_temp = params['snow_params_simple'][node][1]
        snow_melt_temp = params['snow_params_simple'][node][2]
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

def get_snow_simple(data, output, node):
    """"Multicolumn function.

    K) SnowPack [mm]
    L) SnowMelt [mm/d].
    """
    series, params = data['series'], data['params']

    cdef:
        size_t num
        size_t length = len(series['date'])
        double[:] col_snowpack = np.zeros(length)
        double[:] col_snowmelt = np.zeros(len(series['date']))

    if params['snow_process_simple'] == 'disabled':
        col = {}
        col['snowpack'] = col_snowpack.base
        col['snowmelt'] = col_snowmelt.base
        return col

    cdef:
        double start_snow_pack = params['snow_params_simple'][node][0]
        double snow_fall_temp = params['snow_params_simple'][node][1]
        double snow_melt_temp = params['snow_params_simple'][node][2]
        double diff = snow_fall_temp - snow_melt_temp
        size_t zone_tm = params['temperature_zone_mapping'][node] - 1
        double[:] var3 = (snow_melt_temp -
                          series['temperature_ts'][:, zone_tm])/diff
        double[:] var5 = 1 - (np.exp(var3))**2
        double[:] snowfall_o = output['snowfall_o']
        double var6 = (var5[0] if var5[0] > 0 else 0)
        double snowpack = (1 - var6) * start_snow_pack + snowfall_o[0]

    col_snowmelt[0] = start_snow_pack * var6
    col_snowpack[0] = snowpack
    for num in range(1, length):
        if var5[num] < 0:
            var5[num] = 0
        col_snowmelt[num] = snowpack * var5[num]
        snowpack = (1 - var5[num]) * snowpack + snowfall_o[num]
        col_snowpack[num] = snowpack

    return {'snowpack': col_snowpack.base, 'snowmelt': col_snowmelt.base}


##############################################################################

def get_snow_complex(data, output, node):
    """"
    Call snowmelt fn of snowmelt for this node
    """
    series, params = data['series'], data['params']

    cdef:
        size_t day
        size_t days = len(series['date'])
        double[:] col_snowpack = np.zeros(days)
        double[:] col_snowmelt = np.zeros(days)
        double[:] col_snowfall_o = np.zeros(days)
        double[:] col_rainfall_o = np.zeros(days)
        double[:] rainfall_ts = output['rainfall_ts']
        size_t zone_tmax_c = params['tmax_c_zone_mapping'][node] - 1
        size_t zone_tmin_c = params['tmin_c_zone_mapping'][node] - 1
        size_t zone_windsp = params['windsp_zone_mapping'][node] - 1

        double[:] tmax_c = series['tmax_c_ts'][:, zone_tmax_c]
        double[:] tmin_c = series['tmin_c_ts'][:, zone_tmin_c]
        double[:] windsp = series['windsp_ts'][:, zone_windsp]

        double lat_deg = params['snow_params_complex'][node][0]
        double slope = params['snow_params_complex'][node][1]
        double aspect = params['snow_params_complex'][node][2]
        double tempht = params['snow_params_complex'][node][3]
        double windht = params['snow_params_complex'][node][4]
        double groundalbedo = params['snow_params_complex'][node][5]
        double surfemissiv = params['snow_params_complex'][node][6]
        double forest = params['snow_params_complex'][node][7]
        double startingsnowdepth_m = params['snow_params_complex'][node][8]
        double startingsnowdensity_kg_m3 = params['snow_params_complex'][node][9]

    if params['snow_process_complex'] == 'disabled':
        col = {}
        col['snowfall_o'] = output['snowfall_o']
        col['rainfall_o'] = output['rainfall_o']

        if params['snow_process_simple'] == 'disabled':
            col['snowpack'] = col_snowpack.base
            col['snowmelt'] = col_snowmelt.base
        else:
            col['snowpack'] = output['snowpack']
            col['snowmelt'] = output['snowmelt']

        return col

    else:

        sm = SnowMelt()

        big_list = sm.SnowMelt(np.array([d.strftime("%Y-%m-%d") for d in series['date']]),
                               np.asarray(rainfall_ts),
                               np.asarray(tmax_c), np.asarray(tmin_c),
                               lat_deg, slope,
                               aspect, tempht, windht, groundalbedo,
                               surfemissiv, np.asarray(windsp), forest,
                               startingsnowdepth_m,
                               startingsnowdensity_kg_m3)

        for day in range(1, days):
            col_snowmelt[day] = big_list[6][day]
            col_snowpack[day] = big_list[8][day]
            col_snowfall_o[day] = big_list[7][day]
            col_rainfall_o[day] = big_list[3][day]

        del sm

        return {'snowpack': col_snowpack.base,
                'snowmelt': col_snowmelt.base,
                'snowfall_o': col_snowfall_o.base,
                'rainfall_o': col_rainfall_o.base}

##############################################################################

def get_net_rainfall(data, output, node):
    """M) Net Rainfall and Snow Melt [mm/d]."""
    net_rainfall = output['snowmelt'] + output['rainfall_o']
    return {'net_rainfall': net_rainfall}

##############################################################################

def get_rawrew(data, output, node):
    """S) RAWREW (Readily Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']
    if params['fao_process'] == 'enabled':
        rawrew = params['raw'][node][series['months']]
    else:
        rawrew = np.zeros(len(series['date']))
    return {'rawrew': rawrew}

##############################################################################

def get_tawtew(data, output, node):
    """T) TAWTEW (Total Available Water, Readily Evaporable Water)."""
    series, params = data['series'], data['params']

    if params['fao_process'] == 'enabled':
        tawtew = params['taw'][node][series['months']]
    else:
        tawtew = np.zeros(len(series['date']))

    return {'tawtew': tawtew}

##############################################################################

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

    cdef:
        double[:] col_rapid_runoff_c = np.zeros(len(series['date']))
        double[:] col_rapid_runoff = np.zeros(len(series['date']))
        double[:] col_runoff_recharge = np.zeros(len(series['date']))
        double[:] col_macropore_att = np.zeros(len(series['date']))
        double[:] col_macropore_dir = np.zeros(len(series['date']))
        double[:] col_percol_in_root = np.zeros(len(series['date']))
        double[:] col_p_smd = np.zeros(len(series['date']))
        double[:] col_smd = np.zeros(len(series['date']))
        double[:] col_k_slope = np.zeros(len(series['date']))
        double[:] col_ae = np.zeros(len(series['date']))
        size_t zone_mac = params['macropore_zone_mapping'][node] - 1
        size_t zone_ror = params['single_cell_swrecharge_zone_mapping'][node] - 1
        size_t zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
        double ssmd = u.weighted_sum(params['soil_spatial'][node],
                                     s_smd['starting_SMD'])
        long long[:] class_smd = np.array(rrp[zone_rro]['class_smd'],
                                          dtype=np.int64)
        long long[:] class_ri = np.array(rrp[zone_rro]['class_ri'],
                                         dtype=np.int64)
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
        double [:, :] ror_act = params['ror_act']
        double[:, :] macro_prop = params['macro_prop']
        double[:, :] macro_limit = params['macro_limit']
        double[:, :] macro_act = params['macro_act']
        double[:, :] macro_rec = params['macro_rec']
        double[:, :] values = np.array(rrp[zone_rro]['values'])
        size_t len_class_smd = len(class_smd)
        size_t len_class_ri = len(class_ri)
        double last_smd = class_smd[-1] - 1
        double last_ri = class_ri[-1] - 1
        double value = values[-1][0]
        double p_smd = ssmd
        double smd = ssmd
        double var2, var5, var8a, var9, var10, var11, var12, var13
        double rapid_runoff_c, rapid_runoff, macropore, percol_in_root
        double net_pefac, tawtew, rawrew
        size_t num, i, var3, var4, var6
        size_t length = len(series['date'])
        double[:] net_rainfall = output['net_rainfall']
        double[:] net_pefac_a = output['net_pefac']
        double[:] tawtew_a = output['tawtew']
        double[:] rawrew_a = output['rawrew']
        long long[:] months = np.array(series['months'], dtype=np.int64)
        double ma = 0.0
        size_t zone_sw
        double[:] sw_ponding_area = params['sw_pond_area']
        double pond_area, not_ponded

    if params['sw_process_natproc'] == 'enabled':
        zone_sw = params['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    not_ponded = 1.0 - pond_area

    if (params['swrecharge_process'] == 'enabled' or
        params['single_cell_swrecharge_process'] == 'enabled'):
        col_runoff_recharge[:] = 0.0

    for num in range(length):
        var2 = net_rainfall[num]
        var6 = months[num]
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
            rapid_runoff = (0.0 if var2 < 0.0 else var5)
            col_rapid_runoff[num] = rapid_runoff

            if params['single_cell_swrecharge_process'] == 'enabled':
                var6a = rapid_runoff - ror_act[var6][zone_ror]
                if var6a > 0:
                    var7 = ror_prop[var6][zone_ror] * var6a
                    var8 = ror_limit[var6][zone_ror]
                    col_runoff_recharge[num] = (var8 if var7 > var8 else var7)
                else:
                    col_runoff_recharge[num] = 0.0

        if params['macropore_process'] == 'enabled':
            if mac_opt == 'SMD':
                var8a = var2 - col_rapid_runoff[num]
                ma = macro_act[var6][zone_mac]
            else:
                var8a = (var2 - col_rapid_runoff[num]
                         - macro_act[var6][zone_mac])
                ma = sys.float_info.max
            if var8a > 0.0:
                if p_smd < ma:
                    var9 = macro_prop[var6][zone_mac] * var8a
                    var10 = macro_limit[var6][zone_mac]
                    macropore = min(var10, var9)
                else:
                    macropore = 0.0
            else:
                macropore = 0.0

            var10a = macro_rec[var6][zone_mac]
            col_macropore_att[num] = macropore * (1 - var10a)
            col_macropore_dir[num] = macropore * var10a

        percol_in_root = (var2 - col_rapid_runoff[num]
                          - col_macropore_att[num]
                          - col_macropore_dir[num])
        col_percol_in_root[num] = percol_in_root

        if params['fao_process'] == 'enabled':
            smd = max(p_smd, 0.0)
            col_smd[num] = smd
            net_pefac = net_pefac_a[num]
            tawtew = tawtew_a[num]
            rawrew = rawrew_a[num]

            if percol_in_root > net_pefac:
                var11 = -1.0
            else:
                # tmp div zero
                if (tawtew - rawrew) == 0.0:
                    var12 = 1.0
                else:
                    var12 = (tawtew - smd) / (tawtew - rawrew)

                if var12 >= 1.0:
                    var11 = 1.0
                else:
                    var11 = max(var12, 0.0)
            col_k_slope[num] = var11

            var13 = percol_in_root
            if smd < rawrew or percol_in_root > net_pefac:
                var13 = net_pefac
            elif smd >= rawrew and smd <= tawtew:
                var13 = var11 * (net_pefac - percol_in_root)
            else:
                var13 = 0.0
            var13 *= not_ponded
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

        if params["percolation_rejection_use_timeseries"]:
            rej_ts = series['percolation_rejection_ts']
            rej_array = np.zeros(len(series['date']))
            for iday, rts in enumerate(rej_ts):
                rej_array[iday] = (np.array(rts) * zone).sum(axis=0)
            rej = rej_array
            # not sure if this correct
            rejected_recharge[perc > rej] = perc[perc > rej] - rej[perc > rej]
        else:
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
        interflow_zone = params['interflow_zone_mapping'][node]
        coef = params['interflow_store_bypass'][interflow_zone]
    else:
        coef = 1.0

    interflow_bypass = coef * (output['perc_through_root'] +
                               output['subroot_leak'])

    return {'interflow_bypass': interflow_bypass}

###############################################################################


def get_interflow_store_input(data, output, node):
    """AE) Input to Interflow Store [mm/d]."""

    cdef:
        double[:] sw_ponding_area = data['params']['sw_pond_area']
        double pond_area, not_ponded
        size_t zone_sw

    if data['params']['sw_process_natproc'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    not_ponded = 1.0 - pond_area

    interflow_store_input = ((output['perc_through_root'] +
                             output['subroot_leak']) -
                             (not_ponded * output['interflow_bypass']))

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
        double[:] col_interflow_volume = np.zeros(length)
        double[:] col_infiltration_recharge = np.zeros(length)
        double[:] col_interflow_to_rivers = np.zeros(length)
        double[:] interflow_store_input = output['interflow_store_input']
        int interflow_zone = params['interflow_zone_mapping'][node]
        double var0 = params['init_interflow_store'][interflow_zone]
        double[:] var5 = np.full([length],
                                 params['infiltration_limit'][interflow_zone])
        double[:] var8 = np.full([length],
                                 params['interflow_decay'][interflow_zone])
        double volume = var0
        double var1, var6
        size_t num
        double[:] recharge = np.zeros(length)
        double[:] rivers = np.zeros(length)
        double previous_interflow_store_input = 0.0

    if params['interflow_process'] == 'enabled':
        col_interflow_volume = np.full([length], volume)
        col_infiltration_recharge = recharge
        col_interflow_to_rivers = rivers

        if params["infiltration_limit_use_timeseries"]:
            for day in range(length):
                var5[day] = series['infiltration_limit_ts'][day][interflow_zone-1]

        if params["interflow_decay_use_timeseries"]:
            for day in range(length):
                var8[day] = series['interflow_decay_ts'][day][interflow_zone-1]

        recharge = np.where(np.asarray(var5) < volume,
                            volume, np.asarray(var5))
        rivers = (np.full([length], volume) - np.asarray(recharge)) * var8

        previous_interflow_store_input = 0.0
        for num in range(length):
            var1 = volume - min(var5[num], volume)
            volume = previous_interflow_store_input + var1 * (1 - var8[num])
            col_interflow_volume[num] = volume
            col_infiltration_recharge[num] = min(var5[num], volume)
            var6 = (col_interflow_volume[num] - col_infiltration_recharge[num])
            col_interflow_to_rivers[num] = var6 * var8[num]
            previous_interflow_store_input = interflow_store_input[num]

    col = {}
    col['interflow_volume'] = col_interflow_volume.base
    col['infiltration_recharge'] = col_infiltration_recharge.base
    col['interflow_to_rivers'] = col_interflow_to_rivers.base

    return col

###############################################################################


def get_recharge_store_input(data, output, node):
    """AI) Input to Recharge Store [mm/d]."""

    params = data['params']

    cdef:
        double[:] sw_ponding_area = params['sw_pond_area']
        double pond_area
        size_t zone_sw

    if params['sw_process_natproc'] == 'enabled':
        zone_sw = params['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    if ff.use_natproc:
        recharge_store_input = (output['infiltration_recharge'] +
                                ((1.0 - pond_area) *
                                (output['interflow_bypass'] +
                                output['macropore_att'] +
                                output['runoff_recharge'])) +
                                (pond_area * output['pond_atten']))
    else:
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
        double[:] col_recharge_store = np.zeros(length)
        double[:] col_combined_recharge = np.zeros(length)
        double irs = params['recharge_attenuation_params'][node][0]
        double rlp = params['recharge_attenuation_params'][node][1]
        double rll = params['recharge_attenuation_params'][node][2]
        double recharge_store = 0
        double combined_recharge = 0
        double[:] recharge_store_input = output['recharge_store_input']
        double[:] macropore_dir = output['macropore_dir']
        size_t num, zone_sw
        double[:] sw_ponding_area = params['sw_pond_area']
        double macropore_num
        double pond_area

    if params['sw_process_natproc'] == 'enabled':
        zone_sw = params['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    if params['recharge_attenuation_process'] == 'enabled':
        recharge_store = irs
        col_recharge_store[0] = irs
        combined_recharge = (min((irs * rlp), rll) +
                                    ((1.0 - pond_area) *
                                     macropore_dir[0]) +
                                    (pond_area *
                                     output['pond_direct'][0]) +
                                    (pond_area *
                                     output['pond_atten'][0]))
        col_combined_recharge[0] = combined_recharge
        macropore_num = macropore_dir[0]
        for num in range(1, length):
            if ff.use_natproc:
                recharge_store = (recharge_store_input[num-1] +
                                recharge_store -
                                (combined_recharge -
                                ((1.0 - pond_area) *
                                macropore_dir[num-1])
                                - (pond_area * (output['pond_direct'][num-1] +
                                                output['pond_atten'][num-1]))))
            else:
                recharge_store = (recharge_store_input[num - 1] +
                                recharge_store -
                                combined_recharge +
                                macropore_num)
            
            macropore_num = macropore_dir[num]
            combined_recharge = (min((recharge_store * rlp), rll) +
                                          ((1.0 - pond_area) *
                                           macropore_num) +
                                          (pond_area *
                                           (output['pond_direct'][num] +
                                            output['pond_atten'][num])))

            col_recharge_store[num] = recharge_store
            col_combined_recharge[num] = combined_recharge
    else:
        if ff.use_natproc:
            # This branch does not modify or set the whole array here, so I think that setting index zero to irs here is a bug.
            pass
        else:
            col_recharge_store[0] = irs
        for num in range(1, length):
            col_combined_recharge[num] = (recharge_store_input[num] +
                                          ((1.0 - pond_area) *
                                           output['macropore_dir'][num]) +
                                           (pond_area *
                                           (output['pond_direct'][num] +
                                            output['pond_atten'][num])))

    col = {}
    col['recharge_store'] = col_recharge_store.base
    col['combined_recharge'] = col_combined_recharge.base
    return col

###############################################################################


def get_mf6rch_file(data, rchrate):
    """get mf6 RCH object."""

    import os.path

    cdef int node_index, per

    # this is equivalent of strange hardcoded 1000 in format_recharge_row
    #  which is called in the mf6 output function
    fac = 0.001
    path = make_path(data)
    rch_params = data['params']['recharge_node_mapping']
    nper = extract_nper(data)
    nodes = extract_node_count(data)

    node_index_to_rch_index = np.full(nodes, -1, dtype=int)
    if rch_params is not None:
        for node_number, vals in rch_params.iteritems():
            node_index = node_number - 1
            node_index_to_rch_index[node_index] = vals[0] - 1
    else:
        for node_index in range(nodes):
            node_index_to_rch_index[node_index] = node_index

    maxbound = (node_index_to_rch_index >= 0).sum()

    if data['params']['disv']:
        rch_out = make_mf6_rch_file_with_disv(path, nodes, nper, maxbound, node_index_to_rch_index, rchrate, fac)
    else:
        rch_out = make_mf6_rch_file_with_disu(path, nodes, nper, maxbound, node_index_to_rch_index, rchrate, fac)

    return rch_out

def extract_nper(data):
    return len(data['params']['time_periods'])

def extract_node_count(data):
    return data['params']['num_nodes']

def make_path(data):
    fileout = data['params']['run_name']
    return os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)

def make_mf6_rch_file_with_disv(path, nodes, nper, maxbound, node_index_to_rch_index, rchrate, fac):
    m, spd = flopy_adaptor.make_model_with_disv_and_empty_spd_for_rch_out(path, nper, maxbound)

    for per in tqdm(range(nper), desc="Generating MF6 RCH  "):
        spd_index = 0
        for node_index in range(nodes):
            rch_index = node_index_to_rch_index[node_index]
            if rch_index >= 0:
                spd[per][spd_index] = ((0, rch_index),
                            rchrate[(nodes * per) + node_index + 1] * fac)
                spd_index += 1

    rch_out = flopy_adaptor.mf_gwf_rch(m, maxbound, spd)
    spd = None
    return rch_out

def make_mf6_rch_file_with_disu(path, nodes, nper, maxbound, node_index_to_rch_index, rchrate, fac):
    m, spd = flopy_adaptor.make_model_with_disu_and_empty_spd_for_rch_out(path, nper, nodes, maxbound)

    for per in tqdm(range(nper), desc="Generating MF6 RCH  "):
        spd_index = 0
        for node_index in range(nodes):
            rch_index = node_index_to_rch_index[node_index]
            if rch_index >= 0:
                spd[per][spd_index] = ((rch_index,),
                            rchrate[(nodes * per) + node_index + 1] * fac)
                spd_index += 1

    rch_out = flopy_adaptor.mf_gwf_rch(m, maxbound, spd)
    spd = None
    return rch_out

###############################################################################


def get_combined_str(data, output, node):
    """Multicolumn function.

    AL) SW Attenuation Store [mm].
    AM) STR: Combined Surface Flow To Surface Water Courses [mm/d].
    """
    series, params = data['series'], data['params']

    cdef:
        size_t length = len(series['date'])
        double[:] col_attenuation = np.zeros(length)
        double[:] old_col_attenuation = np.zeros(length)
        double[:] col_atten_input = np.zeros(length)
        double[:] col_atten_input_actual = np.zeros(length)
        double[:] col_pond_direct = np.zeros(length)
        double[:] col_pond_atten = np.zeros(length)
        double[:] col_pond_over = np.zeros(length)
        double[:] col_sw_other = np.zeros(length)
        double[:] col_open_water_evap = np.zeros(length)
        double[:] col_open_water_ae = np.zeros(length)
        double[:] col_combined_str = np.zeros(length)
        double[:] combined_str = np.zeros(length)
        long long[:] months = np.array(series['months'], dtype=np.int64)
        size_t zone_sw = params['sw_zone_mapping'][1] - 1
        double[:, :] sw_pe_to_open_water = params['sw_pe_to_open_wat']
        double[:, :] sw_direct_recharge = params['sw_direct_rech']
        double[:, :] sw_activation = params['sw_activ']
        double[:, :] sw_bed_infiltration = params['sw_bed_infiltn']
        double[:, :] sw_downstream = params['sw_downstr']
        double[:] sw_ponding_area = params['sw_pond_area']
        double pond_depth, other_sw_flow, pond_overspill, tmp0, tmp1
        double pond_depth_new, tmp0_new, input_to_atten_store_actual, tmp2
        double input_to_atten_store, pond_direct, pond_atten
        # double[:] some_zeros = np.zeros(length)
        double rlp = params['sw_params'][node][1]
        double base = max((params['sw_params'][node][0] +
                           params['sw_init_ponding'] +
                           output['interflow_to_rivers'][0] +
                           output['swabs_ts'][0] +
                           output['swdis_ts'][0] +
                           output['rapid_runoff'][0] -
                           output['runoff_recharge'][0]), 0.0)
        int month
        size_t num

    combined_str = (output['interflow_to_rivers'] +
                    output['swabs_ts'] +
                    output['swdis_ts'] +
                    output['rapid_runoff'] -
                    output['runoff_recharge'] +
                    output['rejected_recharge'])

    for num in range(length):
        if combined_str[num] < 0.0:
            output['swabs_ts'][num] = (output['interflow_to_rivers'][num] +
                                       output['swdis_ts'][num] +
                                       output['rapid_runoff'][num] -
                                       output['runoff_recharge'][num] +
                                       output['rejected_recharge'][num])

    if params['sw_process'] == 'enabled':
        # don't attenuate negative flows
        if base < 0.0:
            rlp = 1.0
        col_combined_str[0] = rlp * base
        col_attenuation[0] = base - col_combined_str[0]

        for num in range(1, length):
            base = (col_attenuation[num-1] +
                    combined_str[num])
            # don't attenuate negative flows
            if base < 0.0:
                rlp = 1.0
            else:
                rlp = params['sw_params'][node][1]
            col_combined_str[num] = rlp * base
            col_attenuation[num] = base - col_combined_str[num]

    elif params['sw_process_natproc'] == 'enabled':

        col_attenuation[0] = params['sw_init_ponding']
        zone_sw = params['sw_zone_mapping'][node] - 1

        for num in range(1, length):
            base = combined_str[num]
            month = months[num] #+ 1
            open_water_evap = 0.0
            open_water_ae = 0.0
            pond_overspill = 0.0
            other_sw_flow = 0.0
            pond_direct = 0.0
            pond_atten = 0.0
            pond_depth = 0.0
            input_to_atten_store = 0.0
            input_to_atten_store_actual = 0.0

            # don't attenuate negative flows
            if base < 0.0 or (not sw_ponding_area[zone_sw] > 0.0):
                col_combined_str[num] = base
                col_attenuation[num] = col_attenuation[num - 1]
                old_col_attenuation[num] = old_col_attenuation[num - 1]
            else:
                if col_attenuation[num - 1] > sw_activation[month][zone_sw]:
                    open_water_evap = (sw_ponding_area[zone_sw] *
                                       sw_pe_to_open_water[month][zone_sw] *
                                       output['pe_ts'][num])
                input_to_atten_store = (sw_ponding_area[zone_sw] *
                                        output['rainfall_ts'][num] - open_water_evap +
                                        ((1.0 - sw_ponding_area[zone_sw]) *
                                        (output['interflow_to_rivers'][num] +
                                         output['rapid_runoff'][num] +
                                         output['rejected_recharge'][num] -
                                         output['runoff_recharge'][num])))

                tmp0 = ((1.0 - sw_ponding_area[zone_sw])
                        / sw_ponding_area[zone_sw])
                tmp0_new = (1.0 / sw_ponding_area[zone_sw])
                pond_depth = col_attenuation[num - 1] + tmp0 * input_to_atten_store
                pond_depth_new = col_attenuation[num - 1] + tmp0_new * input_to_atten_store

                if pond_depth_new > params['sw_max_ponding']:
                    pond_overspill = pond_depth_new - params['sw_max_ponding']

                tmp1 = col_attenuation[num - 1] + tmp0 * input_to_atten_store - pond_overspill

                if tmp1 > sw_activation[month][zone_sw]:
                    other_sw_flow = (sw_downstream[month][zone_sw] *
                                      (tmp1 - sw_activation[month][zone_sw]))

                col_combined_str[num] = (output['swabs_ts'][num] +
                                         output['swdis_ts'][num] +
                                         (sw_ponding_area[zone_sw]
                                          * (pond_overspill + other_sw_flow)))

                tmp2 = (tmp0 * input_to_atten_store -
                        pond_overspill -
                        other_sw_flow)

                if (col_attenuation[num - 1] + tmp2) > sw_activation[month][zone_sw]:
                    pond_direct = (sw_bed_infiltration[month][zone_sw] *
                                   sw_direct_recharge[month][zone_sw] *
                                   (col_attenuation[num - 1] + tmp2 -
                                    sw_activation[month][zone_sw]))
                    pond_atten = (sw_bed_infiltration[month][zone_sw] *
                                  (1.0 - sw_direct_recharge[month][zone_sw]) *
                                  (col_attenuation[num - 1] + tmp2 -
                                   sw_activation[month][zone_sw]))

                old_col_attenuation[num] = (col_attenuation[num - 1] +
                                            (1.0 / sw_ponding_area[zone_sw]) *
                                            input_to_atten_store -
                                            pond_overspill -
                                            other_sw_flow -
                                            pond_direct -
                                            pond_atten)

                col_attenuation[num] = max(0.0, old_col_attenuation[num])

                if old_col_attenuation[num] < 0.0:
                    open_water_ae = (open_water_evap +
                                     sw_ponding_area[zone_sw]
                                     * old_col_attenuation[num])
                else:
                    open_water_ae = open_water_evap

                input_to_atten_store_actual = (sw_ponding_area[zone_sw] *
                                               output['rainfall_ts'][num] -
                                               open_water_ae +
                                               ((1.0 - sw_ponding_area[zone_sw]) *
                                               (output['interflow_to_rivers'][num] +
                                                output['rapid_runoff'][num] +
                                                output['rejected_recharge'][num] -
                                                output['runoff_recharge'][num])))

                col_pond_direct[num] = pond_direct
                col_pond_atten[num] = pond_atten
                col_pond_over[num] = pond_overspill
                col_sw_other[num] = other_sw_flow
                col_open_water_evap[num] = open_water_evap
                col_open_water_ae[num] = open_water_ae
                col_atten_input[num] = input_to_atten_store
                col_atten_input_actual[num] = input_to_atten_store_actual

    else:
        col_combined_str = combined_str

    col = {}
    col['sw_attenuation'] = col_attenuation.base
    col['pond_direct'] = col_pond_direct.base
    col['pond_atten'] = col_pond_atten.base
    col['pond_over'] = col_pond_over.base
    col['sw_other'] = col_sw_other.base
    col['open_water_evap'] = col_open_water_evap.base
    col['open_water_ae'] = col_open_water_ae.base
    col['combined_str'] = col_combined_str.base
    col['atten_input'] = col_atten_input.base
    col['atten_input_actual'] = col_atten_input_actual.base

    return col

###############################################################################


def get_combined_ae(data, output, node):
    """AN) AE: Combined AE [mm/d]."""

    cdef:
        double[:] sw_ponding_area = data['params']['sw_pond_area']
        double pond_area
        size_t zone_sw

    if data['params']['sw_process_natproc'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    combined_ae = (((1.0 - pond_area) * output['canopy_storage']) +
                   ((1.0 - pond_area) * output['ae']) +
                   output['open_water_ae'])

    return {'combined_ae': combined_ae}

###############################################################################


def get_evt(data, output, node):
    """AO) EVT: Unitilised PE [mm/d]."""

    cdef:
        double[:] sw_ponding_area = data['params']['sw_pond_area']
        double pond_area
        size_t zone_sw

    if data['params']['sw_process_natproc'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    return {'evt': (1.0 - pond_area) * output['unutilised_pe']}

###############################################################################


def get_average_in(data, output, node):
    """AP) AVERAGE IN [mm]."""
    average_in = (output['rainfall_ts'] +
                  output['subroot_leak'] +
                  output['swdis_ts'])
    return {'average_in': average_in}

###############################################################################


def get_average_out(data, output, node):
    """AQ) AVERAGE OUT [mm]."""

    cdef:
        double[:] sw_ponding_area = data['params']['sw_pond_area']
        double pond_area
        size_t zone_sw

    if data['params']['sw_process_natproc'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    if ff.use_natproc:
        # The natproc branch replaced the terms 'ae', 'canopy_storage' and 'swabs_ts'
        # with a term for 'combined_ae'.
        #
        # 'combined_ae' includes terms for:
        #     * 'ae'
        #     * 'canopy_storage'
        #     * 'open_water_ae'
        # but it does not include a term for 'swabs_ts'.
        average_out = (output['combined_str'] +
                    output['combined_recharge'] +
                    output['combined_ae'])
    else:
        average_out = (output['combined_str'] +
                    output['combined_recharge'] +
                    output['ae'] +
                    output['canopy_storage'] +
                    output['swabs_ts'])

    return {'average_out': average_out}

###############################################################################


def get_change(data, output, node):
    """AR) TOTAL STORAGE CHANGE [mm]."""
    series = data['series']
    params = data['params']

    cdef:
        size_t length = len(series['date'])
        double[:] col_change = np.zeros(length)
        double[:] tmp0 = np.zeros(length)
        size_t num, zone_sw
        double[:] sw_ponding_area = params['sw_pond_area']
        double pond_area, not_ponded

    if params['sw_process_natproc'] == 'enabled':
        zone_sw = params['sw_zone_mapping'][node] - 1
        pond_area = sw_ponding_area[zone_sw]
    else:
        pond_area = 0.0

    not_ponded = 1.0 - pond_area

    tmp0 = (output['recharge_store_input'] -
            (output['combined_recharge'] -
             not_ponded * output['macropore_dir'] - pond_area * output['pond_direct']) +
            not_ponded * (output['interflow_store_input'] - output['interflow_to_rivers']) -
            output['infiltration_recharge'] +
            pond_area * output['subroot_leak'] +
            not_ponded * (output['percol_in_root'] - output['ae']))

    col_change = tmp0

    for num in range(1, length):
        if output['p_smd'][num] < 0.0:
            col_change[num] += (not_ponded * output['p_smd'][num])

        if ff.use_natproc:
            # Not sure about this. Introducing the pond_area factor here means that this term will be zero if there is no ponding. This may have been a mistake.
            col_change[num] += (pond_area * (output['sw_attenuation'][num]
                                - output['sw_attenuation'][num-1]))
        else:
            col_change[num] += (output['sw_attenuation'][num]
                                - output['sw_attenuation'][num-1])

    return {'total_storage_change': col_change.base}

###############################################################################


def get_balance(data, output, node):
    """AS) BALANCE [mm]."""
    balance = (output['average_in'] -
               output['average_out'] -
               output['total_storage_change'])

    return {'balance': balance}

###############################################################################

def _calculate_total_mass_leached_from_cell_on_days(
        double max_load_per_year_kg_per_cell,
        double her_at_5_percent,
        double her_at_50_percent,
        double her_at_95_percent,
        days,
        double[:] her_per_day):
    cdef:
        size_t length, i
        double remaining_for_year, her, fraction_leached, mass_leached_for_day
        double[:] result

    length = len(days)
    result = np.zeros(length)
    remaining_for_year = max_load_per_year_kg_per_cell
    for i in range(length):
        day = days[i]
        her = her_per_day[i]

        if (day.month == 10) and (day.day == 1):
            remaining_for_year = max_load_per_year_kg_per_cell

        if her == 0.0:
            mass_leached_for_day = 0.0
        else:            
            fraction_leached = _cumulative_fraction_leaked_per_day(her_at_5_percent,
                her_at_50_percent,
                her_at_95_percent,
                her)
            mass_leached_for_day = min(remaining_for_year, max_load_per_year_kg_per_cell * fraction_leached)
            remaining_for_year -= mass_leached_for_day
        result[i] = mass_leached_for_day
    return result

###############################################################################

def _cumulative_fraction_leaked_per_day(double her_at_5_percent, double her_at_50_percent, double her_at_95_percent, double her_per_day):
    cdef:
        double days_in_year = 365.25
        double her_per_year, y

    her_per_year = days_in_year * her_per_day
    y = _cumulative_fraction_leaked_per_year(her_at_5_percent, her_at_50_percent, her_at_95_percent, her_per_year)
    return y / days_in_year

###############################################################################

def _cumulative_fraction_leaked_per_year(double her_at_5_percent, double her_at_50_percent, double her_at_95_percent, double her_per_year):
    cdef:
        double x, upper, lower, m, c, y

    x = her_per_year
    if her_per_year < her_at_50_percent:
        upper = her_at_50_percent
        lower = her_at_5_percent
    else:
        upper = her_at_95_percent
        lower = her_at_50_percent
    # y = mx + c
    if upper == lower:
        m = 0
    else:
        m = 0.45 / (upper - lower)
    c = 0.5 - (her_at_50_percent * m)
    y = (m * x) + c
    return max(0, y)

###############################################################################

def calculate_mass_reaching_water_table_array_kg_per_day(blackboard):
    cdef:
        double[:] proportion_reaching_water_table_array_per_day = blackboard.proportion_reaching_water_table_array_per_day
        double[:] mi_array_kg_per_day = blackboard.mi_array_kg_per_day
    return _calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)

###############################################################################

def calculate_historical_mass_reaching_water_table_array_kg_per_day(blackboard):
    cdef:
        double[:] proportion_reaching_water_table_array_per_day = blackboard.historic_proportion_reaching_water_table_array_per_day
        double[:] mi_array_kg_per_day = blackboard.truncated_historical_mi_array_kg_per_day

    days = blackboard.days
    return _calculate_historical_mass_reaching_water_table_array_kg_per_day(
        days,
        proportion_reaching_water_table_array_per_day,
        mi_array_kg_per_day)

###############################################################################

def _calculate_mass_reaching_water_table_array_kg_per_day(
        double[:] proportion_reaching_water_table_array_per_day,
        double[:] mi_array_kg_per_day):
    cdef:
        size_t length
        size_t day_nitrate_was_leached
        size_t result_end
        size_t i
        double[:] result_kg
        double mass_leached_on_day_kg

    length = proportion_reaching_water_table_array_per_day.size
    result_kg = np.zeros(length)
    for day_nitrate_was_leached in range(length):
        mass_leached_on_day_kg = mi_array_kg_per_day[day_nitrate_was_leached]
        if mass_leached_on_day_kg == 0:
            continue
        result_end = length - day_nitrate_was_leached
        for i in range(result_end):
            result_kg[day_nitrate_was_leached + i] += proportion_reaching_water_table_array_per_day[i] * mass_leached_on_day_kg

    return np.array(result_kg)

###############################################################################

def _calculate_historical_mass_reaching_water_table_array_kg_per_day(
        days,
        double[:] proportion_reaching_water_table_array_per_day,
        double[:] mi_array_kg_per_day):
    cdef:
        size_t days_count, historic_days_count
        size_t day_nitrate_was_leached
        size_t result_end
        size_t i
        double[:] result_kg
        double mass_leached_on_day_kg

    days_count = len(days)
    historic_days_count = mi_array_kg_per_day.size

    result_kg = np.zeros(days_count)
    for day_nitrate_was_leached in range(historic_days_count):
        mass_leached_on_day_kg = mi_array_kg_per_day[day_nitrate_was_leached]
        if mass_leached_on_day_kg == 0:
            continue
        for i in range(days_count):
            proportion_index = historic_days_count - day_nitrate_was_leached + i
            result_kg[i] += proportion_reaching_water_table_array_per_day[proportion_index] * mass_leached_on_day_kg

    return np.array(result_kg)

###############################################################################

def _calculate_m1a_b_array_kg_per_day(blackboard):
    cdef:
        size_t length
        size_t i
        double mit_kg
        double m1a_kg_per_day
        double m1b_kg_per_day
        double[:] m1_array_kg_per_day = blackboard.m1_array_kg_per_day
        double[:] end_interflow_store_volume_mm = blackboard.interflow_volume
        double[:] infiltration_recharge_mm_per_day = blackboard.infiltration_recharge
        double[:] interflow_to_rivers_mm_per_day = blackboard.interflow_to_rivers
        double[:] interflow_store_components_mm_per_day
        double[:] recharge_proportion
        double[:] interflow_proportion
        double[:,:] m1a_b_array_kg_per_day
        

    interflow_store_components_mm_per_day = np.add(np.add(end_interflow_store_volume_mm, infiltration_recharge_mm_per_day), interflow_to_rivers_mm_per_day)
    recharge_proportion = _divide_arrays(infiltration_recharge_mm_per_day, interflow_store_components_mm_per_day)
    interflow_proportion = _divide_arrays(interflow_to_rivers_mm_per_day, interflow_store_components_mm_per_day)

    length = m1_array_kg_per_day.size
    m1a_b_array_kg_per_day = np.zeros(shape=(2,length))
    mit_kg = 0

    for i in range(length):
        mit_kg += m1_array_kg_per_day[i]
        m1a_kg_per_day = mit_kg * recharge_proportion[i]
        m1b_kg_per_day = mit_kg * interflow_proportion[i]
        mit_kg = mit_kg - m1a_kg_per_day - m1b_kg_per_day
        m1a_b_array_kg_per_day[0,i] = m1a_kg_per_day 
        m1a_b_array_kg_per_day[1,i] = m1b_kg_per_day        
    return m1a_b_array_kg_per_day

###############################################################################

def _divide_arrays(double[:] a, double[:] b):
    cdef:
        double[:] result = np.zeros_like(a)
    
    for i in range(len(a)):
        if b[i] != 0:
            result[i] = a[i] / b[i]
    return result

###############################################################################

def _divide_2D_arrays(double[:,:] a, double[:,:] b):
    cdef:
        double[:,:] result = np.zeros_like(a)
    
    for i in range(a.shape[0]):
       for j in range(a.shape[1]):
        if b[i,j] != 0:
            result[i,j] = a[i,j] / b[i,j]
    return result

###############################################################################

def _aggregate_nitrate(
            time_periods,
            size_t len_time_periods,
            double[:] nitrate_reaching_water_table_array_tons_per_day,
            double[:] combined_recharge_m_cubed,
            double[:,:] aggregation,
            size_t node,
            node_areas):
    cdef:
        size_t time_period_index, first_day_index, last_day_index, i
        double sum_of_nitrate_tons, sum_of_recharge_m_cubed, sum_of_recharge_mm, stored_mass_tons

    stored_mass_tons = 0
    for time_period_index in range(len_time_periods):
        time_period = time_periods[time_period_index]
        first_day_index = time_period[0] - 1
        last_day_index = time_period[1] - 1
        sum_of_nitrate_tons = 0.0
        sum_of_recharge_m_cubed = 0.0
        for i in range(first_day_index, last_day_index):
            sum_of_nitrate_tons += nitrate_reaching_water_table_array_tons_per_day[i]
            sum_of_recharge_m_cubed += combined_recharge_m_cubed[i]
        sum_of_recharge_mm = 1000 * sum_of_recharge_m_cubed / node_areas[node]
        if sum_of_recharge_mm > 1:
            aggregation[time_period_index, node] += (stored_mass_tons + sum_of_nitrate_tons) / sum_of_recharge_m_cubed
            stored_mass_tons = 0
        else:
            stored_mass_tons += sum_of_nitrate_tons
            aggregation[time_period_index, node] = 0

    return aggregation
    
###############################################################################

def _aggregate_surface_water_nitrate(
            time_periods,
            size_t len_time_periods,
            double[:] nitrate_to_surface_water_array_tons_per_day,
            double[:,:] aggregation,
            size_t node):
    cdef:
        size_t time_period_index, first_day_index, last_day_index, i

    for time_period_index in range(len_time_periods):
        time_period = time_periods[time_period_index]
        first_day_index = time_period[0] - 1
        last_day_index = time_period[1] - 1
        for i in range(first_day_index, last_day_index):
            aggregation[time_period_index, node] += nitrate_to_surface_water_array_tons_per_day[i]
        aggregation[time_period_index, node] = aggregation[time_period_index, node] / (last_day_index - first_day_index + 1)

    return aggregation

###############################################################################

def aggregate_mi(
            double[:,:] aggregation,
            time_periods,
            size_t len_time_periods,
            double[:] mi_array_kg_per_day,
            size_t node):
    cdef:
        size_t time_period_index, first_day_index, last_day_index, day_index

    for time_period_index in range(len_time_periods):
        time_period = time_periods[time_period_index]
        first_day_index = time_period[0] - 1
        last_day_index = time_period[1] - 1
        for day_index in range(first_day_index, last_day_index):
            aggregation[node][time_period_index] += mi_array_kg_per_day[day_index]
    return aggregation

###############################################################################

def _calculate_aggregate_mi_unpacking(blackboard):
    cdef:
        size_t length = len(blackboard.historical_nitrate_days)
        double[:] historical_mi_array_kg_per_day = np.zeros(length)
        size_t time_period_index, start_day, end_day, days_in_time_period
        double total_mi_for_time_period_kg, historical_mi_kg_per_day

    for time_period_index in range(len(blackboard.historical_time_periods)):
        time_period = blackboard.historical_time_periods[time_period_index]
        start_day = time_period[0] - 1
        end_day = time_period[1] - 1
        days_in_time_period = end_day - start_day

        total_mi_for_time_period_kg = blackboard.historical_mi_array_kg_per_time_period[time_period_index]
        historical_mi_kg_per_day = total_mi_for_time_period_kg / days_in_time_period

        for day in range(start_day, end_day):
            historical_mi_array_kg_per_day[day] = historical_mi_kg_per_day
    return historical_mi_array_kg_per_day

def write_nitrate_csv(filename, nitrate_aggregation, header_row):
    stress_period_count = nitrate_aggregation.shape[0]
    node_count = nitrate_aggregation.shape[1]

    int_to_bytes = []
    for i in range(1, 1 + max(stress_period_count, node_count)):
        int_to_bytes.append(str(i).encode())

    with open(filename, "wb") as f:
        f.write(header_row)
        for stress_period_index in range(stress_period_count):
            stress_period_bytes = int_to_bytes[stress_period_index]
            for node_index in range(node_count):
                node = node_index + 1
                concentration = nitrate_aggregation[stress_period_index, node_index]
                line = b"%b,%i,%g\r\n" % (stress_period_bytes, node, concentration)
                f.write(line)

###############################################################################


def aggregate(output, area, ponded_frac, reporting=None, index=None):
    """Aggregate reporting over output periods."""
    new_rep = {}

    if index is not None:
        not_scalar = (type(index[0]) is range or type(index[0]) is list)
        convert = np.float64
    else:
        not_scalar = False
        convert = lambda x: x

    for key in output:

        # lookup key in utils constants to see which area to use
        if ff.use_natproc:
            area_fn = u.area_fn()[key]
        else:
            area_fn = lambda area, ponded_fraction: area
        new_rep[key] = []
        if not_scalar:
            new_rep[key] = [output[key][i].mean(dtype=np.float64)
                            * area_fn(area, ponded_frac) for i in index]
        elif index is not None:
            new_rep[key] = [convert(output[key][index[0]]) *
                            area_fn(area, ponded_frac)]
        else:
            new_rep[key] = convert(output[key]) * area_fn(area, ponded_frac)
        if reporting:
            new_rep[key] += convert(reporting[key])
    return new_rep

###############################################################################

def aggregate_op(output, area):
    """Aggregate reporting over output periods."""
    new_rep = {}
    for key in output:
        new_rep[key] = output[key] * area
    return new_rep

###############################################################################

def aggregate_reporting_op(output, area, reporting):
    """Aggregate reporting over output periods."""
    new_rep = {}
    for key in output:
        new_rep[key] = output[key] * area
        if reporting:
            new_rep[key] += reporting[key]
    return new_rep

def aggregate_op(output, area):
    """Aggregate reporting over output periods."""
    new_rep = {}
    for key in output:
        new_rep[key] = output[key] * area
    return new_rep

def aggregate_reporting_op(output, area, reporting):
    """Aggregate reporting over output periods."""
    new_rep = {}
    for key in output:
        new_rep[key] = output[key] * area
        if reporting:
            new_rep[key] += reporting[key]
    return new_rep

###############################################################################

def get_aggregated_sfr_flows(data, nss, sorted_by_ca, runoff_with_area, swac_seg_dic):
    nper = extract_nper(data)
    nodes = extract_node_count(data)
    description = "Accumulating SFR flows  "
    result = np.zeros((nper, nss))
    for per in tqdm(range(nper), desc=description):
        result_A, result_B = get_sfr_flows(sorted_by_ca, runoff_with_area, swac_seg_dic, nodes * per, nodes, nss)
        for iseg in range(nss):
            result[per, iseg] = result_A[iseg] + result_B[iseg]
    return result

def get_aggregated_stream_mass(data, nss, sorted_by_ca, stream_nitrate_aggregation, swac_seg_dic):
    nper = extract_nper(data)
    nodes = extract_node_count(data)
    description = "Accumulating nitrate mass to surface water  "
    result = np.zeros((nper, nss))
    for per in tqdm(range(nper), desc=description):
        result_A, result_B = get_sfr_flows_nitrate(sorted_by_ca, swac_seg_dic, stream_nitrate_aggregation, per, nodes, nss)
        for iseg in range(nss):
            result[per, iseg] = result_A[iseg] + result_B[iseg]
    return result

def get_sfr_flows(sorted_by_ca, runoff, swac_seg_dic, nodes_per, nodes, nss):
    """get flows for one period"""

    source = runoff
    index_offset = nodes_per + 1
    return get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset)

def get_sfr_flows_nitrate(sorted_by_ca, swac_seg_dic, stream_nitrate_aggregation, period, nodes, nss):
    """get flows and nitrate masses for one period"""

    source = stream_nitrate_aggregation[period,:]
    index_offset = 0
    return get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset)

def get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset):
    result_A = np.zeros((nss))
    result_B = np.zeros((nss))
    done = np.zeros((nodes), dtype=int)

    for node_number, line in sorted_by_ca.items():
        node_index = node_number - 1
        downstr, str_flag = line[:2]
        acc = 0.0

        while downstr > 1:
            str_flag = sorted_by_ca[node_number][1]
            is_str = str_flag >= 1
            is_done = done[node_index] == 1

            if is_str:
                stream_cell_index = swac_seg_dic[node_number] - 1

                if is_done:
                    result_B[stream_cell_index] += acc
                    acc = 0.0
                    break
                else:
                    result_A[stream_cell_index] = source[node_index + index_offset]
                    result_B[stream_cell_index] = acc
                    done[node_index] = 1
                    acc = 0.0

            else:
                if not is_done:
                    acc += max(0.0, source[node_index + index_offset])
                    done[node_index] = 1

            node_number = downstr
            node_index = node_number - 1
            downstr = sorted_by_ca[node_number][0]

    return result_A, result_B

def extract_release_proportion(data, stream_ca_order, time_period):
    result = []
    month_key = data["series"]["date"][time_period[0] - 1].month
    zone_index_to_proportion = data["params"]["sfr_flow_monthly_proportions"][month_key]
    sfr_flow_zones = data["params"]["sfr_flow_zones"]
    for node_index, _, _ in stream_ca_order:
        node_number = node_index + 1
        zone_number = sfr_flow_zones[node_number][0]
        zone_index = zone_number - 1
        result.append(zone_index_to_proportion[zone_index])
    return result

def get_attenuated_sfr_flows(sorted_by_ca, swac_seg_dic, nodes, source, index_offset, sfr_store_init, release_proportion):
    nodes_ca_order, stream_ca_order = organise_nodes_and_stream_data(sorted_by_ca, swac_seg_dic)
    coalesced_runoff = calculate_coalesced_runoff(nodes_ca_order, source, index_offset, nodes)
    stream_count = len(swac_seg_dic)
    coalesced_stream_runoff = calculate_coalesced_stream_runoff(stream_ca_order, coalesced_runoff, stream_count)
    sfr_store_total = sfr_store_init + coalesced_stream_runoff
    sfr_released = accumulate_stream_and_calculate_sfr_store_total(stream_ca_order, sfr_store_total, release_proportion, stream_count)
    sfr_store_total = sfr_store_total - sfr_released
    de_accumulated_flows = calculate_de_accumulated_flows(stream_ca_order, sfr_released)
    runoff_result = np.zeros(stream_count)
    return runoff_result, de_accumulated_flows, sfr_store_total

def organise_nodes_and_stream_data(sorted_by_ca, swac_seg_dic):
    nodes_ca_order = []
    stream_ca_order = []
    for node_number, line in sorted_by_ca.items():
        node_index = node_number - 1
        downstr_node_number, str_flag = line[:2]
        downstream_node_index = downstr_node_number - 1
        nodes_ca_order.append((node_index, downstream_node_index, str_flag))
        if str_flag >= 1:
            stream_number = swac_seg_dic[node_number]
            stream_index = stream_number - 1
            if downstream_node_index >= 0:
                downstream_stream_number = swac_seg_dic[downstr_node_number]
                downstream_stream_index = downstream_stream_number - 1
            else:
                downstream_stream_index = -1
            stream_ca_order.append((node_index, stream_index, downstream_stream_index))
    return nodes_ca_order, stream_ca_order

def calculate_coalesced_runoff(nodes_ca_order, source, index_offset, nodes):
    coalesced_runoff = np.zeros(nodes)
    for node_index, downstream_node_index, str_flag in nodes_ca_order:
        if str_flag >= 1:
            coalesced_runoff[node_index] += source[node_index + index_offset]
        elif downstream_node_index >= 0:
            coalesced_runoff[downstream_node_index] += source[node_index + index_offset] + coalesced_runoff[node_index]
    return coalesced_runoff

def calculate_coalesced_stream_runoff(stream_ca_order, coalesced_runoff, stream_count):
    coalesced_stream_runoff = np.zeros(stream_count)
    for node_index, stream_index, _ in stream_ca_order:
        coalesced_stream_runoff[stream_index] = coalesced_runoff[node_index]
    return coalesced_stream_runoff

def accumulate_stream_and_calculate_sfr_store_total(stream_ca_order, sfr_store_total, release_proportion, stream_count):
    sfr_released = np.zeros(stream_count)
    for _, index, downstream_index in stream_ca_order:
        sfr_released[index] = sfr_store_total[index] * release_proportion[index]
        if downstream_index >= 0:
            sfr_store_total[downstream_index] += sfr_released[index]
    return sfr_released

def calculate_de_accumulated_flows(stream_ca_order, sfr_released):
    de_accumulated_flows = np.copy(sfr_released)
    for _, index, downstream_index in stream_ca_order:
        if downstream_index >= 0:
            de_accumulated_flows[downstream_index] -= sfr_released[index]
    return de_accumulated_flows

###############################################################################


def get_sfr_file(data, runoff):
    """get SFR object."""

    if data['params']['gwmodel_type'] == 'mfusg':
        return _get_sfr_file_mfusg(data, runoff)
    elif data['params']['gwmodel_type'] == 'mf6':
        return _get_sfr_file_mf6(data, runoff)

def _get_sfr_file_mfusg(data, runoff):
    sorted_by_ca = make_sorted_by_ca(data)

    path = make_path(data)
    nper = extract_nper(data)
    nodes = extract_node_count(data)
    nstrm = nss = count_nss(sorted_by_ca)
    njag = nodes + 2
    lenx = int((njag/2) - (nodes/2))
    istcb1, istcb2 = data['params']['istcb1'], data['params']['istcb2']
    segment_data = make_segment_data(data, nss, sorted_by_ca, runoff)
    reach_data = make_reach_data(sorted_by_ca, nstrm)
    sfr_heading = "# SFR package for  MODFLOW-USG, generated by SWAcMod."

    return flopy_adaptor.make_sfr_file_mfusg(path, nper, nodes, nstrm, nss, njag, lenx, istcb1, istcb2, segment_data, reach_data, sfr_heading)

def make_reach_data(sorted_by_ca, nstrm):
    reach_data = flopy_adaptor.modflow_sfr2_get_empty_reach_data(nstrm)
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k,  # hcond1
         depth, width) = line
        if str_flag > 0:
            reach_data[str_count]['node'] = node_mf - 1  # external
            reach_data[str_count]['iseg'] = str_count + 1  # serial
            reach_data[str_count]['ireach'] = 1  # str_count + 1 # serial
            reach_data[str_count]['rchlen'] = length  # external
            reach_data[str_count]['strtop'] = z  # external
            reach_data[str_count]['strthick'] = bed_thk  # constant (for now)
            reach_data[str_count]['strhc1'] = str_k  # constant (for now)

            # inc stream counter
            str_count += 1
    return reach_data

def make_segment_data(data, nss, sorted_by_ca, runoff):
    sd = flopy_adaptor.modflow_sfr2_get_empty_segment_data(nss)
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k,  # hcond1
         depth, width) = line
        if str_flag > 0:
            sd[str_count]['nseg'] = str_count + 1  # serial
            sd[str_count]['icalc'] = 0  # constant
            sd[str_count]['outseg'] = 0
            sd[str_count]['iupseg'] = 0  # constant (0)
            sd[str_count]['flow'] = 0.0  # constant (for now - swac)
            sd[str_count]['runoff'] = 0.0  # constant (for now - swac)
            sd[str_count]['etsw'] = 0.0  # # cotnstant (0)
            sd[str_count]['pptsw'] = 0.0  # constant (0)
            # sd[str_count]['hcond1'] = hcond1 # get from lpf
            sd[str_count]['thickm1'] = bed_thk  # constant
            sd[str_count]['elevup'] = z  # get from mf
            sd[str_count]['width1'] = width  # constant
            sd[str_count]['depth1'] = depth  # constant
            sd[str_count]['width2'] = width  # constant
            sd[str_count]['depth2'] = depth  # constant

            # inc stream counter
            str_count += 1

    swac_seg_dic = make_swac_seg_dic(sorted_by_ca)
    seg_swac_dic = make_seg_swac_dic(sorted_by_ca)
    idx = make_idx()
    downstr_index = idx['downstr']
    for iseg in range(nss):
        node_swac = seg_swac_dic[iseg + 1]
        downstr = sorted_by_ca[node_swac][downstr_index]
        if downstr in swac_seg_dic:
            sd[iseg]['outseg'] = swac_seg_dic[downstr]
        else:
            sd[iseg]['outseg'] = 0

    segment_data = {}
    ro_and_flow_accumulator = lambda per, ro, flow: append_runoff_and_flow_to_sd(segment_data, sd, nss, per, ro, flow)
    append_ro_and_flow(data, runoff, sorted_by_ca, swac_seg_dic, ro_and_flow_accumulator)
    return segment_data

def count_nss(sorted_by_ca):
    idx = make_idx()
    str_flag_index = idx['str_flag']
    return sum([value[str_flag_index] > 0
                for value in sorted_by_ca.values()])

def append_ro_and_flow(data, runoff, sorted_by_ca, swac_seg_dic, ro_and_flow_accumulator):
    nper = extract_nper(data)
    areas = data['params']['node_areas']
    # units oddness - lots of hardcoded 1000s in input_output.py
    fac = 0.001

    nodes = extract_node_count(data)
    for per in range(nper):
        for node in range(1, nodes + 1):
            i = (nodes * per) + node
            runoff[i] = runoff[i] * areas[node] * fac

    # populate runoff and flow
    nss = count_nss(sorted_by_ca)
    if data['params']['attenuate_sfr_flows']:
        _, stream_ca_order = organise_nodes_and_stream_data(sorted_by_ca, swac_seg_dic)
        sfr_store = np.zeros(nss)

    for per in tqdm(range(nper), desc="Accumulating SFR flows  "):

        if data['params']['attenuate_sfr_flows']:
            time_period = data['params']['time_periods'][per]
            release_proportion = extract_release_proportion(data, stream_ca_order, time_period)
            ro, flow, sfr_store = get_attenuated_sfr_flows(sorted_by_ca, swac_seg_dic, nodes, runoff, (nodes * per) + 1, sfr_store, release_proportion)
        else:
            ro, flow = get_sfr_flows(sorted_by_ca, runoff, swac_seg_dic, nodes * per, nodes, nss)

        ro_and_flow_accumulator(per, ro, flow)

def append_runoff_and_flow_to_sd(segment_data, sd, nss, per, ro, flow):
    import copy

    for iseg in range(nss):
        sd[iseg]['runoff'] = ro[iseg]
        sd[iseg]['flow'] = flow[iseg]
    # add segment data for this period
    segment_data[per] = copy.deepcopy(sd)

def _get_sfr_file_mf6(data, runoff):
    sorted_by_ca = make_sorted_by_ca(data)
    swac_seg_dic = make_swac_seg_dic(sorted_by_ca)
    seg_swac_dic = make_seg_swac_dic(sorted_by_ca)

    is_disv = data['params']['disv']
    path = make_path(data)
    nper = extract_nper(data)
    nodes = extract_node_count(data)
    nss = count_nss(sorted_by_ca)
    connectiondata = make_connectiondata(data, sorted_by_ca, seg_swac_dic, swac_seg_dic)
    packagedata = make_packagedata(data, sorted_by_ca, connectiondata)
    perioddata = make_perioddata(data, sorted_by_ca, runoff, swac_seg_dic)
    optional_obs_filename = extract_optional_obs_filename(data)
    sfr_heading = "# SFR package for  MODFLOW-USG, generated by SWAcMod."

    return flopy_adaptor.make_sfr_file_mf6(is_disv, path, nper, nodes, nss, packagedata, connectiondata, perioddata, optional_obs_filename, sfr_heading)

def make_perioddata(data, sorted_by_ca, runoff, swac_seg_dic):
    perioddata = {}
    perioddata[0] = []
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        if str_flag > 0:
            perioddata[0].append((str_count, 'STAGE', z + depth))
            perioddata[0].append((str_count, 'STATUS', "SIMPLE"))
            str_count += 1

    nss = count_nss(sorted_by_ca)
    ro_and_flow_accumulator = lambda per, ro, flow: apppend_runoff_and_flow_to_perioddata(perioddata, nss, per, ro, flow)

    append_ro_and_flow(data, runoff, sorted_by_ca, swac_seg_dic, ro_and_flow_accumulator)
    return perioddata

def make_connectiondata(data, sorted_by_ca, seg_swac_dic, swac_seg_dic):
    nodes = extract_node_count(data)
    str_flg = np.zeros((nodes), dtype=int)
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        str_flg[node_swac-1] = str_flag

    connectiondata = []
    nodes = extract_node_count(data)
    idx = make_idx()
    Gs = build_graph(nodes, sorted_by_ca, str_flg, di=False)
    nss = count_nss(sorted_by_ca)
    for iseg in range(nss):
        conn = [iseg]
        node_swac = seg_swac_dic[iseg + 1]
        downstr = sorted_by_ca[node_swac][idx['downstr']]
        for n in Gs.neighbors(node_swac):
            if n in swac_seg_dic:
                if n == downstr:
                    conn.append(-float(swac_seg_dic[n] - 1))
                else:
                    conn.append(float((swac_seg_dic[n] - 1)))

        connectiondata.append(conn)
    return connectiondata

def make_packagedata(data, sorted_by_ca, connectiondata):
    packagedata = []
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        if str_flag > 0:
            if node_mf > 0:
                if data['params']['disv']:
                    n = (0, node_mf - 1)
                else:
                    n = (node_mf - 1,)
            else:
                if data['params']['disv']:
                    n = (-100000000, 0)
                else:
                    n = (-100000000, )

            packagedata.append([str_count, n, length, width,
                        0.0001, z, bed_thk, str_k, 0.0001, 1, 1.0, 0])

            str_count += 1

    nss = count_nss(sorted_by_ca)
    for iseg in range(nss):
        packagedata[iseg][9] = len(connectiondata[iseg]) - 1
    return packagedata

def extract_optional_obs_filename(data):
    if len(data['params']['sfr_obs']) > 0:
        return data['params']['sfr_obs']
    else:
        return None

def apppend_runoff_and_flow_to_perioddata(perioddata, nss, per, ro, flow):
    for iseg in range(nss):
        if per not in perioddata:
            perioddata[per] = []
        perioddata[per].append((iseg, 'RUNOFF', ro[iseg]))
        perioddata[per].append((iseg, 'INFLOW', flow[iseg]))


##############################################################################

def get_str_file(data, runoff):
    """get STR object."""

    sorted_by_ca = make_sorted_by_ca(data)
    nstrm = nss = count_nss(sorted_by_ca)
    m, dis, rd = make_modflow_str(data, nstrm, nss)
    swac_seg_dic = make_swac_seg_dic(sorted_by_ca)
    update_rd(sorted_by_ca, rd, dis)

    nper = extract_nper(data)
    idx = make_idx()
    nss = count_nss(sorted_by_ca)
    str_flg = make_str_flg(data, sorted_by_ca)
    seg_swac_dic = make_seg_swac_dic(sorted_by_ca)
    cd = initialise_segment(data, sorted_by_ca, str_flg, seg_swac_dic, idx, swac_seg_dic, nss)
    segment_data={iper: cd for iper in range(nper)}

    nstrm = count_nss(sorted_by_ca)
    istcb1 = data['params']['istcb1']
    istcb2 = data['params']['istcb2']
    reach_data = make_reach_data_for_str(data, runoff, sorted_by_ca, swac_seg_dic, rd)
    # segment_data
    strm = flopy_adaptor.modflow_str(m, nstrm, istcb1, istcb2, reach_data, segment_data)
    strm.heading = "# DELETE ME"

    return strm

def make_reach_data_for_str(data, runoff, sorted_by_ca, swac_seg_dic, rd):
    import copy

    nss = count_nss(sorted_by_ca)
    runoff_with_area = combine_runoff_with_area(data, runoff)
    str_flow_array = get_aggregated_sfr_flows(data, nss, sorted_by_ca, runoff_with_area, swac_seg_dic)

    reach_data = {}
    nper = extract_nper(data)
    for per in range(nper):
        reach_data[per] = copy.deepcopy(rd)
        for iseg in range(nss):
            reach_data[per][iseg]['flow'] = str_flow_array[per,iseg]
    return reach_data

def make_sorted_by_ca(data):
    rte_topo = data['params']['routing_topology']
    result = OrderedDict(sorted(rte_topo.items(), key=lambda x: x[1][4]))
    return result

def make_idx():
    names = ['downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z', 'bed_thk', 'str_k', 'depth', 'width']
    idx = {y: x for (x, y) in enumerate(names)}
    return idx

def make_modflow_str(data, nstrm, nss):
    m = make_modflow_model(data)
    dis = make_modflow_dis(m, data)
    flopy_adaptor.modflow_bas(m)
    rd, sd = flopy_adaptor.modflow_str_get_empty(nstrm, nss)
    return m, dis, rd

def make_modflow_model(data):
    import os.path
    path = make_path(data)
    result = flopy_adaptor.modflow_model(path, "mf2005", True)
    return result

def make_modflow_dis(m, data):
    nper = extract_nper(data)
    nlay, nrow, ncol = data['params']['mf96_lrc']
    result = flopy_adaptor.modflow_dis(m, nlay, nrow, ncol, nper)
    return result

def make_str_flg(data, sorted_by_ca):
    nodes = extract_node_count(data)
    str_flg = np.zeros((nodes), dtype=int)
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        str_flg[node_swac-1] = str_flag
        if str_flag > 0:
            str_count += 1
    return str_flg

def make_swac_seg_dic(sorted_by_ca):
    str_count = 0
    swac_seg_dic = {}
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        if str_flag > 0:
            swac_seg_dic[node_swac] = str_count + 1
            str_count += 1
    return swac_seg_dic

def make_seg_swac_dic(sorted_by_ca):
    str_count = 0
    seg_swac_dic = {}
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        if str_flag > 0:
            seg_swac_dic[str_count + 1] = node_swac
            str_count += 1
    return seg_swac_dic

def update_rd(sorted_by_ca, rd, dis):
    str_count = 0
    for _, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width) = line
        # for mf6 only
        if str_flag > 0:
            # NB docs say node number should be zero based (node_mf -1)
            #  but doesn't seem to be
            l, r, c = flopy_adaptor.dis_get_lrc(dis, node_mf)[0]
            rd[str_count]['k'] = l - 1
            rd[str_count]['i'] = r - 1
            rd[str_count]['j'] = c - 1
            rd[str_count]['segment'] = str_count + 1
            rd[str_count]['reach'] = 1
            rd[str_count]['stage'] = z + depth
            rd[str_count]['cond'] = (length * width * str_k) / bed_thk
            rd[str_count]['sbot'] = z - bed_thk
            rd[str_count]['stop'] = z
            rd[str_count]['width'] = width
            rd[str_count]['slope'] = 111.111
            rd[str_count]['rough'] = 222.222
            # inc stream counter
            str_count += 1

def initialise_segment(data, sorted_by_ca, str_flg, seg_swac_dic, idx, swac_seg_dic, nss):
    nodes = extract_node_count(data)
    Gs = build_graph(nodes, sorted_by_ca, str_flg, di=False)
    cd = []
    for iseg in range(nss):
        conn = []
        node_swac = seg_swac_dic[iseg + 1]
        downstr = sorted_by_ca[node_swac][idx['downstr']]
        for n in Gs.neighbors(node_swac):
            if n in swac_seg_dic:
                if n == downstr:
                    # do nothing I only want the +ve numbers here
                    pass
                else:
                    if ff.use_natproc:
                        conn.append((swac_seg_dic[n]))
                    else:
                        conn.append((swac_seg_dic[n] - 1))

        # update num connections
        cd.append(conn + [0] * (11 - len(conn)))
    return cd

def combine_runoff_with_area(data, runoff):
    areas = data['params']['node_areas']
    nper = extract_nper(data)
    nodes = extract_node_count(data)
    # units oddness - lots of hardcoded 1000s in input_output.py
    fac = 0.001
    for per in range(nper):
        for node in range(1, nodes + 1):
            i = (nodes * per) + node
            runoff[i] = runoff[i] * areas[node] * fac
    return runoff

##############################################################################

def get_str_nitrate(data, runoff, stream_nitrate_aggregation):
    """integrate flows and nitrate mass in stream cells"""

    cdef:
        double[:,:] stream_conc

    sorted_by_ca = make_sorted_by_ca(data)
    idx = make_idx()
    nss = count_nss(sorted_by_ca)
    swac_seg_dic = make_swac_seg_dic(sorted_by_ca)
    runoff_with_area = combine_runoff_with_area(data, runoff)
    str_flow_array = get_aggregated_sfr_flows(data, nss, sorted_by_ca, runoff_with_area, swac_seg_dic)
    stream_mass_array = get_aggregated_stream_mass(data, nss, sorted_by_ca, stream_nitrate_aggregation, swac_seg_dic)
    stream_conc = _divide_2D_arrays(stream_mass_array, str_flow_array)
    return stream_conc

###############################################################################

def write_sfr(sfr, filename=None):
    """
    Write the package file.

    Returns
    -------
    None

    """

    # tabfiles = False
    # tabfiles_dict = {}
    # transroute = False
    # reachinput = False
    if filename is not None:
        sfr.fn_path = filename

    f_sfr = open(sfr.fn_path, 'w')

    # Item 0 -- header
    f_sfr.write('{0}\n'.format(sfr.heading))

    # Item 1
    if sfr.reachinput:
        """
        When REACHINPUT is specified, variable ISFROPT is read in data set 1c.
        ISFROPT can be used to change the default format for entering reach
        and segment data
        or to specify that unsaturated flow beneath streams will be simulated.
        """
        f_sfr.write('reachinput ')
    if sfr.transroute:
        """When TRANSROUTE is specified, optional variables IRTFLG, NUMTIM,
        WEIGHT, and FLWTOL
        also must be specified in Item 1c.
        """
        f_sfr.write('transroute')
    if sfr.transroute or sfr.reachinput:
        f_sfr.write('\n')
    if sfr.tabfiles:
        """
        tabfiles
        """
        f_sfr.write(
            '{} {} {}\n'.format(sfr.tabfiles, sfr.numtab, sfr.maxval))

    sfr._write_1c(f_sfr)

    # item 2
    sfr._write_reach_data(f_sfr)

    fmt1 = ['{:.0f}'] * 4
    fmt2 = ['{!s}'] * 4

    # items 3 and 4 are skipped (parameters not supported)
    itmpr = range(sfr.dataset_5[0][0])
    cols = ['nseg', 'icalc', 'outseg', 'iupseg', 'flow',
            'runoff', 'etsw', 'pptsw', 'width1', 'depth1']

    for i in range(0, sfr.nper):
        # item 5
        f_sfr.write(' '.join(map(str, sfr.dataset_5[i])) + '\n')
        # Item 6
        for j in itmpr:
            # write datasets 6a, 6b and 6c
            _write_segment_data(sfr, i, j, f_sfr, fmt1, fmt2, cols)
    f_sfr.close()

###############################################################################


def _write_segment_data(sfr, i, j, f_sfr, fmt1, fmt2, cols):

    nseg, icalc, outseg, iupseg, flow, runoff, etsw, pptsw, \
        width1, depth1 = np.array(sfr.segment_data[i])[cols][j]

    f_sfr.write(
        ' '.join(fmt1).format(nseg,
                              icalc,
                              outseg,
                              iupseg)
        + ' ')

    f_sfr.write(
        ' '.join(fmt2).format(flow,
                              runoff,
                              etsw,
                              pptsw)
        + ' ')

    f_sfr.write('\n')

    f_sfr.write('{!s}'.format(width1) + ' ')

    f_sfr.write('{!s}'.format(depth1) + ' ')

    f_sfr.write('\n')

    f_sfr.write('{!s}'.format(width1) + ' ')

    f_sfr.write('{!s}'.format(depth1) + ' ')

    f_sfr.write('\n')

###############################################################################

def get_evt_file(data, evtrate):
    """get EVT object."""

    import os.path

    cdef int i, per, nper, nodes

    # units oddness - lots of hardcoded 1000s in input_output.py
    cdef float fac = 0.001
    path = make_path(data)

    nper = extract_nper(data)
    nodes = extract_node_count(data)
    m = None

    ievtcb = data['params']['ievtcb']
    nevtopt = data['params']['nevtopt']
    evt_params = data['params']['evt_parameters']
    evt_out = None

    ievt = np.zeros((nodes, 1), dtype=int)
    surf = np.zeros((nodes, 1))
    exdp = np.zeros((nodes, 1))
    evtr = np.zeros((nodes, 1))

    for inode, vals in evt_params.iteritems():
        ievt[inode - 1, 0] = vals[0]
        surf[inode - 1, 0] = vals[1]
        exdp[inode - 1, 0] = vals[2]

    evt_dic = {}
    for per in tqdm(range(nper), desc="Generating EVT flux     "):
        for inode in range(1, nodes + 1):
            evtr[inode - 1, 0] = evtrate[(nodes * per) + inode] * fac
        evt_dic[per] = evtr.copy()

    if data['params']['gwmodel_type'] == 'mfusg':
        evt_out = flopy_adaptor.make_mfusg_evt(path, nodes, nper, nevtopt, ievtcb, evt_dic, surf, exdp, ievt)
    elif data['params']['gwmodel_type'] == 'mf6':
        m, spd = flopy_adaptor.make_model_with_disu_and_empty_spd_for_evt_out(path, nper, nodes)

        for per in tqdm(range(nper), desc="Generating MF6 EVT  "):
            for i in range(nodes):
                if ievt[i, 0] > 0:
                    spd[per][i] = ((ievt[i, 0] - 1,),
                                   surf[i, 0],
                                   evt_dic[per][i, 0],
                                   exdp[i, 0],
                                   -999.0, -999.0)

        evt_out = flopy_adaptor.modflow_gwf_evt(m, nodes, spd)

    return evt_out

###############################################################################


def do_swrecharge_mask(data, runoff, recharge):
    if ff.use_natproc:
        return do_swrecharge_mask_natproc(data, runoff, recharge)
    else:
        return do_swrecharge_mask_original(data, runoff, recharge)

###############################################################################


def do_swrecharge_mask_original(data, runoff, recharge):
    """do ror with monthly mask"""
    series, params = data['series'], data['params']
    nnodes = data['params']['num_nodes']

    cdef:
        size_t length = len(series['date'])
        double[:, :] ror_prop = params['ror_prop']
        double[:, :] ror_limit = params['ror_limit']
        long long[:] months = np.array(series['months'], dtype=np.int64)
        size_t zone_ror = params['swrecharge_zone_mapping'][1] - 1
        int day, month

    sorted_by_ca = make_sorted_by_ca(data)

    # 'downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z',
    #         'bed_thk', 'str_k', 'depth', 'width'] # removed hcond1

    # complete graph
    Gc = build_graph(nnodes, sorted_by_ca, np.full((nnodes), 1, dtype='int'))

    def compute_upstream_month_mask(month_number):
        cdef int i = month_number
        cdef int z
        mask = np.full((nnodes), 0, dtype='int')
        for node in range(1, nnodes + 1):
            z = params['swrecharge_zone_mapping'][node] - 1
            fac = ror_prop[i][z]
            lim = ror_limit[i][z]
            if min(fac, lim) > 0.0:
                mask[node-1] = 1
                # add upstream bits
                lst = [n[0] for n in
                       nx.shortest_path_length(Gc, target=node).items()]
                for n in lst:
                    #  for n in nx.ancestors(Gc, node):
                    mask[n-1] = 1
        return build_graph(nnodes, sorted_by_ca, mask)

    # compute monthly mask dictionary
    Gp = {}
    for month in range(12):
        Gp[month] = compute_upstream_month_mask(month)

    # pbar = tqdm(total=range(length))
    for day in tqdm(range(length), desc="Accumulating SW recharge"):
        month = months[day]

        # accumulate flows for today
        acc_flow = get_ror_flows_tree(Gp[month],
                                      runoff, nnodes, day)
        # iterate over nodes relevent to this month's RoR parameters
        for node in list(Gp[month].nodes):
            ro = acc_flow[node - 1]
            if ro > 0.0:
                zone_ror = params['swrecharge_zone_mapping'][node] - 1
                fac_ro = ror_prop[month][zone_ror] * ro
                lim = ror_limit[month][zone_ror]
                qty = min(fac_ro, lim)
                # col_runoff_recharge[day] = qty
                recharge[(nnodes * day) + node] += qty
                runoff[(nnodes * day) + node] -= qty
        # pbar.update(day)
    return runoff, recharge


###############################################################################


def do_swrecharge_mask_natproc(data, runoff, recharge):
    """do ror with monthly mask"""
    series, params = data['series'], data['params']
    nnodes = data['params']['num_nodes']
    areas = data['params']['node_areas']
    catchment = data['params']["reporting_zone_mapping"]
    #  try here
    # cdef double[:] ror = np.zeros(nodes)

    cdef:
        size_t length = len(series['date'])
        double[:, :] ror_prop = params['ror_prop']
        double[:, :] ror_limit = params['ror_limit']
        long long[:] months = np.array(series['months'], dtype=np.int64)
        #double[:] area_array = np.array(areas, dtype=np.float64)
        size_t zone_ror = params['swrecharge_zone_mapping'][1] - 1
        int day, month, node
        double fac = 0.001
        double qty_mmd, qty_m3d

    sorted_by_ca = make_sorted_by_ca(data)

    # 'downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z',
    #         'bed_thk', 'str_k', 'depth', 'width'] # removed hcond1

    for day in tqdm(range(length), desc="Accumulating SW recharge"):
        month = months[day]

        # accumulate flows for today
        acc_flow = get_ror_flows_sfr(sorted_by_ca, runoff, nnodes, day, areas) #, catchment)

        for node, line in sorted_by_ca.items():
            ro = acc_flow[node - 1]
            cat = params["reporting_zone_mapping"][node]
            if ro > 0.0 and cat > 0:
                downstr = line[0]
                zone_ror = params['swrecharge_zone_mapping'][node]
                fac_ro = ror_prop[month][zone_ror - 1] * ro
                lim = ror_limit[month][zone_ror - 1]
                qty_m3d = min(fac_ro, lim)

                if qty_m3d > 0.0:

                    qty_mmd = (qty_m3d / areas[node] / fac)
                    recharge[(nnodes * day) + node] += qty_mmd
                    acc_flow[node - 1] -= qty_m3d

                    # remove qty from accumulated ro at nodes downstream
                    while downstr > 1:
                        acc_flow[downstr - 1] = max(0.0, acc_flow[downstr - 1] - qty_m3d)
                        node_swac = downstr

                        # get new downstr node
                        downstr = sorted_by_ca[node_swac][0]
                        cat = params["reporting_zone_mapping"][node_swac]
                        if cat < 1:
                            break

    return runoff, recharge


###############################################################################

def get_ror_flows_sfr(sorted_by_ca, runoff, nodes, day, areas): #, cat):
    """get flows for one period"""

    cdef int[:] done = np.zeros((nodes), dtype=np.intc)
    cdef double[:] flow = np.zeros(nodes)
    cdef long long node_swac, downstr
    cdef long long c = nodes * day
    cdef double acc
    cdef double fac = 0.001


    for node_swac, line in sorted_by_ca.items():
        # rep_zone = cat[node_swac]
        # if rep_zone > 0:
        downstr = line[0] #, str_flag = line[:2]
        acc = 0.0

        # accumulate pre-stream flows into network
        while downstr > 0: # and rep_zone > 0:

            # not str
            if done[node_swac - 1] < 1:
                acc += max(0.0, runoff[c + node_swac]) * float(areas[node_swac]) * fac
                flow[node_swac - 1] += acc
                done[node_swac - 1] = 1

            else:
                flow[node_swac - 1] += acc
                # acc = 0.0
                # break
            # new node
            node_swac = downstr
            # get new downstr node
            downstr = sorted_by_ca[node_swac][0]
            # rep_zone = cat[node_swac]

    return flow


def get_ror_flows_tree(G, runoff, nodes, day):

    """get total flows for RoR one day with mask"""

    flow = np.zeros((nodes))
    done = np.zeros((nodes), dtype='int')
    c = nodes * day
    leaf_nodes = [x for x in G.nodes()
                  if G.out_degree(x) == 1 and G.in_degree(x) == 0]
    for node_swac in leaf_nodes:
        node = node_swac
        acc = max(0.0, runoff[c + node])

        lst = [nd[0] for nd in nx.shortest_path_length(
            G,
            source=node_swac).items()]
        #  lst = nx.descendants(G, node_swac)
        for d in lst:
            if done[d-1] != 1:
                acc = (flow[node - 1] + max(0.0, runoff[c + d]))
            flow[d - 1] += acc
            node = d
            done[d-1] = 1

    return flow


def build_graph(nnodes, sorted_by_ca, mask, di=True):
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for node in range(1, nnodes + 1):
        if ff.use_natproc:
            if mask[node-1] == 1: #  and sorted_by_ca[node][4] > 0.0:
                G.add_node(node, ca=sorted_by_ca[node][4])
        else:
            if mask[node-1] == 1:
                G.add_node(node)
    for node_swac, line in sorted_by_ca.items():
        if ff.use_natproc:
            downstr = int(line[0])
        else:
            downstr = line[0]
        if downstr > 0:
            if ff.use_natproc:
                if downstr not in G.nodes:
                    G.add_node(downstr, ca=sorted_by_ca[downstr][4])
            else:
                pass
            if mask[node_swac-1] == 1:
                G.add_edge(node_swac, downstr)
    return G


def all_days_mask(data):
    """get overall RoR mask for run"""
    series, params = data['series'], data['params']
    nnodes = data['params']['num_nodes']

    cdef:
        size_t length = len(series['date'])
        double[:, :] ror_prop = params['ror_prop']
        double[:, :] ror_limit = params['ror_limit']
        long long[:] months = np.array(series['months'], dtype=np.int64)
        size_t zone_ror = params['swrecharge_zone_mapping'][1] - 1
        int month

    sorted_by_ca = make_sorted_by_ca(data)

    mask = np.full((nnodes), 0, dtype='int')

    # complete graph
    Gc = build_graph(nnodes, sorted_by_ca, np.full((nnodes), 1, dtype='int'))

    for day in range(length):

        month = months[day]

        for node in range(1, nnodes + 1):
            zone_ror = params['swrecharge_zone_mapping'][node] - 1
            fac = ror_prop[month][zone_ror]
            lim = ror_limit[month][zone_ror]
            if fac != 0.0 or lim != 0.0:
                mask[node-1] = 1

    # do downstream from RoR areas as flows will be different
    for node in range(1, nnodes + 1):
        if mask[node-1] == 1:
            lst = [n[0] for n in nx.shortest_path_length(Gc,
                                                         source=node).items()]
            #  for n in nx.descendants(Gc, node):
            for n in lst:
                mask[n-1] = 1

    return build_graph(nnodes, sorted_by_ca, mask)
