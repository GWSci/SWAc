# -*- coding: utf-8 -*-
# cython: language_level=2

"""SWAcMod model functions in Cython."""

# Third Party Libraries
import numpy as np
cimport numpy as np
from collections import OrderedDict

# Internal modules
from . import utils as u
from tqdm import tqdm
import networkx as nx
import sys


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
    for num in range(1, length):
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
    mac_opt = params['macropore_activation_option']

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
        size_t zone_ror = params['swrecharge_zone_mapping'][node] - 1
        size_t zone_rro = params['rapid_runoff_zone_mapping'][node] - 1
        double ssmd = u.weighted_sum(params['soil_spatial'][node],
                                     s_smd['starting_SMD'])
        long long [:] class_smd = np.array(rrp[zone_rro]['class_smd'],
                                           dtype=np.int64)
        long long [:] class_ri = np.array(rrp[zone_rro]['class_ri'],
                                          dtype=np.int64)
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
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
        double ma = 0.0

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

        if params['swrecharge_process'] == 'enabled':
            col_runoff_recharge[num] = 0.0

        if params['macropore_process'] == 'enabled':
            if mac_opt == 'SMD':
                var8a = var2 - col_rapid_runoff[num]
                ma = macro_act[var6][zone_mac]
            else:
                var8a = var2 - col_rapid_runoff[num] - macro_act[var6][zone_mac]
                ma = sys.float_info.max
            if var8a > 0:
                if p_smd < ma:
                    var9 = macro_prop[var6][zone_mac] * var8a
                    var10 = macro_limit[var6][zone_mac]
                    macropore = (var10 if var9 > var10 else var9)
                else:
                    macropore = 0.0
            else:
                macropore = 0.0
                
            var10a = macro_rec[var6][zone_mac]
            col_macropore_att[num] = macropore * (1 - var10a)
            col_macropore_dir[num] = macropore * var10a

        percol_in_root = (var2 - col_rapid_runoff[num] - col_macropore_att[num])
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

        for num in range(1, length):
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
        double [:] macropore_dir = output['macropore_dir']
        rep_zone = data['params']['reporting_zone_mapping'][node]
        size_t num
        double var1, var2

    for num in range(length):
        if params['recharge_attenuation_process'] == 'enabled':
            if num == 0:
                recharge = irs
            else:
                # wittw try
                recharge = (recharge_store_input[num - 1] +
                            col_recharge_store[num - 1] -
                            col_combined_recharge[num - 1] +
                            macropore_dir[num -1])

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

def get_rch_file(data, rchrate):
    """get mf6 RCH object."""

    import flopy
    import csv
    import numpy as np
    import copy
    import os.path
    from swacmod.input_output import print_progress

    cdef int i, per

    # this is equivalent of strange hardcoded 1000 in format_recharge_row
    #  which is called in the mf6 output function
    fac = 0.001
    areas = data['params']['node_areas']
    fileout = data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    rch_params = data['params']['recharge_node_mapping']
    nper = len(data['params']['time_periods'])
    nodes = data['params']['num_nodes']

    if data['params']['gwmodel_type'] == 'mfusg':
        # not used
        m = flopy.modflow.Modflow(modelname=path,
                                  version='mfusg',
                                  structured=True)
        dis = flopy.modflow.ModflowDis(m, nrow=nodes, ncol=1, nper=nper)
    elif data['params']['gwmodel_type'] == 'mf6':
        sim = flopy.mf6.MFSimulation()
        m = flopy.mf6.mfmodel.MFModel(sim,
                                       modelname=path)
        njag = nodes + 2
        lenx = int((njag/2) - (nodes/2))
        dis = flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(m,
                                                         nodes=nodes,
                                                         ja=np.zeros((njag), dtype=int),
                                                         nja=njag, area=1.0)
        flopy.mf6.modflow.mftdis.ModflowTdis(sim,
                                             loading_package=False,
                                             time_units=None,
                                             start_date_time=None,
                                             nper=nper,
                                             filename=None,
                                             pname=None,
                                             parent_file=None)
        irch = np.zeros((nodes, 1), dtype=int)
        if rch_params is not None:
            for inode, vals in rch_params.iteritems():
                irch[inode - 1, 0] = vals[0]
        else:
            for i in range(nodes):
                irch[i - 1, 0] = i

    if data['params']['gwmodel_type'] == 'mfusg':
        # not used
        # evt_out = flopy.modflow.ModflowEvt(m, nevtop=nevtopt,
        #                                    ipakcb=ievtcb,
        #                                    evtr=evt_dic,
        #                                    surf={0: surf},
        #                                    exdp={0: exdp},
        #                                    ievt={0: ievt})
        pass
    elif data['params']['gwmodel_type'] == 'mf6':
        spd = flopy.mf6.ModflowGwfrch.stress_period_data.empty(m,
                                                               maxbound=nodes,
                                                               nseg=1,
                                                               stress_periods=range(nper))

        for per in tqdm(range(nper), desc="Generating MF6 RCH  "):
            for i in range(nodes):
                if irch[i, 0] > 0:
                    spd[per][i] = ((irch[i, 0] -1,), rchrate[(nodes * per) + i + 1] * fac)

        rch_out = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(m,
                                                           fixed_cell=False,
                                                           print_input=None,
                                                           print_flows=None,
                                                           save_flows=None,
                                                           timeseries=None,
                                                           observations=None,
                                                           maxbound=nodes,
                                                           stress_period_data=spd,
                                                           filename=None,
                                                           pname=None,
                                                           parent_file=None)
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
        double [:] col_attenuation = np.zeros(length)
        double [:] col_combined_str = np.zeros(length)
        double rlp = params['sw_params'][node][1]
        double base = (params['sw_params'][node][0] +
                       output['interflow_to_rivers'][0] +
                       output['swabs_ts'][0] +
                       output['swdis_ts'][0] +
                       output['rapid_runoff'][0] -
                       output['runoff_recharge'][0])
        size_t num

    if params['sw_process'] == 'enabled':
        # don't attenuate negative flows
        if base < 0.0:
            rlp = 1.0
        col_combined_str[0] = rlp * base
        col_attenuation[0] = base - col_combined_str[0]
        for num in range(1, length):
            base = (col_attenuation[num-1] +
                    output['interflow_to_rivers'][num] +
                    output['swabs_ts'][num] +
                    output['swdis_ts'][num] +
                    output['rapid_runoff'][num] -
                    output['runoff_recharge'][num] +
                    output['rejected_recharge'][num])
            # don't attenuate negative flows
            if base < 0.0:
                rlp = 1.0
            else:
                rlp = params['sw_params'][node][1]
            col_combined_str[num] = rlp * base
            col_attenuation[num] = base - col_combined_str[num]
    else:
        col_combined_str = (output['interflow_to_rivers'] +
                            output['swabs_ts'] +
                            output['swdis_ts'] +
                            output['rapid_runoff'] -
                            output['runoff_recharge'] +
                            output['rejected_recharge'])

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
    for num in range(1, length):
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

    if index is not None:
        not_scalar = (type(index[0]) is range or type(index[0]) is list)
    else:
        not_scalar = False

    for ik, key in enumerate(output):
        new_rep[key] = []
        if not_scalar:
            new_rep[key] = [output[key][i].mean(dtype=np.float64)
                            * area for i in index]
        elif index is not None:
            new_rep[key] = [output[key][index[0]] * area]
        else:
            new_rep[key] = output[key] * area
        if reporting:
            new_rep[key] += reporting[key]
    return new_rep


###############################################################################
def get_sfr_flows(sorted_by_ca, idx, runoff, done, areas, swac_seg_dic, ro,
                  flow, nodes_per):
    
    """get flows for one period"""

    ro[:] = 0.0
    flow[:] = 0.0
    
    for node_swac, line in sorted_by_ca.items():
        downstr, str_flag = line[:2]
        acc = 0.0

        # accumulate pre-stream flows into network
        while downstr > 1:

            str_flag = sorted_by_ca[node_swac][1] #[idx['str_flag']]

            # not str
            if str_flag < 1: # or node_mf < 1:
                # not not done
                if done[node_swac - 1] < 1:
                    acc += runoff[nodes_per + node_swac]
                    done[node_swac - 1] = 1
            else:
                # stream cell
                iseg = swac_seg_dic[node_swac]

                # not done
                if done[node_swac - 1] < 1:
                    ro[iseg - 1] = runoff[nodes_per + node_swac]
                    flow[iseg - 1] = acc
                    done[node_swac - 1] = 1
                    acc = 0.0

                # stream cell been done
                else:
                    flow[iseg - 1] += acc
                    acc = 0.0
                    break

            # new node
            node_swac = downstr
            # get new downstr node
            downstr = sorted_by_ca[node_swac][0] #[idx['downstr']]

    return ro, flow

###############################################################################

def get_sfr_file(data, runoff):
    """get SFR object."""
    
    import flopy
    import csv
    import numpy as np
    import copy
    import os.path
    from swacmod.input_output import print_progress
    
    # units oddness - lots of hardcoded 1000s in input_output.py
    fac = 0.001
    
    areas = data['params']['node_areas']
    fileout = data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)

    nper = len(data['params']['time_periods'])
    nodes = data['params']['num_nodes']

    njag = nodes + 2
    lenx = int((njag/2) - (nodes/2))

    sorted_by_ca = OrderedDict(sorted(data['params']['routing_topology'].items(),
                                      key=lambda x: x[1][4]))

    names = ['downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z',
             'bed_thk', 'str_k', 'depth', 'width'] # removed hcond1

    idx = dict((y, x) for (x, y) in enumerate(names))

    nstrm = nss = sum(value[idx['str_flag']] > 0 for value in sorted_by_ca.values())

    istcb1, istcb2 = data['params']['istcb1'], data['params']['istcb2']
    
    if data['params']['gwmodel_type'] == 'mfusg':
        m = flopy.modflow.Modflow(modelname=path,
                                  version='mfusg', structured=False)

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
        sd = flopy.modflow.ModflowSfr2.get_empty_segment_data(nss)
        rd = flopy.modflow.ModflowSfr2.get_empty_reach_data(nstrm,
                                                            structured=False)

    elif data['params']['gwmodel_type'] == 'mf6':
        sim = flopy.mf6.MFSimulation()
        m = flopy.mf6.mfmodel.MFModel(sim,
                                       modelname=path)
        njag = nodes + 2
        lenx = int((njag/2) - (nodes/2))
        dis = flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(m,
                                                         nodes=nodes,
                                                         ja=np.zeros((njag), dtype=int),
                                                         nja=njag)
        flopy.mf6.modflow.mftdis.ModflowTdis(sim,
                                             loading_package=False,
                                             time_units=None,
                                             start_date_time=None,
                                             nper=nper,
                                             filename=None,
                                             pname=None,
                                             parent_file=None)
        cd = []
        rd = []
        sd = {}
        sd[0] = []
    seg_data = {}
    swac_seg_dic = {}
    seg_swac_dic = {}
    done = np.zeros((nodes), dtype=int)
    # for mf6 only
    str_flg = np.zeros((nodes), dtype=int)
    
    # initialise reach & segment data
    str_count = 0
    for node_swac, line in sorted_by_ca.items():
        (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, # hcond1,
         depth, width) = line
        # for mf6 only
        str_flg[node_swac-1] = str_flag
        if str_flag > 0: # and node_mf > 0:
            swac_seg_dic[node_swac] = str_count + 1
            seg_swac_dic[str_count + 1] = node_swac

            if data['params']['gwmodel_type'] == 'mfusg':
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
                # sd[str_count]['hcond1'] = hcond1 # get from lpf
                sd[str_count]['thickm1'] = bed_thk # constant
                sd[str_count]['elevup'] = z # get from mf
                sd[str_count]['width1'] = width # constant
                sd[str_count]['depth1'] = depth # constant
                sd[str_count]['width2'] = width # constant
                sd[str_count]['depth2'] = depth # constant

            elif data['params']['gwmodel_type'] == 'mf6':
                if node_mf > 0:
                    n = (node_mf - 1,)
                else:
                    n = (-100000000, )

                rd.append([str_count, n, length, width,
                           0.0001, z, bed_thk, str_k, 0.0001, 1, 1.0, 0])
                sd[0].append((str_count, 'STAGE', z + depth))
                sd[0].append((str_count, 'STATUS', "SIMPLE"))

            # inc stream counter
            str_count += 1

    if data['params']['gwmodel_type'] == 'mfusg':
        for iseg in range(nss):
            node_swac = seg_swac_dic[iseg + 1]
            downstr = sorted_by_ca[node_swac][idx['downstr']]
            if downstr in swac_seg_dic:
                sd[iseg]['outseg'] = swac_seg_dic[downstr]
            else:
                sd[iseg]['outseg'] = 0

    elif data['params']['gwmodel_type'] == 'mf6':
        Gs = build_graph(nodes, sorted_by_ca, str_flg, di=False)
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

            # update num connections
            cd.append(conn)
            rd[iseg][9] = len(cd[iseg]) - 1

    for per in range(nper):
        for node in range(1, nodes + 1):
            i = (nodes * per) + node
            runoff[i] = runoff[i] * areas[node] * fac
            
    ro, flow = np.zeros((nss)), np.zeros((nss))

    # populate runoff and flow
    for per in tqdm(range(nper), desc="Accumulating SFR flows  "):

        ro, flow = get_sfr_flows(sorted_by_ca, idx, runoff, done, areas,
                                 swac_seg_dic, ro, flow, nodes * per)

        if data['params']['gwmodel_type'] == 'mfusg':
            for iseg in range(nss):
                sd[iseg]['runoff'] = ro[iseg]
                sd[iseg]['flow'] = flow[iseg]

        elif data['params']['gwmodel_type'] == 'mf6':
            for iseg in range(nss):
                if per not in sd:
                    sd[per] = []
                sd[per].append((iseg, 'RUNOFF', ro[iseg]))
                sd[per].append((iseg, 'INFLOW', flow[iseg]))

        # add segment data for this period
        if data['params']['gwmodel_type'] == 'mfusg':
            seg_data[per] = copy.deepcopy(sd)
        done[:] = 0
            
    isfropt = 1
    if data['params']['gwmodel_type'] == 'mfusg':
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

    elif data['params']['gwmodel_type'] == 'mf6':
        sfr = flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr(m,
                                                       loading_package=False,
                                                       auxiliary=None,
                                                       boundnames=None,
                                                       print_input=None,
                                                       print_stage=None,
                                                       print_flows=None,
                                                       save_flows=None,
                                                       stage_filerecord=None,
                                                       budget_filerecord=None,
                                                       timeseries=None,
                                                       observations=None,
                                                       mover=None,
                                                       maximum_iterations=None,
                                                       maximum_depth_change=None,
                                                       unit_conversion=86400.0,
                                                       nreaches=nss,
                                                       packagedata=rd,
                                                       connectiondata=cd,
                                                       diversions=None,
                                                       perioddata=sd,
                                                       filename=None,
                                                       pname=None,
                                                       parent_file=None)


    sfr.heading = "# SFR package for  MODFLOW-USG, generated by SWAcMod."

    # compute the slopes
    if data['params']['gwmodel_type'] == 'mfusg':
        m.sfr.get_slopes()

    return sfr

##############################################################################

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
        ISFROPT can be used to change the default format for entering reach and segment data
        or to specify that unsaturated flow beneath streams will be simulated.
        """
        f_sfr.write('reachinput ')
    if sfr.transroute:
        """When TRANSROUTE is specified, optional variables IRTFLG, NUMTIM, WEIGHT, and FLWTOL
        also must be specified in Item 1c.
        """
        f_sfr.write('transroute')
    if sfr.transroute or sfr.reachinput:
        f_sfr.write('\n')
    if sfr.tabfiles:
        """
        tabfiles
        An optional character variable that is a flag to indicate that inflows to one or more stream
        segments will be specified with tabular inflow files.
        numtab
        An integer value equal to the number of tabular inflow files that will be read if TABFILES
        is specified. A separate input file is required for each segment that receives specified inflow.
        Thus, the maximum value of NUMTAB that can be specified is equal to the total number of
        segments specified in Item 1c with variables NSS. The name (Fname) and unit number (Nunit)
        of each tabular file must be specified in the MODFLOW-2005 Name File using tile type (Ftype) DATA.
        maxval

        """
        f_sfr.write(
            '{} {} {}\n'.format(sfr.tabfiles, sfr.numtab, sfr.maxval))

    sfr._write_1c(f_sfr)

    # item 2
    sfr._write_reach_data(f_sfr)

    fmt1 = ['{:.0f}'] * 4
    fmt2 = ['{!s}'] * 4
    
    # items 3 and 4 are skipped (parameters not supported)
    import time
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


def get_swdis(data, output, node):
    """Surface water discharges"""

    from swacmod.utils import monthdelta, weekdelta
    
    series, params = data['series'], data['params']
    areas = data['params']['node_areas']
    fac = 1000.0
    freq_flag = data['params']['swdis_f']
    dates = series['date']

    swdis_ts = np.zeros(len(series['date']))
    start_date = dates[0]
    
    if node in params['swdis_locs']:
        area = areas[node]
        zone_swdis = params['swdis_locs'][node] - 1

        if freq_flag == 0:
            # daily input just populate
            swdis_ts = series['swdis_ts'][:, zone_swdis] / area * fac
        elif freq_flag == 1:
            # if weeks convert to days
            for iday, day in enumerate(dates):
                week = weekdelta(start_date, day)
                swdis_ts[iday] = series['swdis_ts'][week, zone_swdis] / area * fac
        elif freq_flag == 2:
            # if months convert to days
            for iday, day in enumerate(dates):    
                month = monthdelta(start_date, day)
                swdis_ts[iday] = series['swdis_ts'][month, zone_swdis] / area * fac

    return {'swdis_ts': swdis_ts}

###############################################################################


def get_swabs(data, output, node):
    """Surface water abtractions"""

    from swacmod.utils import monthdelta, weekdelta
    
    series, params = data['series'], data['params']
    areas = data['params']['node_areas']
    fac = 1000.0
    freq_flag = data['params']['swabs_f']
    dates = series['date']

    swabs_ts = np.zeros(len(series['date']))
    start_date = dates[0]

    if node in params['swabs_locs']:
        area = areas[node]
        zone_swabs = params['swabs_locs'][node] - 1

        if freq_flag == 0:
            # daily input just populate
            swabs_ts = series['swabs_ts'][:, zone_swabs] / area * fac
            
        elif freq_flag == 1:
            # if weeks convert to days
            for iday, day in enumerate(dates):
                week = weekdelta(start_date, day)
                swabs_ts[iday] = series['swabs_ts'][week, zone_swabs] / area * fac

        elif freq_flag == 2:
            # if months convert to days
            for iday, day in enumerate(dates):
                month = monthdelta(start_date, day)
                swabs_ts[iday] = series['swabs_ts'][month, zone_swabs] / area * fac

    return {'swabs_ts': swabs_ts}


###############################################################################


def get_evt_file(data, evtrate):
    """get EVT object."""

    import flopy
    import csv
    import numpy as np
    import copy
    import os.path
    from swacmod.input_output import print_progress

    cdef int i, per, nper, nodes

    # units oddness - lots of hardcoded 1000s in input_output.py
    cdef float fac = 0.001
    areas = data['params']['node_areas']
    fileout = data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)

    nper = len(data['params']['time_periods'])
    nodes = data['params']['num_nodes']

    if data['params']['gwmodel_type'] == 'mfusg':
        m = flopy.modflow.Modflow(modelname=path,
                                  version='mfusg',
                                  structured=True)
        dis = flopy.modflow.ModflowDis(m, nrow=nodes, ncol=1, nper=nper)
    elif data['params']['gwmodel_type'] == 'mf6':
        sim = flopy.mf6.MFSimulation()
        m = flopy.mf6.mfmodel.MFModel(sim,
                                       modelname=path)
        njag = nodes + 2
        lenx = int((njag/2) - (nodes/2))
        dis = flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(m,
                                                         nodes=nodes,
                                                         ja=np.zeros((njag), dtype=int),
                                                         nja=njag)
        flopy.mf6.modflow.mftdis.ModflowTdis(sim,
                                             loading_package=False,
                                             time_units=None,
                                             start_date_time=None,
                                             nper=nper,
                                             filename=None,
                                             pname=None,
                                             parent_file=None)

    ievtcb = data['params']['ievtcb']
    nevtopt = data['params']['nevtopt']
    evt_params = data['params']['evt_parameters']

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
        evt_out = flopy.modflow.ModflowEvt(m, nevtop=nevtopt,
                                           ipakcb=ievtcb,
                                           evtr=evt_dic,
                                           surf={0: surf},
                                           exdp={0: exdp},
                                           ievt={0: ievt})
    elif data['params']['gwmodel_type'] == 'mf6':
        spd = flopy.mf6.ModflowGwfevt.stress_period_data.empty(m,
                                                               maxbound=nodes,
                                                               nseg=1,
                                                               stress_periods=range(nper))

        for per in tqdm(range(nper), desc="Generating MF6 EVT  "):
            for i in range(nodes):
                if ievt[i, 0] > 0:
                    spd[per][i] = ((ievt[i, 0] -1,),
                                   surf[i, 0],
                                   evt_dic[per][i, 0],
                                   exdp[i, 0],
                                   -999.9)

        evt_out = flopy.mf6.modflow.mfgwfevt.ModflowGwfevt(m,
                                                           fixed_cell=False,
                                                           print_input=None,
                                                           print_flows=None,
                                                           save_flows=None,
                                                           timeseries=None,
                                                           observations=None,
                                                           surf_rate_specified=False,
                                                           maxbound=nodes,
                                                           nseg=1,
                                                           stress_period_data=spd,
                                                           filename=None,
                                                           pname=None,
                                                           parent_file=None)

    return evt_out

###############################################################################


def do_swrecharge_mask(data, runoff, recharge):
    """do ror with monthly mask"""
    series, params = data['series'], data['params']
    nnodes = data['params']['num_nodes']
    cdef:
        size_t length = len(series['date'])
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
        long long [:] months = np.array(series['months'], dtype=np.int64)
        size_t zone_ror = params['swrecharge_zone_mapping'][1] - 1

    sorted_by_ca = OrderedDict(sorted(data['params']['routing_topology'].items(),
                                      key=lambda x: x[1][4]))
    
    names = ['downstr', 'str_flag', 'node_mf', 'length', 'ca', 'z',
             'bed_thk', 'str_k', 'depth', 'width'] # removed hcond1

    # complete graph
    Gc = build_graph(nnodes, sorted_by_ca, np.full((nnodes), 1, dtype='int'))

    def compute_upstream_month_mask(month_num):
        mask = np.full((nnodes), 0, dtype='int')
        for node in range(1, nnodes + 1):
            zone_ror = params['swrecharge_zone_mapping'][node] - 1
            fac = ror_prop[month_num][zone_ror]
            lim = ror_limit[month_num][zone_ror]
            if min(fac, lim) > 0.0:
                mask[node-1] = 1
                # add upstream bits
                lst = [n for n, d in nx.shortest_path_length(Gc, target=node).items()]
                for n in lst:
                # for n in nx.ancestors(Gc, node):
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
            
            ro = acc_flow[node -1]

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


def get_ror_flows_tree(G, runoff, nodes, day):
    
    """get total flows for RoR one day with mask"""
    
    flow = np.zeros((nodes))
    done = np.zeros((nodes), dtype='int')
    c = nodes * day
    leaf_nodes = [x for x in G.nodes()
                  if G.out_degree(x) == 1 and G.in_degree(x) == 0]
    for node_swac in leaf_nodes:
        node = node_swac
        acc = runoff[c + node]

        lst = [n for n, d in nx.shortest_path_length(G,
                                                     source=node_swac).items()]
        #lst = nx.descendants(G, node_swac)
        for i, d in enumerate(lst):
            if done[d-1] != 1:
                acc = (flow[node -1] + runoff[c + d])
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
        if mask[node-1] == 1:
            G.add_node(node)
    for node_swac, line in sorted_by_ca.items():
        if mask[node_swac-1] == 1 and line[0] > 0:
            G.add_edge(node_swac, line[0])
    return G


def all_days_mask(data):
    """get overall RoR mask for run"""
    series, params = data['series'], data['params']
    nnodes = data['params']['num_nodes']
    cdef:
        size_t length = len(series['date'])
        double [:, :] ror_prop = params['ror_prop']
        double [:, :] ror_limit = params['ror_limit']
        long long [:] months = np.array(series['months'], dtype=np.int64)
        size_t zone_ror = params['swrecharge_zone_mapping'][1] - 1

    sorted_by_ca = OrderedDict(sorted(data['params']['routing_topology'].items(),
                                      key=lambda x: x[1][4]))

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
            lst = [n for n, d in nx.shortest_path_length(Gc,
                                                         source=node).items()]
            #for n in nx.descendants(Gc, node):
            for n in lst:
                mask[n-1] = 1

    return build_graph(nnodes, sorted_by_ca, mask)
