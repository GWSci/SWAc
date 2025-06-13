#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import logging
import multiprocessing

# Internal modules
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.checks as c
from swacmod.input_files.parsed_input_data import ParsedInputData

def val_time_periods(data, name):
    c.check_values_limits(
        values=[i for j in (data["params"][name]) for i in j],
        name=name,
        low_l=0,
        high_l=len(data["series"]["date"]) + 1,
        include_high=True,
    )

    all_days = []
    for time_range in data["params"][name]:
        all_days += range(time_range[0], time_range[1])

    if set(all_days) != set(range(1, len(data["series"]["date"]) + 1)):
        msg = (
            'Parameter "%s" requires all days to be included'
            " in one (and only one) of the periods"
        )
        raise u.ValidationError(msg % name)

def val_rainfall_ts(data, name):
    rts = data["series"][name]
    rzn = data["params"]["rainfall_zone_names"]
    c.check_type(
        param=rts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(rzn)],
    )

def val_pe_ts(data, name):
    pts = data["series"][name]
    pzn = data["params"]["pe_zone_names"]
    c.check_type(
        param=pts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(pzn)],
    )

def val_temperature_ts(data, name):
    tts = data["series"][name]
    tzn = set(data["params"]["temperature_zone_mapping"].values())
    c.check_type(
        param=tts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_tmax_c_ts(data, name):
    tts = data["series"][name]
    tzn = set(data["params"]["tmax_c_zone_mapping"].values())
    c.check_type(
        param=tts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_tmin_c_ts(data, name):
    tts = data["series"][name]
    tzn = set(data["params"]["tmin_c_zone_mapping"].values())
    c.check_type(
        param=tts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_windsp_ts(data, name):
    tts = data["series"][name]
    tzn = set(data["params"]["windsp_zone_mapping"].values())
    c.check_type(
        param=tts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_subroot_leakage_ts(data, name):
    pass
    # TODO: Validation needs to be done post-finalization
    sts = data["series"][name]
    szn = data["params"]["subroot_zone_names"]

    c.check_type(
        param=sts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(szn)],
    )

def val_swdis_ts(data, name):
    pass
    # TODO: Validation needs to be done post-finalization
    from swacmod.utils import monthdelta, weekdelta

    swdists = data["series"][name]

    swdisn = data["params"]["swdis_locs"]
    dates = data["series"]["date"]

    freq_flag = data["params"]["swdis_f"]
    ndays = len(dates)
    nweeks = weekdelta(dates[0], dates[-1]) + 1
    nmonths = monthdelta(dates[0], dates[-1]) + 1

    length = [ndays, nweeks, nmonths]
    if swdisn != {0: 0}:
        c.check_type(
            param=swdists,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[length[freq_flag], len(swdisn)],
        )

def val_swabs_ts(data, name):
    pass
    # TODO: Validation needs to be done post-finalization
    # from swacmod.utils import monthdelta, weekdelta

    # swabsts = data["series"][name]
    # swabsn = data["params"]["swabs_locs"]
    # dates = data["series"]["date"]

    # freq_flag = data["params"]["swabs_f"]
    # ndays = len(dates)
    # nweeks = weekdelta(dates[0], dates[-1]) + 1
    # nmonths = monthdelta(dates[0], dates[-1]) + 1

    # length = [ndays, nweeks, nmonths]

    # if swabsn != {0: 0}:
    #     c.check_type(
    #         param=swabsts,
    #         name=name,
    #         t_types=data["specs"][name]["type"],
    #         len_list=[length[freq_flag], len(swabsn)],
    #     )

def val_swdis_locs(data, name):
    swdisl = data["params"][name]
    swdisn = data["params"]["swdis_locs"]
    tot = len(data["params"]["swdis_locs"]) + 1

    if swdisn != {0: 0}:
        c.check_type(
            param=swdisl, name=name, t_types=data["specs"][name]["type"],
            keys=range(1, tot)
        )
        c.validate_min_inclusive(swdisl.values(), "zone in %s" % name, 1)
        c.validate_max_inclusive(swdisl.values(), "zone in %s" % name, tot)
        c.validate_min_inclusive(swdisl.keys(), "node in %s" % name, 1)
        c.validate_max_inclusive(swdisl.keys(), "node in %s" % name, data["params"]["num_nodes"])

def val_swabs_locs(data, name):
    swabsl = data["params"][name]
    swabsn = data["params"]["swabs_locs"]
    if swabsn != {0: 0}:
        tot = len(data["params"]["swabs_locs"]) + 1
        c.check_type(
            param=swabsl, name=name, t_types=data["specs"][name]["type"],
            keys=range(1, tot)
        )
        c.validate_min_inclusive(swabsl.values(), "zone in %s" % name, 1)
        c.validate_max_inclusive(swabsl.values(), "zone in %s" % name, tot)
        c.validate_min_inclusive(swabsl.keys(), "node in %s" % name, 1)
        c.validate_max_inclusive(swabsl.keys(), "node in %s" % name, data["params"]["num_nodes"])

def val_node_areas(data, name):
    c.check_type(
        param=data["params"][name],
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, data["params"]["num_nodes"] + 1),
    )
    c.validate_min_inclusive(data["params"][name].values(), name, 0)

def val_reporting_zone_mapping(data, name):
    tot_name = "num_nodes"
    zone_name = "reporting_zone_names"
    rzm = data["params"][name]
    tot = data["params"][tot_name]
    rzn = data["params"][zone_name]
    c.check_type(
        param=rzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(rzm.values(), "zone in %s" % name, 0)
    c.validate_max_inclusive(rzm.values(), "zone in %s" % name, len(rzn))

def val_rainfall_zone_mapping(data, name):
    tot_name = "num_nodes"
    zone_name = "rainfall_zone_names"
    rzm = data["params"][name]
    tot = data["params"][tot_name]
    rzn = data["params"][zone_name]
    c.check_type(
        param=rzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive([i[0] for i in rzm.values()], "zone in %s" % name, 1)
    c.validate_max_inclusive([i[0] for i in rzm.values()], "zone in %s" % name, len(rzn))

def val_pe_zone_mapping(data, name):
    pzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    pzn = data["params"]["pe_zone_names"]
    c.check_type(
        param=pzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive([i[0] for i in pzm.values()], "zone in %s" % name, 0)
    c.validate_max_inclusive([i[0] for i in pzm.values()], "zone in %s" % name, len(pzn))

def val_tmax_c_zone_mapping(data, name):
    tzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"]["tmax_c_zone_names"]
    c.check_type(
        param=tzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(tzm.values(), name, 0)
    c.validate_max_inclusive(tzm.values(), name, len(tzn))

def val_tmin_c_zone_mapping(data, name):
    tzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"]["tmin_c_zone_names"]
    c.check_type(
        param=tzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(tzm.values(), name, 0)
    c.validate_max_inclusive(tzm.values(), name, len(tzn))

def val_windsp_zone_mapping(data, name):
    tzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"]["tmin_c_zone_names"]
    c.check_type(
        param=tzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(tzm.values(), name, 0)
    c.validate_max_inclusive(tzm.values(), name, len(tzn))

def val_temperature_zone_mapping(data, name):
    tzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"]["temperature_zone_names"]
    c.check_type(
        param=tzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(tzm.values(), name, 0)
    c.validate_max_inclusive(tzm.values(), name, len(tzn))

def val_subroot_zone_mapping(data, name):
    szm = data["params"][name]
    tot = data["params"]["num_nodes"]
    szn = data["params"]["subroot_zone_names"]
    c.check_type(
        param=szm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive([i[0] for i in szm.values()], "zone in %s" % name, 0)
    c.validate_max_inclusive([i[0] for i in szm.values()], "zone in %s" % name, len(szn))

def val_rapid_runoff_zone_mapping(data, name):
    rrzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["rapid_runoff_zone_names"]
    c.check_type(
        param=rrzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(rrzm.values(), name, 0)
    c.validate_max_inclusive(rrzm.values(), name, len(rzn))

def val_interflow_zone_mapping(data, name):
    rrzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["interflow_zone_names"]

    c.check_type(
        param=rrzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(rrzm.values(), name, 0)
    c.validate_max_inclusive(rrzm.values(), name, len(rzn))

def val_swrecharge_zone_mapping(data, name):
    rorzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["swrecharge_zone_names"]
    c.check_type(
        param=rorzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.validate_min_inclusive(rorzm.values(), name, 0)
    c.validate_max_inclusive(rorzm.values(), name, len(rzn))

def val_single_cell_swrecharge_zone_mapping(data, name):
    rorzm = data['params'][name]
    tot = data['params']['num_nodes']
    rzn = data['params']['single_cell_swrecharge_zone_names']
    c.check_type(param=rorzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))
    c.validate_min_inclusive(rorzm.values(), name, 0)
    c.validate_max_inclusive(rorzm.values(), name, len(rzn))

def val_macropore_zone_mapping(data, name):
    mzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    mzn = data["params"]["macropore_zone_names"]
    c.check_type(
        param=mzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(mzm.values(), name, 0)
    c.validate_max_inclusive(mzm.values(), name, len(mzn))

def val_macropore_activation_option(data, name):
    x = data["params"][name]
    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])
    c.validate_constraints([x], name, data["specs"][name]["constraints"])

def val_canopy_zone_mapping(data, name):
    rrzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["canopy_zone_names"]
    c.check_type(
        param=rrzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(rrzm.values(), name, 0)
    c.validate_max_inclusive(rrzm.values(), name, len(rzn))

def val_free_throughfall(data, name):
    fth = data["params"][name]
    tot = len(data["params"]["canopy_zone_names"])
    c.check_type(
        param=fth,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(fth.values(), name, 0)
    c.validate_max_inclusive(fth.values(), name, 1.0)

def val_max_canopy_storage(data, name):
    mcs = data["params"][name]
    tot = len(data["params"]["canopy_zone_names"])
    c.check_type(
        param=mcs,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(mcs.values(), name, 0)

def val_snow_params_simple(data, name):
    if data["params"]["snow_process_simple"] == "disabled":
        return
    snp = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=snp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[3],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive([i[0] for i in snp.values()], "starting_snow_pack in %s" % name, 0)

def val_snow_params_complex(data, name):
    if data["params"]["snow_process_complex"] == "disabled":
        return
    snp = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=snp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[10],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive([i[0] for i in snp.values()], "starting_snow_pack in %s" % name, 0)

def val_rapid_runoff_params(data, name):
    rrp = data["params"][name]
    rzn = data["params"]["rapid_runoff_zone_names"]
    c.check_type(
        param=rrp, name=name, t_types=data["specs"][name]["type"],
        len_list=[len(rzn)]
    )
    keys = ["class_smd", "class_ri", "values"]
    for zone in rrp:
        c.check_type(param=zone, name=name, t_types=[dict], keys=keys)
        c.check_type(
            param=zone["values"],
            name=name,
            t_types=[list, list],
            len_list=[len(zone["class_ri"]), len(zone["class_smd"])],
        )
        c.validate_min_inclusive([i for j in zone["values"] for i in j], '"values" in "%s"' % name, 0)
        c.validate_max_inclusive([i for j in zone["values"] for i in j], '"values" in "%s"' % name, 1)

def val_rorecharge_process(data, name):
    rop = data['params'][name]
    rrp = data['params']['rapid_runoff_process']
    if rop == 'enabled' and rrp == 'disabled':
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        raise u.ValidationError(msg % (name, 'rapid_runoff_process'))

def val_single_cell_swrecharge_proportion(data, name):
    rrp = data['params'][name]
    rzn = data['params']['single_cell_swrecharge_zone_names']
    c.check_type(param=rrp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)],
                 keys=range(1, 13))

    c.validate_min_inclusive([j for i in rrp.values() for j in i], name, 0)
    c.validate_max_inclusive([j for i in rrp.values() for j in i], name, 1.0)

def val_single_cell_swrecharge_limit(data, name):
    rrl = data['params'][name]
    rzn = data['params']['single_cell_swrecharge_zone_names']
    c.check_type(param=rrl,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)],
                 keys=range(1, 13))

def val_single_cell_swrecharge_activation(data, name):
    rra = data['params'][name]
    rzn = data['params']['single_cell_swrecharge_zone_names']
    c.check_type(param=rra,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)],
                 keys=range(1, 13))

def val_swrecharge_process(data, name):
    rop = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if rop == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        raise u.ValidationError(msg % (name, "rapid_runoff_process"))

def val_swrecharge_proportion(data, name):
    rrp = data["params"][name]
    rzn = data["params"]["swrecharge_zone_names"]
    c.check_type(
        param=rrp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(rzn)],
        keys=range(1, 13),
    )
    c.validate_min_inclusive([j for i in rrp.values() for j in i], name, 0)
    c.validate_max_inclusive([j for i in rrp.values() for j in i], name, 1.0)

def val_swrecharge_limit(data, name):
    rrl = data["params"][name]
    rzn = data["params"]["swrecharge_zone_names"]
    c.check_type(
        param=rrl,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(rzn)],
        keys=range(1, 13),
    )

def val_macropore_process(data, name):
    mpp = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if mpp == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        raise u.ValidationError(msg % (name, "rapid_runoff_process"))

def val_macropore_proportion(data, name):
    mpp = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.check_type(
        param=mpp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )

    c.validate_min_inclusive([j for i in mpp.values() for j in i], name, 0)
    c.validate_max_inclusive([j for i in mpp.values() for j in i], name, 1.0)

def val_macropore_limit(data, name):
    mpl = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.check_type(
        param=mpl,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )

def val_macropore_activation(data, name):
    mpa = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.check_type(
        param=mpa,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )

def val_macropore_recharge(data, name):
    mpr = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.check_type(
        param=mpr,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )
    c.validate_min_inclusive([j for i in mpr.values() for j in i], name, 0)
    c.validate_max_inclusive([j for i in mpr.values() for j in i], name, 1.0)

def val_soil_static_params(data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    ssp = data["params"][name]
    szn = data["params"]["soil_zone_names"]
    c.check_type(
        param=ssp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(szn)],
        keys=["FC", "WP", "p"],
    )

def val_smd(data, name):
    smd = data["params"][name]
    szn = data["params"]["soil_zone_names"]
    c.check_type(
        param=smd,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(szn)],
        keys=["starting_SMD"],
    )

def val_soil_spatial(data, name):
    sos = data["params"][name]
    soz = data["params"]["soil_zone_names"]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=sos,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[len(soz)],
    )
    if not all(sum(i) == 1.0 for i in sos.values()):
        msg = 'Parameter "%s" requires the sum of its values to be 1.0'
        raise u.ValidationError(msg % name)

def val_lu_spatial(data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    lus = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=lus,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[len(lzn.values())],
    )
    if not all(abs(1 - sum(i)) < 1e-5 for i in lus.values()):
        msg = ('Parameter "%s" requires the sum of its values '
               'to be 1.0 within a tolerance of 1e-5')
        raise u.ValidationError(msg % name)

def val_zr(data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    zrn = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.check_type(
        param=zrn,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(lzn)],
        keys=range(1, 13),
    )

def val_kc(data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    kcn = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.check_type(
        param=kcn,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(lzn)],
        keys=range(1, 13),
    )

def val_taw(data, name):
    if data["params"]["fao_process"] == "disabled":
        return

    taw = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]

    c.check_type(
        param=taw,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(lzn)],
        keys=range(1, 13),
    )

def val_raw(data, name):
    if data["params"]["fao_process"] == "disabled":
        return

    raw = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]

    c.check_type(
        param=raw,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(lzn)],
        keys=range(1, 13),
    )

def val_percolation_rejection(data, name):
    if data["params"]["fao_process"] == "disabled":
        return

    per = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]

    c.check_type(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(lzn)],
        keys=["percolation_rejection"],
    )
    c.validate_min_inclusive(list(per.values())[0], name, 0.0)

def val_percolation_rejection_ts(data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    if not data["params"]['percolation_rejection_use_timeseries']:
        return
    # TODO: Validation needs to be done post-finalization
    # per = data["series"][name]
    # lzn = data["params"]["landuse_zone_names"]
    # c.check_type(
    #     param=per,
    #     name=name,
    #     t_types=data["specs"][name]["type"],
    #     len_list=[len(data["series"]["date"]), len(lzn)],
    #     keys=["percolation_rejection_ts"]
    # )
    # c.validate_min_inclusive(per[0], name=name, 0.0)

def val_subroot_leakage_fraction(data, name):
    lea = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=lea,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

def val_init_interflow_store(data, name):
    nda = data["params"][name]
    tot = len(data["params"]["interflow_zone_names"])
    c.check_type(
        param=nda,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(nda.values(), name, 0)

def val_interflow_store_bypass(data, name):
    nda = data["params"][name]
    tot = len(data["params"]["interflow_zone_names"])
    c.check_type(
        param=nda,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(nda.values(), name, 0)

def val_infiltration_limit(data, name):
    nda = data["params"][name]
    tot = len(data["params"]["interflow_zone_names"])
    c.check_type(
        param=nda,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(nda.values(), name, 0)

def val_interflow_decay(data, name):
    nda = data["params"][name]
    tot = len(data["params"]["interflow_zone_names"])
    c.check_type(
        param=nda,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )
    c.validate_min_inclusive(nda.values(), name, 0)

def val_infiltration_limit_ts(data, name):
    if data["params"]["interflow_process"] == "disabled":
        return
    if not data["params"]['infiltration_limit_use_timeseries']:
        return
    # TODO: Validation needs to be done post-finalization
    # per = data["series"][name]
    # lzn = data["params"]["interflow_zone_names"]
    # c.check_type(
    #     param=per,
    #     name=name,
    #     t_types=data["specs"][name]["type"],
    #     len_list=[len(data["series"]["date"]), len(lzn)],
    #     keys=["infiltration_limit_ts"]
    # )
    # c.validate_min_inclusive(per[0], name=name, 0.0,
    #                       include_low=True)

def val_interflow_decay_ts(data, name):
    if data["params"]["interflow_process"] == "disabled":
        return
    if not data["params"]['interflow_decay_use_timeseries']:
        return
    # TODO: Validation needs to be done post-finalization
    # per = data["series"][name]
    # lzn = data["params"]["interflow_zone_names"]
    # c.check_type(
    #     param=per,
    #     name=name,
    #     t_types=data["specs"][name]["type"],
    #     len_list=[len(data["series"]["date"]), len(lzn)],
    #     keys=["interflow_decay_ts"]
    # )
    # c.validate_min_inclusive(per[0], name, 0.0,
    #                       include_low=True)

def val_recharge_attenuation_params(data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[3],
    )
    c.validate_min_inclusive([i[1] for i in rpn.values()], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive([i[1] for i in rpn.values()], "release_proportion in %s" % name, 1.0)

def val_sw_zone_names(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrn = data["params"][name]
        c.check_type(param=rrn, name=name, t_types=data["specs"][name]["type"])

def val_sw_zone_mapping(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rorzm = data["params"][name]
        tot = data["params"]["num_nodes"]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rorzm,
            name=name,
            t_types=data["specs"][name]["type"],
            keys=range(1, tot + 1),
        )
        c.validate_min_inclusive(rorzm.values(), name, 0)
        c.validate_max_inclusive(rorzm.values(), name, len(rzn))

def val_sw_downstream(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rrp,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[len(rzn)],
            keys=range(1, 13),
        )
        c.validate_min_inclusive([j for i in rrp.values() for j in i], name, 0)
        c.validate_max_inclusive([j for i in rrp.values() for j in i], name, 1.0)

def val_sw_bed_infiltration(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rrp,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[len(rzn)],
            keys=range(1, 13),
        )
        c.validate_min_inclusive([j for i in rrp.values() for j in i], name, 0)
        c.validate_max_inclusive([j for i in rrp.values() for j in i], name, 1.0)

def val_sw_direct_recharge(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rrp,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[len(rzn)],
            keys=range(1, 13),
        )
        c.validate_min_inclusive([j for i in rrp.values() for j in i], name, 0)
        c.validate_max_inclusive([j for i in rrp.values() for j in i], name, 1.0)

def val_sw_activation(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rrp,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[len(rzn)],
            keys=range(1, 13),
        )

def val_sw_pe_to_open_water(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.check_type(
            param=rrp,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[len(rzn)],
            keys=range(1, 13),
        )

def val_sw_init_ponding(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        num = data["params"][name]
        c.check_type(param=num, name=name, t_types=data["specs"][name]["type"])

def val_sw_max_ponding(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        num = data["params"][name]
        c.check_type(param=num, name=name, t_types=data["specs"][name]["type"])

def val_sw_ponding_area(data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        num = data["params"][name]
        c.check_type(param=num, name=name, t_types=data["specs"][name]["type"])
        values = [i for i in num.values()]
        c.validate_max_inclusive(values, name, 1.0)
        c.validate_min_exclusive(values, name, 0)

def val_sw_params(data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[2],
    )
    c.validate_min_inclusive([i[1] for i in rpn.values()], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive([i[1] for i in rpn.values()], "release_proportion in %s" % name, 1.0)

def val_routing_topology(data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[10],
    )

def val_recharge_node_mapping(data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[1],
    )

def val_swdis_f(data, name):
    x = data["params"][name]
    c.validate_constraints([x], name, data["specs"][name]["constraints"])

def val_swabs_f(data, name):
    x = data["params"][name]
    c.validate_constraints([x], name, data["specs"][name]["constraints"])

def val_evt_parameters(data, name):
    rpn = data["params"][name]
    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        # keys=range(1, tot + 1),
        len_list=[3],
    )

def val_nevtopt(data, name):
    x = data["params"][name]
    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])
    c.validate_constraints([x], name, data["specs"][name]["constraints"])

def validate_2(data, specs):
    errors = validate(data, specs)
    warnings = []
    return ParsedInputData(data, errors, warnings)

def validate(data, specs):
    errors = []
    _validate_params(errors, specs, data)
    _validate_series(errors, specs, data)
    return errors

def _validate_params(errors, specs, data):
    do_validation(errors, data, val_node_areas, "node_areas")
    do_validation(errors, data, val_time_periods, "time_periods")
    do_validation(errors, data, val_reporting_zone_mapping, "reporting_zone_mapping")
    do_validation(errors, data, val_rainfall_zone_mapping, "rainfall_zone_mapping")
    do_validation(errors, data, val_rapid_runoff_zone_mapping, "rapid_runoff_zone_mapping")
    do_validation(errors, data, val_pe_zone_mapping, "pe_zone_mapping")
    do_validation(errors, data, val_temperature_zone_mapping, "temperature_zone_mapping")
    do_validation(errors, data, val_tmax_c_zone_mapping, "tmax_c_zone_mapping")
    do_validation(errors, data, val_tmin_c_zone_mapping, "tmin_c_zone_mapping")
    do_validation(errors, data, val_windsp_zone_mapping, "windsp_zone_mapping")
    do_validation(errors, data, val_subroot_zone_mapping, "subroot_zone_mapping")
    do_validation(errors, data, val_swrecharge_zone_mapping, "swrecharge_zone_mapping")
    do_validation(errors, data, val_sw_zone_names, "sw_zone_names")
    do_validation(errors, data, val_sw_zone_mapping, "sw_zone_mapping")
    do_validation(errors, data, val_macropore_zone_mapping, "macropore_zone_mapping")
    do_validation(errors, data, val_recharge_node_mapping, "recharge_node_mapping")
    do_validation(errors, data, val_macropore_activation_option, "macropore_activation_option")
    do_validation(errors, data, val_free_throughfall, "free_throughfall")
    do_validation(errors, data, val_max_canopy_storage, "max_canopy_storage")
    do_validation(errors, data, val_snow_params_simple, "snow_params_simple")
    do_validation(errors, data, val_snow_params_complex, "snow_params_complex")
    do_validation(errors, data, val_rapid_runoff_params, "rapid_runoff_params")
    do_validation(errors, data, val_swrecharge_process, "swrecharge_process")
    do_validation(errors, data, val_swrecharge_proportion, "swrecharge_proportion")
    do_validation(errors, data, val_swrecharge_limit, "swrecharge_limit")
    do_validation(errors, data, val_macropore_process, "macropore_process")
    do_validation(errors, data, val_macropore_proportion, "macropore_proportion")
    do_validation(errors, data, val_macropore_limit, "macropore_limit")
    do_validation(errors, data, val_macropore_activation, "macropore_activation")
    do_validation(errors, data, val_macropore_recharge, "macropore_recharge")
    do_validation(errors, data, val_soil_static_params, "soil_static_params")
    do_validation(errors, data, val_smd, "smd")
    do_validation(errors, data, val_soil_spatial, "soil_spatial")
    do_validation(errors, data, val_lu_spatial, "lu_spatial")
    do_validation(errors, data, val_zr, "zr")
    do_validation(errors, data, val_kc, "kc")
    # do_validation(errors, data, val_taw, "taw")
    # do_validation(errors, data, val_raw, "raw")
    do_validation(errors, data, val_percolation_rejection, "percolation_rejection")
    do_validation(errors, data, val_subroot_leakage_fraction, "subroot_leakage_fraction")
    do_validation(errors, data, val_init_interflow_store, "init_interflow_store")
    do_validation(errors, data, val_interflow_store_bypass, "interflow_store_bypass")
    do_validation(errors, data, val_infiltration_limit, "infiltration_limit")
    do_validation(errors, data, val_interflow_decay, "interflow_decay")
    do_validation(errors, data, val_recharge_attenuation_params, "recharge_attenuation_params")
    do_validation(errors, data, val_sw_init_ponding, "sw_init_ponding")
    do_validation(errors, data, val_sw_max_ponding, "sw_max_ponding")
    do_validation(errors, data, val_sw_downstream, "sw_downstream")
    do_validation(errors, data, val_sw_activation, "sw_activation")
    do_validation(errors, data, val_sw_bed_infiltration, "sw_bed_infiltration")
    do_validation(errors, data, val_sw_direct_recharge, "sw_direct_recharge")
    do_validation(errors, data, val_sw_pe_to_open_water, "sw_pe_to_open_water")
    do_validation(errors, data, val_sw_params, "sw_params")
    do_validation(errors, data, val_sw_ponding_area, "sw_ponding_area")
    do_validation(errors, data, val_swdis_locs, "swdis_locs")
    do_validation(errors, data, val_swabs_locs, "swabs_locs")
    do_validation(errors, data, val_routing_topology, "routing_topology")
    do_validation(errors, data, val_swdis_f, "swdis_f")
    do_validation(errors, data, val_swabs_f, "swabs_f")
    do_validation(errors, data, val_evt_parameters, "evt_parameters")
    do_validation(errors, data, val_nevtopt, "nevtopt")
    do_validation(errors, data, val_interflow_zone_mapping, "interflow_zone_mapping")
    do_validation(errors, data, val_canopy_zone_mapping, "canopy_zone_mapping")

def _validate_series(errors, specs, data):
    do_validation(errors, data, val_rainfall_ts, "rainfall_ts")
    do_validation(errors, data, val_pe_ts, "pe_ts")
    do_validation(errors, data, val_temperature_ts, "temperature_ts")
    do_validation(errors, data, val_tmax_c_ts, "tmax_c_ts")
    do_validation(errors, data, val_tmin_c_ts, "tmin_c_ts")
    do_validation(errors, data, val_windsp_ts, "windsp_ts")
    do_validation(errors, data, val_subroot_leakage_ts, "subroot_leakage_ts")
    do_validation(errors, data, val_swdis_ts, "swdis_ts")
    do_validation(errors, data, val_swabs_ts, "swabs_ts")
    do_validation(errors, data, val_percolation_rejection_ts, "percolation_rejection_ts")
    do_validation(errors, data, val_infiltration_limit_ts, "infiltration_limit_ts")
    do_validation(errors, data, val_interflow_decay_ts, "interflow_decay_ts")

def do_validation(errors, data, function, param):
    params = data["params"]
    series = data["series"]
    is_param_skipped = (
        ((param in params) and (params[param] == None))
        or ((param in series) and (series[param] is None)))
    if not is_param_skipped:
        try:
            function(data, param)
        except u.ValidationError as err:
            errors.append(err.args[0])
        logging.debug('\t\t"%s" validated', param)
