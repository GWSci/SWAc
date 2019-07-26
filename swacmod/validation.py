#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import logging
import multiprocessing

# Internal modules
from . import utils as u
from . import checks as c


###############################################################################
def val_run_name(data, name):
    """Validate run_name.

    1) type has to be string
    """
    rnm = data["params"][name]
    c.check_type(param=rnm, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_num_cores(data, name):
    """Validate num_cores.

    1) type has to be integer
    2) value has to be 0 < x <= number of machine cores
    """
    num = data["params"][name]
    c.check_type(param=num, name=name, t_types=data["specs"][name]["type"])
    c.check_values_limits(
        values=[num],
        name=name,
        low_l=0,
        high_l=multiprocessing.cpu_count(),
        include_high=True,
    )


###############################################################################
def val_num_nodes(data, name):
    """Validate num_nodes.

    1) type has to be integer
    2) value has to be > 0
    """
    num = data["params"][name]
    c.check_type(param=num, name=name, t_types=data["specs"][name]["type"])
    c.check_values_limits(values=[num], name=name, low_l=0)


###############################################################################
def val_start_date(data, name):
    """Validate start_date.

    NOTE: 1/12/70 is ambiguous
          if Excel conversion I need the datemode of the file

    1) type has to be datetime object (string is parsed in io module)
    """
    dat = data["params"][name]
    c.check_type(param=dat, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_time_periods(data, name):
    """Validate time_periods.

    1) type has to be a list of lists of integers
    2) start and end times have to be positive and less than the # of days
    3) time periods need to be lists of length 2
    4) start time has to be smaller than end time
    5) end time is not inclusive
    6) all days are assigned to a time period
    """
    tmp = data["params"][name]
    c.check_type(param=tmp, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[i for j in tmp for i in j],
        name=name,
        low_l=0,
        high_l=len(data["series"]["date"]) + 1,
        include_high=True,
    )

    all_days = []
    for time_range in tmp:
        if len(time_range) != 2:
            msg = 'Parameter "%s" requires arrays of length 2'
            raise u.ValidationError(msg % name)
        if not time_range[0] < time_range[1]:
            msg = 'Parameter "%s" requires start_date < end_date'
            raise u.ValidationError(msg % name)
        all_days += range(time_range[0], time_range[1])

    if set(all_days) != set(range(1, len(data["series"]["date"]) + 1)):
        msg = (
            'Parameter "%s" requires all days to be included'
            " in one (and only one) of the periods"
        )
        raise u.ValidationError(msg % name)


###############################################################################
def val_output_recharge(data, name):
    """Validate output_recharge.

    1) type has to be a boolean
    """
    opr = data["params"][name]

    c.check_type(param=opr, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_routing_process(data, name):
    """Validate routing_process.

    """
    opr = data["params"][name]

    c.check_type(param=opr, name=name, t_types=data["specs"][name]["type"])

###############################################################################
def val_output_sfr(data, name):
    """Validate output_sfr.

    1) type has to be a boolean
    """
    opr = data["params"][name]

    c.check_type(param=opr, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_output_individual(data, name):
    """Validate output_individual.

    1) type has to be a set of integers
    2) all ids in the list have also to be node ids
    """
    oin = data["params"][name]

    c.check_type(param=oin, name=name, t_types=data["specs"][name]["type"])

    ids = set(range(1, data["params"]["num_nodes"] + 1))

    if not all(i in ids for i in oin):
        msg = 'Parameter "%s" requires all node ids to be 1 <= x <= ' "num_nodes"
        raise u.ValidationError(msg % name)


###############################################################################
def val_irchcb(data, name):
    """Validate irchcb.

    1) type has to be an integer
    """
    irc = data["params"][name]

    c.check_type(param=irc, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_nodes_per_line(data, name):
    """Validate nodes_per_line.

    1) type has to be a positive integer
    """
    npl = data["params"][name]

    c.check_type(param=npl, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(values=[npl], name=name, low_l=0)


###############################################################################
def val_output_fac(data, name):
    """Validate output_fac.

    1) type has to be a positive float
    """
    fac = data["params"][name]

    c.check_type(param=fac, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(values=[fac], name=name, low_l=0.0)


###############################################################################
def val_spatial_output_date(data, name):
    """Validate spatial_output_date.

    NOTE: 1/12/70 is ambiguous
          if Excel conversion I need the datemode of the file

    1) type has to be datetime object (string is parsed in io module) or None
    """
    dat = data["params"][name]
    if dat is None or dat != "mean":
        return

    c.check_type(param=dat, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_rainfall_ts(data, name):
    """Validate rainfall_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days x number of zones
    """
    rts = data["series"][name]
    rzn = data["params"]["rainfall_zone_names"]

    c.check_type(
        param=rts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(rzn)],
    )


###############################################################################
def val_pe_ts(data, name):
    """Validate pe_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days x number of zones
    """
    pts = data["series"][name]
    pzn = data["params"]["pe_zone_names"]

    c.check_type(
        param=pts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(pzn)],
    )


###############################################################################
def val_temperature_ts(data, name):
    """Validate temperature_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days x number of zones
    """
    tts = data["series"][name]
    tzn = set(data["params"]["temperature_zone_mapping"].values())

    c.check_type(
        param=tts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )


###############################################################################
def val_subroot_leakage_ts(data, name):
    """Validate subroot_leakage_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days x number of zones
    """
    sts = data["series"][name]
    szn = data["params"]["subroot_zone_names"]

    c.check_type(
        param=sts,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(szn)],
    )


###############################################################################
def val_swdis_ts(data, name):
    """Validate swdis_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days/weeks/months
    x number of zones
    """

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


###############################################################################


def val_swabs_ts(data, name):
    """Validate swabs_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of days/weeks/months
    x number of zones
    """

    from swacmod.utils import monthdelta, weekdelta

    swabsts = data["series"][name]
    swabsn = data["params"]["swabs_locs"]
    dates = data["series"]["date"]

    freq_flag = data["params"]["swabs_f"]
    ndays = len(dates)
    nweeks = weekdelta(dates[0], dates[-1]) + 1
    nmonths = monthdelta(dates[0], dates[-1]) + 1

    length = [ndays, nweeks, nmonths]

    if swabsn != {0: 0}:
        c.check_type(
            param=swabsts,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[length[freq_flag], len(swabsn)],
        )


###############################################################################
def val_swdis_locs(data, name):
    """Validate swdis_locs.

    1) type has to be a dictionary of integers
    2) all swabs ids have to be present
    3) values (i.e. swabs ids) have to be > 1 and <= number of swabs
    """
    swdisl = data["params"][name]

    tot = len(data["params"]["swdis_locs"]) + 1

    c.check_type(
        param=swdisl, name=name, t_types=data["specs"][name]["type"], keys=range(1, tot)
    )

    c.check_values_limits(
        values=swdisl.values(),
        name="zone in %s" % name,
        low_l=0,
        include_low=True,
        high_l=tot,
        include_high=True,
    )


###############################################################################
def val_swabs_locs(data, name):
    """Validate swabs_locs.

    1) type has to be a dictionary of integers
    2) all swabs ids have to be present
    3) values (i.e. swabs ids) have to be > 1 and <= number of swabs
    """
    swabsl = data["params"][name]

    tot = len(data["params"]["swabs_locs"]) + 1

    c.check_type(
        param=swabsl, name=name, t_types=data["specs"][name]["type"], keys=range(1, tot)
    )

    c.check_values_limits(
        values=swabsl.values(),
        name="zone in %s" % name,
        low_l=0,
        include_low=True,
        high_l=tot,
        include_high=True,
    )


###############################################################################
def val_node_areas(data, name):
    """Validate node_areas.

    1) type has to be a dict of floats
    2) all node ids have to be present
    3) values have to be >= 0.
    """
    nda = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=nda,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(values=nda.values(), name=name, low_l=0, include_low=True)


###############################################################################
def val_reporting_zone_names(data, name):
    """Validate reporting_zone_names.

    1) type has to be a dictionary of strings
    """
    rzn = data["params"][name]
    c.check_type(param=rzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_reporting_zone_mapping(data, name):
    """Validate reporting_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    rzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["reporting_zone_names"]

    c.check_type(
        param=rzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=rzm.values(),
        name="zone in %s" % name,
        low_l=0,
        include_low=True,
        high_l=len(rzn),
        include_high=True,
    )


###############################################################################
def val_rainfall_zone_names(data, name):
    """Validate rainfall_zone_names.

    1) type has to be a dictionary of strings
    """
    rzn = data["params"][name]
    c.check_type(param=rzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_rainfall_zone_mapping(data, name):
    """Validate rainfall_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    rzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["rainfall_zone_names"]

    c.check_type(
        param=rzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=[i[0] for i in rzm.values()],
        name="zone in %s" % name,
        low_l=1,
        include_low=True,
        high_l=len(rzn),
        include_high=True,
    )


###############################################################################
def val_pe_zone_names(data, name):
    """Validate pe_zone_names.

    1) type has to be a dictionary of strings
    """
    pzn = data["params"][name]
    c.check_type(param=pzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_pe_zone_mapping(data, name):
    """Validate pe_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    pzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    pzn = data["params"]["pe_zone_names"]

    c.check_type(
        param=pzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=[i[0] for i in pzm.values()],
        name="zone in %s" % name,
        low_l=0,
        include_low=True,
        high_l=len(pzn),
        include_high=True,
    )


###############################################################################
def val_temperature_zone_names(data, name):
    """Validate temperature_zone_names.

    1) type has to be a dictionary of strings
    """
    tzn = data["params"][name]
    c.check_type(param=tzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_temperature_zone_mapping(data, name):
    """Validate temperature_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    tzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"]["temperature_zone_names"]

    c.check_type(
        param=tzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=tzm.values(),
        name=name,
        low_l=0,
        include_low=True,
        high_l=len(tzn),
        include_high=True,
    )


###############################################################################
def val_subroot_zone_names(data, name):
    """Validate subroot_zone_names.

    1) type has to be a dictionary of strings
    """
    szn = data["params"][name]
    c.check_type(param=szn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_subroot_zone_mapping(data, name):
    """Validate subroot_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    szm = data["params"][name]
    tot = data["params"]["num_nodes"]
    szn = data["params"]["subroot_zone_names"]

    c.check_type(
        param=szm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=[i[0] for i in szm.values()],
        name="zone in %s" % name,
        low_l=0,
        include_low=True,
        high_l=len(szn),
        include_high=True,
    )


###############################################################################
def val_rapid_runoff_zone_names(data, name):
    """Validate rapid_runoff_zone_names.

    1) type has to be a dictionary of strings
    """
    rrn = data["params"][name]
    c.check_type(param=rrn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_rapid_runoff_zone_mapping(data, name):
    """Validate rapid_runoff_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    rrzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["rapid_runoff_zone_names"]

    c.check_type(
        param=rrzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=rrzm.values(),
        name=name,
        low_l=0,
        include_low=True,
        high_l=len(rzn),
        include_high=True,
    )


###############################################################################
def val_swrecharge_zone_names(data, name):
    """Validate swrecharge_zone_names.

    1) type has to be a dictionary of strings
    """
    rrn = data["params"][name]
    c.check_type(param=rrn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_swrecharge_zone_mapping(data, name):
    """Validate swrecharge_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    rorzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    rzn = data["params"]["swrecharge_zone_names"]

    c.check_type(
        param=rorzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=rorzm.values(),
        name=name,
        low_l=0,
        include_low=True,
        high_l=len(rzn),
        include_high=True,
    )


###############################################################################
def val_macropore_zone_names(data, name):
    """Validate macropore_zone_names.

    1) type has to be a dictionary of strings
    """
    mzn = data["params"][name]
    c.check_type(param=mzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_macropore_zone_mapping(data, name):
    """Validate macropore_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be 0 <= x <= number of zones
    """
    mzm = data["params"][name]
    tot = data["params"]["num_nodes"]
    mzn = data["params"]["macropore_zone_names"]

    c.check_type(
        param=mzm,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=mzm.values(),
        name=name,
        low_l=0,
        include_low=True,
        high_l=len(mzn),
        include_high=True,
    )


###############################################################################
def val_soil_zone_names(data, name):
    """Validate soil_zone_names.

    1) type has to be a dictionary of strings
    """
    szn = data["params"][name]
    c.check_type(param=szn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_landuse_zone_names(data, name):
    """Validate landuse_zone_names.

    1) type has to be a dictionary of strings
    """
    lzn = data["params"][name]
    c.check_type(param=lzn, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_canopy_process(data, name):
    """Validate canopy_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    cpr = data["params"][name]

    c.check_type(param=cpr, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[cpr], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_free_throughfall(data, name):
    """Validate free_throughfall.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values have to be 0 <= x <= 1
    """
    fth = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=fth,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=fth.values(),
        name=name,
        low_l=0,
        high_l=1.0,
        include_low=True,
        include_high=True,
    )


###############################################################################
def val_max_canopy_storage(data, name):
    """Validate max_canopy_storage.

    1) type has to be a dictionary of floats
    2) all node ids have to be present
    3) values have to be >= 0
    """
    mcs = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=mcs,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(values=mcs.values(), name=name, low_l=0, include_low=True)


###############################################################################
def val_snow_process(data, name):
    """Validate snow_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    spr = data["params"][name]

    c.check_type(param=spr, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[spr], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_snow_params(data, name):
    """Validate snow_params.

    1) type has to be a dictionary of lists of numbers
    2) all node ids have to be present
    3) values have to lists with 3 elements
    4) the first element (starting_snow_pack) is a number >= 0
    """
    if data["params"]["snow_process"] == "disabled":
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

    c.check_values_limits(
        values=[i[0] for i in snp.values()],
        name="starting_snow_pack in %s" % name,
        low_l=0,
        include_low=True,
    )


###############################################################################
def val_rapid_runoff_process(data, name):
    """Validate rapid_runoff_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    rrp = data["params"][name]

    c.check_type(param=rrp, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[rrp], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_rapid_runoff_params(data, name):
    """Validate rapid_runoff_params.

    1) type has to be a list of dictionaries of lists
    2) list needs to have length equal to the number of zones
    3) all dictionaries need 3 keys: class_smd, class_ri, values
    4) all "values" has dimensions class_smd x class_ri
    5) all values are floats between 0 <= x <= 1
    """
    rrp = data["params"][name]
    rzn = data["params"]["rapid_runoff_zone_names"]

    c.check_type(
        param=rrp, name=name, t_types=data["specs"][name]["type"], len_list=[len(rzn)]
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

        c.check_values_limits(
            values=[i for j in zone["values"] for i in j],
            name='"values" in "%s"' % name,
            low_l=0,
            high_l=1,
            include_low=True,
            include_high=True,
        )


###############################################################################
def val_swrecharge_process(data, name):
    """Validate swrecharge_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    rop = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if rop == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        raise u.ValidationError(msg % (name, "rapid_runoff_process"))

    c.check_type(param=rop, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[rop], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_swrecharge_proportion(data, name):
    """Validate swrecharge_proportion.

    1) type has to be a dict of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    4) all elements of each list have to be 0 <= x <= 1
    """
    rrp = data["params"][name]
    rzn = data["params"]["swrecharge_zone_names"]

    c.check_type(
        param=rrp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(rzn)],
        keys=range(1, 13),
    )

    c.check_values_limits(
        values=[j for i in rrp.values() for j in i],
        name=name,
        low_l=0,
        high_l=1.0,
        include_low=True,
        include_high=True,
    )


###############################################################################
def val_swrecharge_limit(data, name):
    """Validate swrecharge_limit.

    1) type has to be a dict of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    rrl = data["params"][name]
    rzn = data["params"]["swrecharge_zone_names"]

    c.check_type(
        param=rrl,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(rzn)],
        keys=range(1, 13),
    )

###############################################################################
def val_macropore_process(data, name):
    """Validate macropore_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    mpp = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if mpp == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        raise u.ValidationError(msg % (name, "rapid_runoff_process"))

    c.check_type(param=mpp, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[mpp], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_macropore_proportion(data, name):
    """Validate macropore_proportion.

    1) type has to be a dict of lists
    2) the dict requires length 12 (months)
    3) the lists require lenght equal to the number of zones
    4) all elements of each list have to be 0 <= x <= 1
    """
    mpp = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]

    c.check_type(
        param=mpp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )

    c.check_values_limits(
        values=[j for i in mpp.values() for j in i],
        name=name,
        low_l=0,
        high_l=1.0,
        include_low=True,
        include_high=True,
    )


###############################################################################
def val_macropore_limit(data, name):
    """Validate macropore_limit.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    mpl = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]

    c.check_type(
        param=mpl,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )


###############################################################################
def val_macropore_activation(data, name):
    """Validate macropore_activation.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    mpa = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]

    c.check_type(
        param=mpa,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )


###############################################################################
def val_macropore_recharge(data, name):
    """Validate macropore_recharge.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    4) all elements of each list have to be 0 <= x <= 1
    """
    mpr = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]

    c.check_type(
        param=mpr,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(mzn)],
        keys=range(1, 13),
    )

    c.check_values_limits(
        values=[j for i in mpr.values() for j in i],
        name=name,
        low_l=0,
        high_l=1.0,
        include_low=True,
        include_high=True,
    )


###############################################################################
def val_fao_process(data, name):
    """Validate fao_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    fao = data["params"][name]

    c.check_type(param=fao, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[fao], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_fao_input(data, name):
    """Validate fao_input.

    1) type has to be a string
    2) value has to be one in ['ls', 'l']
    """
    fao = data["params"][name]

    c.check_type(param=fao, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[fao], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_soil_static_params(data, name):
    """Validate soil_static_params.

    1) type has to be a dictionary of lists of floats
    2) values have to be lists with length equal to the number of zones
    3) dictionary needs 4 keys: FC, WP, p
    """
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


###############################################################################
def val_smd(data, name):
    """Validate smd.

    1) type has to be a dictionary of lists of floats
    2) values have to be lists with length equal to the number of zones
    3) dictionary needs 1 key: starting_SMD
    """
    if data["params"]["fao_process"] == "disabled":
        return

    smd = data["params"][name]
    szn = data["params"]["soil_zone_names"]

    c.check_type(
        param=smd,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(szn)],
        keys=["starting_SMD"],
    )


###############################################################################
def val_soil_spatial(data, name):
    """Validate soil_spatial.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length equal to the number of zones
    4) the sum of each row has to be 1.0
    """
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return

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


###############################################################################
def val_lu_spatial(data, name):
    """Validate lu_spatial.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length equal to the number of zones
    4) the sum of each row has to be 1.0
    """
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


###############################################################################
def val_zr(data, name):
    """Validate ZR.

    1) type has to be a dict of lists of floats
    2) dictionary needs to have integer keys, from 1 to 12
    3) lists need to have length equal to the number of zones
    """
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


###############################################################################
def val_kc(data, name):
    """Validate KC.

    1) type has to be a dict of lists of floats
    2) dictionary needs to have integer keys, from 1 to 12
    3) lists need to have length equal to the number of zones
    """
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


###############################################################################
def val_taw(data, name):
    """Validate TAW.

    1) type has to be a dict of lists of floats
    2) dictionary needs to have integer keys, from 1 to 12
    3) lists need to have length equal to the number of months
    """
    if data["params"]["fao_process"] == "disabled":
        return

    taw = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=taw,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[12],
        keys=range(1, tot + 1),
    )


###############################################################################
def val_raw(data, name):
    """Validate RAW.

    1) type has to be a dict of lists of floats
    2) dictionary needs to have integer keys, from 1 to 12
    3) lists need to have length equal to the number of zones
    """
    if data["params"]["fao_process"] == "disabled":
        return

    raw = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=raw,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[12],
        keys=range(1, tot + 1),
    )


###############################################################################
def val_percolation_rejection(data, name):
    """Validate percolation_rejection.

    1) type has to be a dictionary of lists of floats
    2) values have to be lists with length equal to the number of zones
    3) dictionary needs 1 key: percolation_rejection
    4) value should be >= 0.0
    """
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
    c.check_values_limits(values=list(per.values())[0], name=name, low_l=0.0, include_low=True)


###############################################################################
def val_leakage_process(data, name):
    """Validate leakage_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    lep = data["params"][name]

    c.check_type(param=lep, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[lep], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_subsoilzone_leakage_fraction(data, name):
    """Validate subsoilzone_leakage_fraction.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    """
    lea = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=lea,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
    )


###############################################################################
def val_interflow_process(data, name):
    """Validate interflow_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    ifp = data["params"][name]

    c.check_type(param=ifp, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[ifp], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_interflow_params(data, name):
    """Validate interflow_params.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with 4 elements
    4) floats need to be >= 0
    5) the first and third elements of each list has to be <= 1
    """
    ifp = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=ifp,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[4],
        keys=range(1, tot + 1),
    )

    c.check_values_limits(
        values=[i for j in ifp.values() for i in j],
        name=name,
        low_l=0,
        include_low=True,
    )

    c.check_values_limits(
        values=[j for i in ifp.values() for j in [i[1], i[3]]],
        name=("store_bypass and interflow_to_rivers in %s" % name),
        high_l=1.0,
        include_high=True,
    )


###############################################################################
def val_recharge_attenuation_process(data, name):
    """Validate recharge_attenuation_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    rap = data["params"][name]

    c.check_type(param=rap, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[rap], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_recharge_attenuation_params(data, name):
    """Validate recharge_attenuation_params.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length 3
    4) the first element of each list has to be 0 <= x <= 1
    """
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[3],
    )

    c.check_values_limits(
        values=[i[1] for i in rpn.values()],
        name="release_proportion in %s" % name,
        low_l=0.0,
        include_low=True,
        high_l=1.0,
        include_high=True,
    )


###############################################################################
def val_sw_process(data, name):
    """Validate sw_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    rap = data["params"][name]

    c.check_type(param=rap, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[rap], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_sw_params(data, name):
    """Validate sw_params.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length 2
    4) the first element of each list has to be 0 <= x <= 1
    """
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[2],
    )

    c.check_values_limits(
        values=[i[1] for i in rpn.values()],
        name="release_proportion in %s" % name,
        low_l=0.0,
        include_low=True,
        high_l=1.0,
        include_high=True,
    )


###############################################################################
def val_routing_topology(data, name):
    """Validate routing .

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length 11
    """
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[10],
    )

###############################################################################
def val_recharge_node_mapping(data, name):
    """Validate recharge node .

    """
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[1],
    )


###############################################################################
def val_istcb1(data, name):
    """Validate istcb1.

    1) type has to be an integer
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_istcb2(data, name):
    """Validate istcb2.

    1) type has to be an integer
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_swdis_f(data, name):
    """Validate swdis_f.

    1) type has to be an integer
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[x], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_swabs_f(data, name):
    """Validate swabs_f.

    1) type has to be an integer
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[x], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_output_evt(data, name):
    """Validate output_evt.

    1) type has to be a boolean
    """
    opr = data["params"][name]

    c.check_type(param=opr, name=name, t_types=data["specs"][name]["type"])

###############################################################################
def val_excess_sw_process(data, name):
    """Validate excess_sw_process.

    1) type has to be a string
    2) value has to be one in ['enabled', 'disabled']
    """
    cpr = data["params"][name]

    c.check_type(param=cpr, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[cpr], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_evt_parameters(data, name):
    """Validate evt parameters .

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length 3
    """
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]

    c.check_type(
        param=rpn,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=range(1, tot + 1),
        len_list=[3],
    )


###############################################################################
def val_ievtcb(data, name):
    """Validate ievtcb.

    1) type has to be an integer
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])


###############################################################################
def val_nevtopt(data, name):
    """Validate nevtopt.

    1) type has to be an integer 1, 2 or 3
    """
    x = data["params"][name]

    c.check_type(param=x, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[x], name=name, constraints=data["specs"][name]["constraints"]
    )


###############################################################################
def val_gwmodel_type(data, name):
    """Validate gwmodel_type.

    1) type has to be a string
    2) value has to be one in ['mf6', 'mfusg']
    """
    rap = data["params"][name]

    c.check_type(param=rap, name=name, t_types=data["specs"][name]["type"])

    c.check_values_limits(
        values=[rap], name=name, constraints=data["specs"][name]["constraints"]
    )



FUNC_PARAMS = [
    val_run_name,
    val_num_cores,
    val_num_nodes,
    val_node_areas,
    val_start_date,
    val_time_periods,
    val_output_recharge,
    val_output_individual,
    val_irchcb,
    val_nodes_per_line,
    val_output_fac,
    val_spatial_output_date,
    val_reporting_zone_names,
    val_reporting_zone_mapping,
    val_rainfall_zone_names,
    val_rainfall_zone_mapping,
    val_rapid_runoff_zone_names,
    val_rapid_runoff_zone_mapping,
    val_pe_zone_names,
    val_pe_zone_mapping,
    val_temperature_zone_names,
    val_temperature_zone_mapping,
    val_subroot_zone_names,
    val_subroot_zone_mapping,
    val_swrecharge_zone_names,
    val_swrecharge_zone_mapping,
    val_macropore_zone_names,
    val_macropore_zone_mapping,
    val_soil_zone_names,
    val_landuse_zone_names,
    val_canopy_process,
    val_free_throughfall,
    val_max_canopy_storage,
    val_snow_process,
    val_snow_params,
    val_rapid_runoff_process,
    val_rapid_runoff_params,
    val_swrecharge_process,
    val_swrecharge_proportion,
    val_swrecharge_limit,
    val_macropore_process,
    val_macropore_proportion,
    val_macropore_limit,
    val_macropore_activation,
    val_macropore_recharge,
    val_fao_process,
    val_fao_input,
    val_soil_static_params,
    val_smd,
    val_soil_spatial,
    val_lu_spatial,
    val_zr,
    val_kc,
    val_taw,
    val_raw,
    val_percolation_rejection,
    val_leakage_process,
    val_subsoilzone_leakage_fraction,
    val_interflow_process,
    val_interflow_params,
    val_recharge_attenuation_process,
    val_recharge_attenuation_params,
    val_sw_process,
    val_sw_params,
    val_swdis_locs,
    val_swabs_locs,
    val_istcb1,
    val_istcb2,
    val_routing_topology,
    val_swdis_f,
    val_swabs_f,
    val_output_evt,
    val_evt_parameters,
    val_ievtcb,
    val_nevtopt,
    val_gwmodel_type,
    val_excess_sw_process
]


FUNC_SERIES = [
    val_rainfall_ts,
    val_pe_ts,
    val_temperature_ts,
    val_subroot_leakage_ts,
    val_swdis_ts,
    val_swabs_ts,
]


###############################################################################
def validate_params(data):
    """Validate all parameters using their specifications."""
    logging.info("\tValidating parameters")

    for function in FUNC_PARAMS:
        param = function.__name__.replace("val_", "")
        function(data, param)
        logging.debug('\t\t"%s" validated', param)

    logging.info("\tDone.")


###############################################################################
def validate_series(data):
    """Validate all time series using their specifications."""
    logging.info("\tValidating time series")

    for function in FUNC_SERIES:
        series = function.__name__.replace("val_", "")
        function(data, series)
        logging.debug('\t\t"%s" validated', series)

    logging.info("\tDone.")
