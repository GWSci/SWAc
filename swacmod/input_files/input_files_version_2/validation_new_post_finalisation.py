import logging
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.checks as c
from swacmod.input_files.parsed_input_data import ParsedInputData
from swacmod.utils import monthdelta, weekdelta

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
    rzn = data["params"]["rainfall_zone_names"]
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(rzn)],
    )

def val_pe_ts(data, name):
    pzn = data["params"]["pe_zone_names"]
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(pzn)],
    )

def val_temperature_ts(data, name):
    tzn = set(data["params"]["temperature_zone_mapping"].values())
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_tmax_c_ts(data, name):
    tzn = set(data["params"]["tmax_c_zone_mapping"].values())
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_tmin_c_ts(data, name):
    tzn = set(data["params"]["tmin_c_zone_mapping"].values())
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_windsp_ts(data, name):
    tzn = set(data["params"]["windsp_zone_mapping"].values())
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(tzn)],
    )

def val_subroot_leakage_ts(data, name):
    szn = data["params"]["subroot_zone_names"]
    c.validate_list_length(
        param=(data["series"][name]),
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(szn)],
    )

def val_swdis_ts(data, name):
    swdists = data["series"][name]

    swdisn = data["params"]["swdis_locs"]
    dates = data["series"]["date"]

    freq_flag = data["params"]["swdis_f"]
    ndays = len(dates)
    nweeks = weekdelta(dates[0], dates[-1]) + 1
    nmonths = monthdelta(dates[0], dates[-1]) + 1

    length = [ndays, nweeks, nmonths]
    if swdisn != {0: 0}:
        c.validate_list_length(
            param=swdists,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[length[freq_flag], len(swdisn)],
        )

def val_swabs_ts(data, name):
    swabsts = data["series"][name]
    swabsn = data["params"]["swabs_locs"]
    dates = data["series"]["date"]

    freq_flag = data["params"]["swabs_f"]
    ndays = len(dates)
    nweeks = weekdelta(dates[0], dates[-1]) + 1
    nmonths = monthdelta(dates[0], dates[-1]) + 1

    length = [ndays, nweeks, nmonths]

    if swabsn != {0: 0}:
        c.validate_list_length(
            param=swabsts,
            name=name,
            t_types=data["specs"][name]["type"],
            len_list=[length[freq_flag], len(swabsn)],
        )

def val_percolation_rejection_ts(data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    if not data["params"]['percolation_rejection_use_timeseries']:
        return
    per = data["series"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.validate_keys(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=["percolation_rejection_ts"]
    )
    c.validate_list_length(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(lzn)],
    )
    c.validate_min_inclusive(per[0], name, 0.0)

def val_infiltration_limit_ts(data, name):
    if data["params"]["interflow_process"] == "disabled":
        return
    if not data["params"]['infiltration_limit_use_timeseries']:
        return
    per = data["series"][name]
    lzn = data["params"]["interflow_zone_names"]
    c.validate_keys(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=["infiltration_limit_ts"]
    )
    c.validate_list_length(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(lzn)],
    )
    c.validate_min_inclusive(per[0], name, 0.0)

def val_interflow_decay_ts(data, name):
    if data["params"]["interflow_process"] == "disabled":
        return
    if not data["params"]['interflow_decay_use_timeseries']:
        return
    per = data["series"][name]
    lzn = data["params"]["interflow_zone_names"]
    c.validate_keys(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        keys=["interflow_decay_ts"]
    )
    c.validate_list_length(
        param=per,
        name=name,
        t_types=data["specs"][name]["type"],
        len_list=[len(data["series"]["date"]), len(lzn)],
    )
    c.validate_min_inclusive(per[0], name, 0.0)

def validate_2(data, specs):
    errors = validate(data, specs)
    warnings = []
    return ParsedInputData(data, errors, warnings)

def validate(data, specs):
    errors = []
    do_validation(errors, data, val_time_periods, "time_periods")
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
    return errors

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
