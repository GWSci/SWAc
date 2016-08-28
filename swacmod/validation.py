#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import re
import logging

# Third Party Libraries
from dateutil import parser

# Internal modules
from . import utils as u
from . import checks as c


###############################################################################
def val_run_name(data, name):
    """Validate run_name.

    1) type has to be string, convert it otherwise
    2) replace spaces and non-alphanumeric characters with underscores
    """
    rnm = data['params'][name]

    if not isinstance(rnm, basestring):
        data['params'][name] = str(rnm)
        logging.info('\t\tConverted "%s" to string', name)

    new_value = re.sub(r'[^a-zA-Z\-0-9]', '_', data['params'][name])
    if new_value != data['params'][name]:
        data['params'][name] = new_value
        logging.info('\t\tNew "%s": %s', name, new_value)


###############################################################################
def val_log_file(data, name):
    """Validate log_file.

    NOTE: made this optional

    1) path has to be absolute, convert it otherwise
    """
    log = data['params'][name]
    data['params'][name] = c.check_path(path=log)
    if data['params'][name] != log:
        logging.info('\t\tNormalized "%s" path to %s', name,
                     data['params'][name])


###############################################################################
def val_num_nodes(data, name):
    """Validate num_nodes.

    1) type has to be integer
    2) value has to be > 0
    """
    num = data['params'][name]
    c.check_type(param=num, name=name, t_types=data['specs'][name]['type'])
    c.check_values_limits(values=[num], name=name, low_l=0)


###############################################################################
def val_start_date(data, name):
    """Validate start_date.

    NOTE: 1/12/70 is ambiguous
          if Excel conversion I need the datemode of the file

    1) type has to be str
    2) value has to be parsable by dateutil.parser
    """
    dat = data['params'][name]
    c.check_type(param=dat, name=name, t_types=data['specs'][name]['type'])

    try:
        data['params'][name] = parser.parse(dat)
    except Exception:
        msg = 'Parameter "%s" shoud be in the format "1980-01-01"'
        raise u.ValidationError(msg % name)


###############################################################################
def val_time_periods(data, name):
    """Validate time_periods.

    1) type has to be a dictionary of lists of integers
    2) all node ids have to be present
    3) values have to lists with 2 elements
    4) start and end times have to be positive
    5) start time has to be smaller than end time
    6) all days are assigned to a time period
    """
    tmp = data['params'][name]

    c.check_type(param=tmp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[2],
                 keys=range(1, len(tmp) + 1))

    c.check_values_limits(values=[i for j in tmp.values() for i in j],
                          name=name,
                          low_l=0)

    all_days = []
    for time_range in tmp.values():
        if not time_range[0] < time_range[1]:
            msg = 'Parameter "%s" requires start_date < end_date'
            raise u.ValidationError(msg % name)
        all_days += range(time_range[0], time_range[1] + 1)

    if set(all_days) != set(range(1, len(data['series']['rainfall_ts']) + 1)):
        msg = ('Parameter "%s" requires all days to be included'
               ' in one of the periods')
        raise u.ValidationError(msg % name)


###############################################################################
def val_rainfall_ts(data, name):
    """Validate rainfall_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of zones
    """
    rts = data['series'][name]
    rzn = data['params']['rainfall_zone_names']

    c.check_type(param=rts,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)])


###############################################################################
def val_pe_ts(data, name):
    """Validate pe_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of zones
    """
    pts = data['series'][name]
    pzn = data['params']['pe_zone_names']

    c.check_type(param=pts,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(pzn)])


###############################################################################
def val_temperature_ts(data, name):
    """Validate temperature_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of zones
    """
    tts = data['series'][name]
    tzn = data['params']['temperature_zone_names']

    c.check_type(param=tts,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(tzn)])


###############################################################################
def val_subroot_leakage_ts(data, name):
    """Validate subroot_leakage_ts.

    1) type has to be a dictionary of lists of floats
    2) list length has to be equal to the number of zones
    """
    sts = data['series'][name]
    szn = data['params']['subroot_zone_names']

    c.check_type(param=sts,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(szn)])


###############################################################################
def val_node_areas(data, name):
    """Validate node_areas.

    1) type has to be a dict of floats
    2) all node ids have to be present
    3) values have to be >= 0.
    """
    nda = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=nda,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=nda.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_reporting_zone_names(data, name):
    """Validate reporting_zone_names.

    1) type has to be a dictionary of strings
    """
    rzn = data['params'][name]
    c.check_type(param=rzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_reporting_zone_mapping(data, name):
    """Validate reporting_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    4) number of distinct zone ids has to be equal to number of zone names
    """
    rzm = data['params'][name]
    tot = data['params']['num_nodes']
    rzn = data['params']['reporting_zone_names']

    c.check_type(param=rzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=rzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)

    zones = [i for i in rzm.values() if i != 0]
    if len(set(zones)) != len(rzn):
        msg = ('Parameter "zone_mapping" has to include a number of zones'
               ' equal to the one in "zone_names"')
        raise u.ValidationError(msg)


###############################################################################
def val_rainfall_zone_names(data, name):
    """Validate rainfall_zone_names.

    1) type has to be a dictionary of strings
    """
    rzn = data['params'][name]
    c.check_type(param=rzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_rainfall_zone_mapping(data, name):
    """Validate rainfall_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    rzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=rzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=rzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_pe_zone_names(data, name):
    """Validate pe_zone_names.

    1) type has to be a dictionary of strings
    """
    pzn = data['params'][name]
    c.check_type(param=pzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_pe_zone_mapping(data, name):
    """Validate pe_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    pzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=pzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=pzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_temperature_zone_names(data, name):
    """Validate temperature_zone_names.

    1) type has to be a dictionary of strings
    """
    tzn = data['params'][name]
    c.check_type(param=tzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_temperature_zone_mapping(data, name):
    """Validate temperature_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    tzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=tzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=tzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_subroot_zone_names(data, name):
    """Validate subroot_zone_names.

    1) type has to be a dictionary of strings
    """
    szn = data['params'][name]
    c.check_type(param=szn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_subroot_zone_mapping(data, name):
    """Validate subroot_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    szm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=szm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=szm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_rapid_runoff_zone_names(data, name):
    """Validate rapid_runoff_zone_names.

    1) type has to be a dictionary of strings
    """
    rrn = data['params'][name]
    c.check_type(param=rrn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_rapid_runoff_zone_mapping(data, name):
    """Validate rapid_runoff_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    rrzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=rrzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=rrzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_ror_zone_names(data, name):
    """Validate ror_zone_names.

    1) type has to be a dictionary of strings
    """
    rrn = data['params'][name]
    c.check_type(param=rrn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_ror_zone_mapping(data, name):
    """Validate ror_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    rorzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=rorzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=rorzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_macropore_zone_names(data, name):
    """Validate macropore_zone_names.

    1) type has to be a dictionary of strings
    """
    mzn = data['params'][name]
    c.check_type(param=mzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_macropore_zone_mapping(data, name):
    """Validate macropore_zone_mapping.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values (i.e. zone ids) have to be >= 0
    """
    mzm = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=mzm,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=mzm.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_soil_zone_names(data, name):
    """Validate soil_zone_names.

    1) type has to be a dictionary of strings
    """
    szn = data['params'][name]
    c.check_type(param=szn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_land_zone_names(data, name):
    """Validate land_zone_names.

    1) type has to be a dictionary of strings
    """
    lzn = data['params'][name]
    c.check_type(param=lzn, name=name, t_types=data['specs'][name]['type'])


###############################################################################
def val_free_throughfall(data, name):
    """Validate free_throughfall.

    1) type has to be a dictionary of integers
    2) all node ids have to be present
    3) values have to be >= 0
    """
    fth = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=fth,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=fth.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_max_canopy_storage(data, name):
    """Validate max_canopy_storage.

    1) type has to be a dictionary of floats
    2) all node ids have to be present
    3) values have to be >= 0
    """
    mcs = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=mcs,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=mcs.values(),
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_snow_params(data, name):
    """Validate snow_params.

    1) type has to be a dictionary of lists of numbers
    2) all node ids have to be present
    3) values have to lists with 3 elements
    4) the first element (starting_snow_pack) is a number >= 0
    """
    snp = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=snp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[3],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=[i for j in snp.values() for i in j],
                          name='starting_snow_pack',
                          low_l=0,
                          include_low=True)


###############################################################################
def val_rapid_runoff_params(data, name):
    """Validate rapid_runoff_params.

    1) type has to be a list of dictionaries of lists
    2) list needs to have length equal to the number of zones
    3) dictionaries need 3 keys: class_smd, class_ri, values
    4) the value of "values" has dimensions class_smd x class_ri
    5) all values are floats between 0 < x < 1
    """
    rrp = data['params'][name]
    rzn = data['params']['rapid_runoff_zone_names']

    c.check_type(param=rrp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)])

    keys = ['class_smd', 'class_ri', 'values']
    for zone in rrp:

        c.check_type(param=zone,
                     name=name,
                     t_types=[dict],
                     keys=keys)

        c.check_type(param=zone['values'],
                     name=name,
                     t_types=[list, list],
                     len_list=[len(zone['class_ri']), len(zone['class_smd'])])

        c.check_values_limits(values=[i for j in zone['values'] for i in j],
                              name='"values" in "%s"' % name,
                              low_l=0,
                              high_l=1,
                              include_low=True,
                              include_high=True)


###############################################################################
def val_recharge_proportion(data, name):
    """Validate recharge_proportion.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    rrp = data['params'][name]
    rzn = data['params']['ror_zone_names']

    c.check_type(param=rrp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)],
                 keys=range(1, 13))


###############################################################################
def val_recharge_limit(data, name):
    """Validate recharge_limit.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    rrl = data['params'][name]
    rzn = data['params']['ror_zone_names']

    c.check_type(param=rrl,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(rzn)],
                 keys=range(1, 13))


###############################################################################
def val_macropore_proportion(data, name):
    """Validate macropore_proportion.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    mpp = data['params'][name]
    mzn = data['params']['macropore_zone_names']

    c.check_type(param=mpp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(mzn)],
                 keys=range(1, 13))


###############################################################################
def val_macropore_limit(data, name):
    """Validate macropore_limit.

    1) type has to be a list of lists
    2) the top list requires length 12 (months)
    3) the bottom list requires lenght equal to the number of zones
    """
    mpl = data['params'][name]
    mzn = data['params']['macropore_zone_names']

    c.check_type(param=mpl,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(mzn)],
                 keys=range(1, 13))


###############################################################################
def val_interflow_params(data, name):
    """Validate interflow_params.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with 4 elements
    """
    ifp = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=ifp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[4],
                 keys=range(1, tot + 1))

    c.check_values_limits(values=[i for j in ifp.values() for i in j],
                          name=name,
                          low_l=0,
                          include_low=True)


###############################################################################
def val_soil_static_params(data, name):
    """Validate soil_static_params.

    1) type has to be a dictionary of lists of floats
    2) values have to be lists with length equal to the number of zones
    3) dictionary needs 4 keys: FC, WP, p, starting_SMD
    """
    ssp = data['params'][name]
    szn = data['params']['soil_zone_names']

    c.check_type(param=ssp,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(szn)],
                 keys=['FC', 'WP', 'p', 'starting_SMD'])


###############################################################################
def val_leakage(data, name):
    """Validate leakage.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    """
    lea = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=lea,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1))


###############################################################################
def val_soil_spatial(data, name):
    """Validate soil_spatial.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length equal to the number of zones
    """
    sos = data['params'][name]
    soz = data['params']['soil_zone_names']
    tot = data['params']['num_nodes']

    c.check_type(param=sos,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1),
                 len_list=[len(soz)])


###############################################################################
def val_lu_spatial(data, name):
    """Validate lu_spatial.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length equal to the number of zones
    """
    lus = data['params'][name]
    lzn = data['params']['land_zone_names']
    tot = data['params']['num_nodes']

    c.check_type(param=lus,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1),
                 len_list=[len(lzn)])


###############################################################################
def val_zr(data, name):
    """Validate ZR.

    1) type has to be a list of lists of floats
    2) dimensionds = 12 x number of zones
    """
    zrn = data['params'][name]
    lzn = data['params']['land_zone_names']

    c.check_type(param=zrn,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(lzn)],
                 keys=range(1, 13))


###############################################################################
def val_kc(data, name):
    """Validate KC.

    1) type has to be a list of lists of floats
    2) dimensionds = 12 x number of zones
    """
    kcn = data['params'][name]
    lzn = data['params']['land_zone_names']

    c.check_type(param=kcn,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 len_list=[len(lzn)],
                 keys=range(1, 13))


###############################################################################
def val_release_params(data, name):
    """Validate release_params.

    1) type has to be a dictionary of lists of floats
    2) all node ids have to be present
    3) values have to be lists with length 3
    """
    rpn = data['params'][name]
    tot = data['params']['num_nodes']

    c.check_type(param=rpn,
                 name=name,
                 t_types=data['specs'][name]['type'],
                 keys=range(1, tot + 1),
                 len_list=[3])


###############################################################################
def validate_params(data):
    """Validate all parameters using their specifications."""
    logging.info('\tValidating parameters')

    for function in [val_run_name,
                     val_log_file,
                     val_num_nodes,
                     val_node_areas,
                     val_start_date,
                     val_time_periods,
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
                     val_ror_zone_names,
                     val_ror_zone_mapping,
                     val_macropore_zone_names,
                     val_macropore_zone_mapping,
                     val_soil_zone_names,
                     val_land_zone_names,
                     val_free_throughfall,
                     val_max_canopy_storage,
                     val_snow_params,
                     val_rapid_runoff_params,
                     val_recharge_proportion,
                     val_recharge_limit,
                     val_macropore_proportion,
                     val_macropore_limit,
                     val_interflow_params,
                     val_soil_static_params,
                     val_leakage,
                     val_soil_spatial,
                     val_lu_spatial,
                     val_zr,
                     val_kc,
                     val_release_params]:

        param = function.__name__.replace('val_', '')
        function(data, param)
        logging.info('\t\t"%s" validated', param)

    logging.info('\tDone.')


###############################################################################
def validate_series(data):
    """Validate all time series using their specifications."""
    logging.info('\tValidating series')

    for function in [val_rainfall_ts,
                     val_pe_ts,
                     val_temperature_ts,
                     val_subroot_leakage_ts]:

        series = function.__name__.replace('val_', '')
        function(data, series)
        logging.info('\t\t"%s" validated', series)

    logging.info('\tDone.')
