# -*- coding: utf-8 -*-
from __future__ import print_function
"""SWAcMod input/output functions."""

# Standard Library
import os
import csv
import ast
import sys
import logging
import datetime

# Third Party Libraries
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import h5py
import numpy
from dateutil import parser
from tqdm import tqdm

# Internal modules
from . import utils as u
from . import checks as c
from . import validation as v
from . import finalization as f
from . import __version__


try:
    basestring
except NameError:
    basestring = str

if sys.version_info > (3,):
    long = int
    raw_input = input


###############################################################################
def start_logging(level=logging.INFO, path=None, run_name=None):
    """Start logging output.

    If path is None, run_name has to be provided.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(asctime)s --- (%(process)d) %(levelname)s - %(message)s"

    disc = """
    Although this program has been subjected to rigorous review, Groundwater
    Science Ltd. reserves the right to update the software as needed pursuant
    to further analysis and review. No warranty, expressed or implied, is made
    by Groundwater Science Ltd. as to the functionality of the software nor
    shall the fact of release constitute any such warranty. Furthermore, the
    tool is released on condition that Groundwater Science Ltd. shall not be
    held liable for any damages resulting from its authorised or unauthorised
    use.
    """

    if path is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = "%s_%s.log" % (run_name, now)
        path = os.path.join(u.CONSTANTS["OUTPUT_DIR"], name)
        logging.basicConfig(filename=path, format="%(message)s", level=level)
        logging.info("Version " + __version__)
        logging.info(disc)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=path, format=log_format, level=level)

    return path


###############################################################################
def format_time(diff):
    """Format a time difference for output."""
    secs = int(round(diff))
    return str(datetime.timedelta(0, secs))


###############################################################################
def print_progress(progress, total, prefix):
    """Print progress bar."""
    perc = progress * 1.0 / total
    perc_big = perc * 100
    spaces = int(perc_big / 2)
    progress_bar = "=" * spaces + ">" + " " * (50 - spaces)

    sys.stdout.write("\b" * 100 + "%s: [%s] %d%%\r" %
                     (prefix, progress_bar, perc_big))
    sys.stdout.flush()

    if progress == total:
        print()


###############################################################################
def load_yaml(filein):
    """Load a YAML file, lowercase its keys."""
    logging.debug("\t\tLoading %s", filein)

    with open(filein, "r") as fp:
        yml = yaml.load(fp, Loader=Loader)
    try:
        keys = yml.keys()
    except AttributeError:
        return yml

    for key in keys:
        if isinstance(key, basestring):
            if not key.islower():
                new_key = key.lower()
                value = yml.pop(key)
                yml[new_key] = value
    return yml


###############################################################################
def format_recharge_row(row):
    """Convert list of values to output string."""
    final = []
    for value in row:
        if value >= 0:
            string = "%.6e" % (value / 1000.0)
        else:
            string = "%.5e" % (value / 1000.0)
        splitter = "e+" if "e+" in string else "e-"
        split = string.split(splitter)
        if len(split[1]) == 2:
            split[1] = "0%s" % split[1]
        final.append(splitter.join(split))
    return " ".join(final) + "\n"


###############################################################################
def get_recharge_path(data):
    """Return the path of the recharge file."""
    fileout = "%s_recharge.rch" % data["params"]["run_name"]
    path = os.path.join(u.CONSTANTS["OUTPUT_DIR"], fileout)
    return path


###############################################################################
def get_output_path(data, file_format, output_dir, node=None, zone=None):
    """Return the path of the recharge file."""
    run = data["params"]["run_name"]

    if node is not None:
        _ = len(str(data["params"]["num_nodes"]))
        counter = eval("'%%0%dd' % _") % node
        fileout = "%s_n_%s.%s" % (run, counter, file_format)
    elif zone is not None:
        zones = data["params"]["reporting_zone_mapping"].values()
        _ = len(str(len(set(zones))))
        counter = eval("'%%0%dd' % _") % zone
        fileout = "%s_z_%s.%s" % (run, counter, file_format)

    path = os.path.join(output_dir, fileout)
    return path


###############################################################################
def check_open_files(data, file_format, output_dir):
    """Check if any of the scheduled output files can't be open."""
    paths = []
    if (data["params"]["output_recharge"]
            and data["params"]["gwmodel_type"] == "mfusg"):
        paths.append(get_recharge_path(data))

    if data["params"]["spatial_output_date"]:
        for p in get_spatial_path(data, output_dir):
            paths.append(p)

    for node in data["params"]["output_individual"]:
        paths.append(get_output_path(data, file_format, output_dir, node=node))

    zones = set(data["params"]["reporting_zone_mapping"].values())
    for zone in zones:
        if zone == 0:
            continue
        paths.append(get_output_path(data, file_format, output_dir, zone=zone))

    for path in paths:
        while True:
            try:
                fileobj = open(path, "a")
                fileobj.close()
                break
            except IOError:
                _ = raw_input('\nCannot write to "%s", make sure the file is '
                              "not in use then press Enter." % path)


###############################################################################
def dump_recharge_file(data, recharge):
    """Write recharge to file."""

    nnodes = data["params"]["num_nodes"]

    if data["params"]["recharge_node_mapping"] is None:
        nrchop, inrech, inirch = 3, 1, 0
    else:
        inirch = sum(1
                     for x in data["params"]["recharge_node_mapping"].values()
                     if x[0] != 0)
        nrchop, inrech = 2, 1

    fileout = "%s_recharge.rch" % data["params"]["run_name"]
    path = os.path.join(u.CONSTANTS["OUTPUT_DIR"], fileout)
    logging.info("\tDumping recharge file")

    with open(path, "w") as rech_file:
        rech_file.write("# MODFLOW-USGs Recharge Package\n")
        rech_file.write(" %d %d\n" % (nrchop, data["params"]["irchcb"]))
        if nrchop == 2:
            inrech = sum(
                1 for x in data["params"]["recharge_node_mapping"].values()
                if x[0] != 0)
            rech_file.write("%d\n" % (inrech))

        for per in range(len(data["params"]["time_periods"])):
            rech_file.write("%d %d\n" % (inrech, inirch))
            inirch = -1  # no longer needed

            rech_file.write("INTERNAL  1.000000e+000  (FREE)  -1  RECHARGE\n")
            row = []
            for node in range(data["params"]["num_nodes"]):
                if nrchop == 2:
                    if data["params"]["recharge_node_mapping"][node +
                                                               1] != [0]:
                        row.append(recharge[(nnodes * per) + node + 1])
                else:
                    row.append(recharge[(nnodes * per) + node + 1])

                if len(row) == data["params"]["nodes_per_line"]:
                    rech_file.write(format_recharge_row(row))
                    row = []
            if row:
                rech_file.write(format_recharge_row(row))

            if per == 0 and nrchop == 2:
                # write irch for destination node (only for first period)
                rech_file.write("INTERNAL  1              (FREE)  -1  IRCH\n")
                row = []
                for node in range(nnodes):
                    if data["params"]["recharge_node_mapping"][node +
                                                               1] != [0]:
                        row.append(
                            data["params"]["recharge_node_mapping"][node + 1])
                    if len(row) == data["params"]["nodes_per_line"]:
                        rech_file.write(" ".join(str(i[0])
                                                 for i in row) + "\n")
                        row = []
                if row:
                    rech_file.write(" ".join(str(i[0]) for i in row) + "\n")

###############################################################################


def dump_mf96_recharge_file(data, recharge):
    """Write recharge to mf96 rch file."""

    nnodes = data["params"]["num_nodes"]
    nrchop, inrech, inirch = 3, 1, -1
    fileout = "%s.rch" % data["params"]["run_name"]
    path = os.path.join(u.CONSTANTS["OUTPUT_DIR"], fileout)
    logging.info("\tDumping recharge file")
    fmt_float = "{:12.4f}"
    npl = 6
    with open(path, "w") as rech_file:
        rech_file.write("{0:10d}{1:10d}\n".format(nrchop,
                                                  data["params"]["irchcb"]))
        for per in range(len(data["params"]["time_periods"])):
            rech_file.write("{0:10d}{1:10d}\n".format(inrech, inirch))
            rech_file.write("        18    1.0000(6g12.4)" +
                            "                    -3\n")
            row = ""
            i_row = 0
            for node in range(data["params"]["num_nodes"]):
                row += fmt_float.format(recharge[(nnodes * per) + node + 1])
                i_row += 1
                if i_row == npl:
                    rech_file.write(row + "\n")
                    row = ""
                    i_row = 0
            if row:
                rech_file.write(row + "\n")


def get_spatial_path(data, output_dir):
    ###########################################################################
    """Get the path of the spatial data output CSV file."""
    if data["params"]["spatial_output_date"] == "mean":
        nam_list = ["mean"] + ['month' + str(i + 1) for i in range(12)]
    else:
        nam_list = [str(data["params"]["spatial_output_date"].date())]
    run = data["params"]["run_name"]
    path = [
        os.path.join(output_dir, "%sSpatial%s.csv" % (run, string))
        for string in nam_list
    ]
    return path


###############################################################################
def dump_spatial_output(data, spatials, output_dir, reduced=False):
    """Write spatial output to file."""

    areas = data["params"]["node_areas"]
    paths = get_spatial_path(data, output_dir)
    mult = data["params"]["output_fac"] * 0.001
    fn = get_row_spatial_reduced if reduced else get_row_spatial

    for isp, path in enumerate(paths):

        string = os.path.split(path)[1]
        logging.info("\tDumping spatial output for %s", string)
        print("\t-" + str(1 + isp) + ' of ' + str(len(paths)))
        with open(path, "w") as outfile:
            header = ["Node"]
            header += [
                i[0] for i in u.CONSTANTS["BALANCE_CONVERSIONS"]
                if i[0] not in ["DATE", "nDays"]
            ]
            if reduced:
                header += [
                    i for i in header
                    if u.CONSTANTS["BALANCE_CONVERSIONS"][i][2]
                ]
            writer = csv.writer(outfile,
                                delimiter=",",
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(header)

            writer.writerows([[node, areas[node]] +
                              fn({key1: val1[isp]
                                  for key1, val1 in spatials[node].items()},
                                 mult) for node in spatials.keys()])
            writer = None

###############################################################################


def dump_sfr_output(sfr):
    from swacmod.model import write_sfr

    logging.info("\tDumping SFR file")
    write_sfr(sfr)
    # sfr.write_file()


###############################################################################


def dump_evt_output(evt, f=None):
    """
        Write the package file.

        Returns
        -------
    None

        """
    logging.info("\tDumping EVT file")
    nrow, ncol, nlay, nper = evt.parent.nrow_ncol_nlay_nper
    if f is not None:
        f_evt = f
    else:
        f_evt = open(evt.fn_path, "w")
    evt.heading = "# EVT package for  MODFLOW-USG, generated by SWAcMod."
    f_evt.write("{0:s}\n".format(evt.heading))
    f_evt.write("{0:10d}{1:10d}\n".format(evt.nevtop, evt.ipakcb))

    if evt.nevtop == 2:
        mxndevt = evt.ievt[0].shape[0]
        f_evt.write("{0:10d}\n".format(mxndevt))

    for n in range(nper):
        insurf, surf = evt.surf.get_kper_entry(n)
        inevtr, evtr = evt.evtr.get_kper_entry(n)
        inexdp, exdp = evt.exdp.get_kper_entry(n)
        inievt, ievt = evt.ievt.get_kper_entry(n)
        if inievt > 0:
            inievt = mxndevt
        comment = "Evapotranspiration  dataset 5 for stress period " + str(n +
                                                                           1)
        f_evt.write("{0:10d}{1:10d}{2:10d}{3:10d} # {4:s}\n".format(
            insurf, inevtr, inexdp, inievt, comment))
        if insurf >= 0:
            f_evt.write(surf)
        if inevtr >= 0:
            f_evt.write(evtr)
        if inexdp >= 0:
            f_evt.write(exdp)
        if evt.nevtop == 2 and inievt >= 0:
            f_evt.write(ievt)
    f_evt.close()


###############################################################################
def dump_water_balance(data,
                       output,
                       file_format,
                       output_dir,
                       node=None,
                       zone=None,
                       reduced=False):
    """Write report to file."""
    areas = data["params"]["node_areas"]
    periods = data["params"]["time_periods"]

    if node:
        string = "for node %d" % node
        area = areas[node]
    elif zone:
        string = "for zone %d" % zone
        items = data["params"]["reporting_zone_mapping"].items()
        area = sum([areas[i[0]] for i in items if i[1] == zone])

    path = get_output_path(data, file_format, output_dir, node=node, zone=zone)
    logging.info("\tDumping water balance %s", string)
    aggregated = u.aggregate_output(data, output, method="sum")
    fac = data["params"]["output_fac"]
    mult = area * fac / 1000 if node else fac / 1000

    if file_format == "csv":
        with open(path, "w") as outfile:
            if reduced:
                header = [
                    i[0] for i in u.CONSTANTS["BALANCE_CONVERSIONS"] if i[2]
                ]
            else:
                header = [i[0] for i in u.CONSTANTS["BALANCE_CONVERSIONS"]]
            writer = csv.writer(outfile,
                                delimiter=",",
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(header)
            for num, period in enumerate(periods):
                row = get_row_balance(aggregated, num, reduced, mult)
                row = row.tolist()
                row.insert(0, aggregated["date"][num])
                row.insert(1, period[1] - period[0])
                row.insert(2, area)
                writer.writerow(row)

    elif file_format in ["hdf5", "h5"]:
        final = None
        for num, period in enumerate(periods):
            row = get_row_balance(aggregated, num, reduced, mult)
            if not reduced:
                row = numpy.insert(row, 0, period[1] - period[0])
                row = numpy.insert(row, 1, area)
            if final is None:
                final = row
            else:
                final = numpy.vstack((final, row))
        try:
            os.remove(path)
        except OSError:
            pass
        with h5py.File(path) as outfile:
            root = "swacmod_output"
            outfile.create_dataset(root, data=final, compression="gzip")


###############################################################################
def get_row_spatial(vector, mult):
    """Get a row of data for output."""

    row = [
        vector[key] * mult for key in u.CONSTANTS["COL_ORDER"]
        if key not in
        ["date", "unutilised_pe", "k_slope", "rapid_runoff_c"]
    ]

    return row


###############################################################################
def get_row_spatial_reduced(vector, mult):
    """Get a reduced row of data for output."""

    keys = [
        "combined_recharge", "combined_str", "combined_ae", "unutilised_pe"
    ]
    row = [vector[key] * mult for key in keys]

    return row


###############################################################################
def get_row_balance(aggregated, num, reduced, mult):
    """Get a row of data for output."""
    if reduced:
        keys = [
            "combined_recharge", "combined_str", "combined_ae", "evt"
        ]
        row = [aggregated[key][num] for key in keys]
    else:
        row = [
            aggregated[key][num] for key in u.CONSTANTS["COL_ORDER"] if key
            not in ["date", "unutilised_pe", "k_slope", "rapid_runoff_c"]
        ]

    row = numpy.array(row) * mult
    return row


###############################################################################
def convert_all_yaml_to_csv(specs_file, input_dir):
    """Convert all YAML files to CSV for parameters that accept this option."""
    specs = load_yaml(specs_file)
    to_csv = [
        i for i in specs
        if "alt_format" in specs[i] and "csv" in specs[i]["alt_format"]
    ]
    for filein in os.listdir(input_dir):
        if filein.endswith(".yml"):
            path = os.path.join(input_dir, filein)
            loaded = load_yaml(path)
            param = loaded.items()[0][0]
            if len(loaded.items()) == 1 and param in to_csv:
                print(path)
                convert_one_yaml_to_csv(path)


###############################################################################
def convert_one_yaml_to_csv(filein):
    """Convert a YAML file to a CSV file.

    The opposite function is tricky, as CSV reader does not understand types.
    """
    fileout = filein.replace(".yml", ".csv")
    readin = load_yaml(filein).items()[0][1]

    with open(fileout, "wb") as csvfile:
        writer = csv.writer(csvfile,
                            delimiter=",",
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        if isinstance(readin, dict):
            for item in readin.items():
                row = [item[0]]
                if isinstance(item[1], list):
                    row += item[1]
                elif isinstance(item[1], (float, int, long, str)):
                    row += [item[1]]
                else:
                    print("Could not recognize object: %s" % type(item[1]))
                writer.writerow(row)
        elif isinstance(readin, list):
            for item in readin:
                row = []
                if isinstance(item, list):
                    row += item
                elif isinstance(item, (float, int, long, str)):
                    row += [item]
                else:
                    print("Could not recognize object: %s" % type(item))
                writer.writerow(row)


###############################################################################

def categorise_param(param, value):
    time_period_params = [
        "infiltration_limit_ts",
        "interflow_decay_ts",
        "percolation_rejection_ts",
        "rainfall_ts",
        "subroot_leakage_ts",
        "swdis_ts",
        "swabs_ts",
        "temperature_ts",
        "tmax_c_ts",
        "tmin_c_ts",
        "pe_ts",
        "windsp_ts",
    ]
    non_time_period_params = [
        "canopy_zone_mapping",
        "canopy_zone_names",
        "evt_parameters",
        "free_throughfall",
        "infiltration_limit",
        "init_interflow_store",
        "interflow_decay",
        "interflow_store_bypass",
        "interflow_zone_mapping",
        "interflow_zone_names",
        "kc",
        "landuse_zone_names",
        "lu_spatial",
        "macropore_activation",
        "macropore_limit",
        "macropore_proportion",
        "macropore_recharge",
        "macropore_zone_mapping",
        "macropore_zone_names",
        "max_canopy_storage",
        "node_areas",
        "node_xy",
        "pe_zone_mapping",
        "pe_zone_names",
        "percolation_rejection",
        "rainfall_zone_mapping",
        "rainfall_zone_names",
        "rapid_runoff_params",
        "rapid_runoff_zone_mapping",
        "rapid_runoff_zone_names",
        "raw",
        "recharge_attenuation_params",
        "recharge_node_mapping",
        "reporting_zone_mapping",
        "reporting_zone_names",
        "routing_topology",
        "snow_params_simple",
        "soil_static_params",
        "soil_zone_names",
        "subroot_zone_names",
        "subsoilzone_leakage_fraction",
        "snow_params_complex",
        "soil_spatial",
        "subroot_zone_mapping",
        "smd",
        "swdis_locs",
        "sw_params",
        "swabs_locs",
        "swrecharge_limit",
        "swrecharge_proportion",
        "swrecharge_zone_mapping",
        "swrecharge_zone_names",
        "taw",
        "temperature_zone_mapping",
        "temperature_zone_names",
        "time_periods",
        "tmax_c_zone_mapping",
        "tmax_c_zone_names",
        "tmin_c_zone_mapping",
        "tmin_c_zone_names",
        "windsp_zone_mapping",
        "windsp_zone_names",
        "zr",
    ]
    if param in time_period_params:
        return "time_peroiod_param"
    if param in non_time_period_params:
        return "non_time_peroiod_param"
    print(f"Uncategorised param. param = {param}. value = {value}.")
    return "uncategorised_param"

def load_params_from_yaml(
        specs_file=u.CONSTANTS["SPECS_FILE"],
        input_file=u.CONSTANTS["INPUT_FILE"],
        input_dir=u.CONSTANTS["INPUT_DIR"],
):
    """Load model specifications, parameters and time series."""
    logging.info("\tLoading parameters and time series")

    specs = load_yaml(specs_file)
    params = load_yaml(input_file)

    no_list = ([
        "node_areas",
        "free_throughfall",
        "max_canopy_storage",
        "subsoilzone_leakage_fraction",
    ] + [i for i in params if "zone_names" in i] + [
        i for i in params if ("zone_mapping" in i or "_locs" in i) and i not in
        ["rainfall_zone_mapping", "pe_zone_mapping", "subroot_zone_mapping"]
    ])

    for param in tqdm(params, desc="SWAcMod load params     "):
        if isinstance(params[param], str) and "alt_format" in specs[param]:
            absolute = os.path.join(input_dir, params[param])
            param_category = categorise_param(param, params[param])
            ext = params[param].split(".")[-1]
            if ext not in specs[param]["alt_format"]:
                continue
            if ext == "csv":
                try:
                    reader = csv.reader(open(absolute, "r"))
                except IOError as err:
                    msg = "Could not import %s: %s" % (param, err)
                    raise u.InputOutputError(msg)
                try:
                    rows = [[ast.literal_eval(j) for j in row]
                            for row in reader]

                    if param.endswith("_ts") or param == "time_periods":
                        params[param] = rows
                    else:
                        if param not in no_list:
                            params[param] = dict(
                                (row[0], row[1:]) for row in rows)
                        else:
                            params[param] = dict(
                                (row[0], row[1]) for row in rows)
                except IndexError as err:
                    msg = "Could not import %s: %s" % (param, err)
                    raise u.InputOutputError(msg)
            elif ext == "yml":
                try:
                    params[param] = load_yaml(absolute)[param]
                except (IOError, KeyError) as err:
                    msg = "Could not import %s: %s" % (param, err)
                    raise u.InputOutputError(msg)
    for key in specs:
        if key not in params:
            params[key] = None

    series = {}
    keys = [i for i in params if i.endswith("_ts")]
    for key in keys:
        series[key] = params.pop(key)

    data = {"specs": specs, "series": series, "params": params}

    return data


###############################################################################
def load_and_validate(specs_file, input_file, input_dir):
    """Load, finalize and validate model parameters and time series."""
    data = load_params_from_yaml(specs_file=specs_file,
                                 input_file=input_file,
                                 input_dir=input_dir)

    f.finalize_params(data)
    f.finalize_series(data)
    c.check_required(data)
    v.validate_params(data)
    v.validate_series(data)

    return data


###############################################################################
def load_results():
    """Load 'Calculations' sheet."""
    check = dict((k, []) for k in u.CONSTANTS["COL_ORDER"])

    with open(u.CONSTANTS["TEST_RESULTS_FILE"], "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            values = []
            for num, cell in enumerate(row):
                if num == 0:
                    values.append(parser.parse(cell))
                else:
                    values.append(float(cell))
            for num, value in enumerate(values):
                check[u.CONSTANTS["COL_ORDER"][num]].append(value)

    return check
