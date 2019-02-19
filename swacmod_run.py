#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""SWAcMod main."""

# Standard Library
import os
import sys
import time
import random
import logging
import argparse
from multiprocessing import Process, Manager, freeze_support, Array, Queue

# Third Party Libraries
import numpy as np
from tqdm import tqdm

# Internal modules
from swacmod import utils as u
from swacmod import input_output as io

# win fix
sys.maxint = 2 ** 63 - 1

# sentinel for iteration count
SENTINEL = 1

# Compile and import model
u.compile_model()
from swacmod import model as m


class Worker:
    def __init__(self, name, result_queue, process, verbose=True):
        self.name = name
        self.result_queue = result_queue
        self.process = process
        self.verbose = verbose

    def start(self):
        if self.verbose:
            print("starting ", self.name)
        self.process.start()

    def join(self):
        if self.verbose:
            print("join ", self.name)
        self.process.join()


###############################################################################
def aggregate_reporting(reporting):
    """Aggregate zones across processes."""
    logging.info("\tAggregating reporting across processes")
    new_rep = {}
    for key in reporting.keys():
        if key[1] not in new_rep:
            new_rep[key[1]] = reporting[key].copy()
        else:
            for key2 in reporting[key]:
                new_rep[key[1]][key2] += reporting[key][key2]
    return new_rep


###############################################################################
def get_output(data, node):
    """Run the model."""
    logging.debug("\tRunning model for node %d", node)

    start = time.time()

    output = {}
    for function in [
        m.get_precipitation,
        m.get_pe,
        m.get_pefac,
        m.get_canopy_storage,
        m.get_net_pefac,
        m.get_precip_to_ground,
        m.get_snowfall_o,
        m.get_rainfall_o,
        m.get_snow,
        m.get_net_rainfall,
        m.get_rawrew,
        m.get_tawtew,
        m.get_ae,
        m.get_unutilised_pe,
        m.get_rejected_recharge,
        m.get_perc_through_root,
        m.get_subroot_leak,
        m.get_interflow_bypass,
        m.get_interflow_store_input,
        m.get_interflow,
        m.get_recharge_store_input,
        m.get_recharge,
        m.get_swabs,
        m.get_swdis,
        m.get_combined_str,
        m.get_combined_ae,
        m.get_evt,
        m.get_average_in,
        m.get_average_out,
        m.get_change,
        m.get_balance,
    ]:

        columns = function(data, output, node)
        output.update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()

    logging.debug("\tNode %d done (%dms).", node, (end - start) * 1000)
    return output


###############################################################################
def run_process(
    num,
    ids,
    data,
    test,
    reporting_agg,
    recharge_agg,
    runoff_agg,
    evtr_agg,
    recharge,
    runoff,
    log_path,
    level,
    file_format,
    reduced,
    output_dir,
    spatial,
    spatial_index,
    reporting,
    single_node_output,
    q,
):
    """Run model for a chunk of nodes."""
    io.start_logging(path=log_path, level=level)
    logging.info("Process %d started (%d nodes)", num, len(ids))
    nnodes = data["params"]["num_nodes"]

    for node in ids:

        q.put(SENTINEL)

        rep_zone = data["params"]["reporting_zone_mapping"][node]
        if rep_zone != 0:
            output = get_output(data, node)
            logging.debug("RAM usage is %.2fMb", u.get_ram_usage_for_process())
            if not test:
                if node in data["params"]["output_individual"]:
                    # if this node for individual output then preserve
                    single_node_output[node] = output.copy()
                key = (num, rep_zone)
                area = data["params"]["node_areas"][node]
                if key not in reporting_agg:
                    reporting_agg[key] = m.aggregate(output, area)
                else:
                    reporting_agg[key] = m.aggregate(
                        output, area, reporting=reporting_agg[key]
                    )

                if data["params"]["output_recharge"]:
                    rech = {"recharge": output["combined_recharge"].copy()}
                    for i, p in enumerate(
                        u.aggregate_output_col(data, rech, "recharge",
                                               method="average")):
                        recharge_agg[(nnodes * i) + int(node)] = p
                    rech = None

                if data["params"]["swrecharge_process"] == "enabled":

                    rech = output["combined_recharge"].copy()
                    for i, p in enumerate(rech):
                        recharge[(nnodes * i) + int(node)] = p
                    rech = None

                    ro = output["combined_str"].copy()
                    for i, p in enumerate(ro):
                        runoff[(nnodes * i) + int(node)] = p
                    ro = None

                if (data["params"]["output_sfr"] or
                    data["params"]["excess_sw_process"] == "enabled"):
                    ro = {"runoff": output["combined_str"].copy()}
                    for i, p in enumerate(
                        u.aggregate_output_col(data, ro, "runoff",
                                               method="average")):
                        runoff_agg[(nnodes * i) + int(node)] = p
                    ro = None

                if data["params"]["output_evt"]:
                    evt = {"evtr": output["unutilised_pe"].copy()}
                    for i, p in enumerate(
                        u.aggregate_output_col(data, evt, "evtr",
                                               method="average")):
                        evtr_agg[(nnodes * i) + int(node)] = p
                    evt = None

                if data["params"]["spatial_output_date"]:
                    spatial[node] = m.aggregate(output, area,
                                                index=spatial_index)

    logging.info("Process %d ended", num)

    return (
        reporting_agg,
        recharge_agg,
        spatial,
        runoff_agg,
        evtr_agg,
        recharge,
        runoff,
        reporting,
        single_node_output,
    )


###############################################################################

# this stuff stranded here for windows - multiprocessing cannot handle
#  - functions not in the top level
#  - not pickleable objects as argumnets to that function :(


def listener(q, total):
    pbar = tqdm(total=total, desc="SWAcMod Parallel        ")
    for item in iter(q.get, None):
        pbar.update()
    pbar.close()


###############################################################################


def run(test=False, debug=False, file_format=None, reduced=False, skip=False):
    """Run model for all nodes."""
    times = {"start_of_run": time.time()}

    manager = Manager()
    reporting_agg = manager.dict()
    reporting_agg2 = {}
    reporting = manager.dict()
    spatial = manager.dict()

    single_node_output = manager.dict()
    specs_file = u.CONSTANTS["SPECS_FILE"]
    if test:
        input_file = u.CONSTANTS["TEST_INPUT_FILE"]
        input_dir = u.CONSTANTS["TEST_INPUT_DIR"]
    else:
        input_file = u.CONSTANTS["INPUT_FILE"]
        input_dir = u.CONSTANTS["INPUT_DIR"]

    level = logging.DEBUG if debug else logging.INFO
    params = io.load_yaml(input_file)
    log_path = io.start_logging(level=level, run_name=params["run_name"])

    print('\nStart "%s"' % params["run_name"])
    logging.info("Start SWAcMod run")

    data = io.load_and_validate(specs_file, input_file, input_dir)
    if not skip:
        io.check_open_files(data, file_format, u.CONSTANTS["OUTPUT_DIR"])

    per = len(data["params"]["time_periods"])
    nnodes = data["params"]["num_nodes"]
    len_rch_agg = (nnodes * per) + 1
    recharge_agg = Array("f", len_rch_agg)  # recharge by output period (agg)
    runoff_agg = Array("f", len_rch_agg)
    runoff_recharge_agg = np.zeros((len_rch_agg))
    evtr_agg = Array("f", len_rch_agg)
    days = len(data["series"]["date"])
    len_rch = (nnodes * days) + 1

    if params["swrecharge_process"] == "enabled":
        recharge = Array("f", len_rch)
        runoff = Array("f", len_rch)
    else:
        recharge = Array("f", 1)
        runoff = Array("f", 1)

    ids = range(1, nnodes + 1)
    random.shuffle(list(ids))
    chunks = np.array_split(ids, data["params"]["num_cores"])
    times["end_of_input"] = time.time()

    if data["params"]["spatial_output_date"] == "mean":
        spatial_index = range(per)
    elif data["params"]["spatial_output_date"] is not None:
        spatial_index = (
            data["params"]["spatial_output_date"] - data["params"]["start_date"]
        ).days
    else:
        spatial_index = None

    workers = []
    q = Queue()
    lproc = Process(target=listener, args=(q, nnodes))
    lproc.start()

    for process, chunk in enumerate(chunks):

        if chunk.size == 0:
            continue

        proc = Process(
            target=run_process,
            args=(
                process,
                chunk,
                data,
                test,
                reporting_agg,
                recharge_agg,
                runoff_agg,
                evtr_agg,
                recharge,
                runoff,
                log_path,
                level,
                file_format,
                reduced,
                u.CONSTANTS["OUTPUT_DIR"],
                spatial,
                spatial_index,
                reporting,
                single_node_output,
                q,
            ),
        )

        workers.append(Worker("worker%d" % process, q, proc, verbose=False))

    for p in workers:
        p.start()

    for p in workers:
        p.join()

    q.put(None)
    lproc.join()

    times["end_of_model"] = time.time()

    if not test:

        # aggregate over processes
        reporting_agg = aggregate_reporting(reporting_agg)

        if params["swrecharge_process"] == "enabled":

            for cat in data["params"]["reporting_zone_mapping"].values():
                reporting_agg2[cat] = {}

            # ended up needing this for catchment output - bit silly
            runoff_recharge = np.array(runoff).copy()
            runoff_recharge = np.frombuffer(runoff.get_obj(), dtype=np.float32).copy()

            # do RoR
            runoff, recharge = m.do_swrecharge_mask(data, runoff, recharge)
            # get RoR for cat output purposes
            runoff_recharge -= np.frombuffer(runoff.get_obj(), dtype=np.float32).copy()

            # aggregate amended recharge & runoff arrays by output periods
            for node in tqdm(
                list(m.all_days_mask(data).nodes), desc="Aggregating Fluxes      "
            ):
                # get indices of output for this node
                idx = range(node, (nnodes * days) + 1, nnodes)

                tmp = np.frombuffer(recharge.get_obj(), dtype=np.float32).copy()
                rch_array = np.array(tmp[idx], dtype=np.float64, copy=True)
                tmp = np.frombuffer(runoff.get_obj(), dtype=np.float32).copy()
                ro_array = np.array(tmp[idx], dtype=np.float64, copy=True)
                ror_array = np.array(runoff_recharge[idx], dtype=np.float64, copy=True)

                # aggregate single node of recharge array
                rch_agg = u.aggregate_array(data, rch_array)
                # aggregate single node of runoff array
                ro_agg = u.aggregate_array(data, ro_array)
                # aggregate single node of runoff rech array
                ror_agg = u.aggregate_array(data, ror_array)

                for period, val in enumerate(rch_agg):
                    recharge_agg[(nnodes * period) + int(node)] = val
                for period, val in enumerate(ro_agg):
                    runoff_agg[(nnodes * period) + int(node)] = val
                for period, val in enumerate(ror_agg):
                    runoff_recharge_agg[(nnodes * period) + int(node)] = val

                # amend catchment output values
                rep_zone = data["params"]["reporting_zone_mapping"][node]
                area = data["params"]["node_areas"][node]
                rech = {"combined_recharge": rch_array}
                sw = {"combined_str": ro_array}
                ror = {"runoff_recharge": ror_array}

                if "runoff_recharge" not in reporting_agg2[rep_zone]:
                    reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(ror, area)
                else:
                    reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
                        ror, area, reporting=reporting_agg2[rep_zone]["runoff_recharge"]
                    )

                # check for single node
                if node in data["params"]["output_individual"]:
                    # amend single_node_output with ror values
                    # this method required due to upstream bug
                    tmp_node = single_node_output[node]
                    tmp_node["runoff_recharge"] = ror_array.copy()
                    tmp_node["combined_recharge"] = np.copy(rch_array)
                    tmp_node["combined_str"] = np.copy(ro_array)
                    single_node_output[node] = tmp_node

            # copy new bits into cat output
            term = "runoff_recharge"
            for cat in reporting_agg2:
                reporting_agg[cat]["combined_recharge"] += reporting_agg2[cat][term][
                    term
                ]
                reporting_agg[cat]["combined_str"] -= reporting_agg2[cat][term][term]
                reporting_agg[cat]["runoff_recharge"] = reporting_agg2[cat][term][term]

        if data["params"]["output_sfr"]:
            # copy array into this fn to avoid unit conversion
            sfr = m.get_sfr_file(data, np.copy(np.array(runoff_agg)))

        if data["params"]["output_evt"]:
            if data["params"]["excess_sw_process"] == "enabled":
                evt = m.get_evt_file(data, evtr_agg -
                                     np.copy(np.array(runoff_agg)))
            else:
                evt = m.get_evt_file(data, evtr_agg)

        print("\nWriting output files:")
        if not skip:
            io.check_open_files(data, file_format, u.CONSTANTS["OUTPUT_DIR"])

        for num, key in enumerate(reporting_agg.keys()):
            print("\t- Report file (%d of %d)" % (num + 1, len(reporting_agg.keys())))
            io.dump_water_balance(
                data,
                reporting_agg[key],
                file_format,
                u.CONSTANTS["OUTPUT_DIR"],
                zone=key,
                reduced=reduced,
            )

        for node in list(data["params"]["output_individual"]):
            print("\t- Node output file")
            io.dump_water_balance(
                data,
                single_node_output[node],
                file_format,
                u.CONSTANTS["OUTPUT_DIR"],
                node=node,
                reduced=reduced,
            )

        if data["params"]["output_recharge"]:
            print("\t- Recharge file")
            if data['params']['gwmodel_type'] == 'mfusg':
                io.dump_recharge_file(data, recharge_agg)
            elif data['params']['gwmodel_type'] == 'mf6':
                m.get_rch_file(data, recharge_agg).write()

        if data["params"]["spatial_output_date"]:
            print("\t- Spatial file")
            io.dump_spatial_output(
                data, spatial, u.CONSTANTS["OUTPUT_DIR"], reduced=reduced
            )

        if data["params"]["output_sfr"]:
            print("\t- SFR file")
            if data['params']['gwmodel_type'] == 'mfusg':
                io.dump_sfr_output(sfr)
            elif data['params']['gwmodel_type'] == 'mf6':
                sfr.write()

        if data["params"]["output_evt"]:
            print("\t- EVT file")
            if data['params']['gwmodel_type'] == 'mfusg':
                io.dump_evt_output(evt)
            elif data['params']['gwmodel_type'] == 'mf6':
                evt.write()

    times["end_of_run"] = time.time()

    diff = times["end_of_run"] - times["start_of_run"]
    total = io.format_time(diff)

    per_node = int(round(diff * 1000 / data["params"]["num_nodes"]))

    cores = (
        "%d cores" % data["params"]["num_cores"]
        if data["params"]["num_cores"] != 1
        else "1 core"
    )

    print("\nPerformance (%s)" % cores)
    print(
        "Input time:  %s"
        % io.format_time(times["end_of_input"] - times["start_of_run"])
    )
    print(
        "Run time:    %s"
        % io.format_time(times["end_of_model"] - times["end_of_input"])
    )
    print(
        "Output time: %s" % io.format_time(times["end_of_run"] - times["end_of_model"])
    )
    print("Total time:  %s (%d msec/node)" % (total, per_node))
    print("")

    logging.info("End SWAcMod run")


###############################################################################
if __name__ == "__main__":
    freeze_support()

    # Parser for command line arguments
    DESCRIPTION = """
    Invoke this script to run SWAcMod.
    e.g. 'python swacmod_run.py'"""
    FORM = argparse.RawTextHelpFormatter

    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    PARSER.add_argument("-t", "--test", help="run with no output", action="store_true")
    PARSER.add_argument("-d", "--debug", help="verbose log", action="store_true")
    PARSER.add_argument("-r", "--reduced", help="reduced output", action="store_true")
    PARSER.add_argument(
        "-i", "--input_yml", help="path to input yaml file inside input directory"
    )
    PARSER.add_argument("-o", "--output_dir", help="path to output directory")
    PARSER.add_argument(
        "-f",
        "--format",
        help="output file format",
        choices=["hdf5", "h5", "csv"],
        default="csv",
    )
    PARSER.add_argument(
        "-s",
        "--skip_prompt",
        help="skip user prompts and warnings",
        action="store_true",
    )

    ARGS = PARSER.parse_args()
    if ARGS.input_yml:
        if not ARGS.input_yml.endswith(".yml"):
            print(
                '\nError: use "-i" or "--input_yml" to specify the path '
                'to "input.yml"\n'
            )
            sys.exit()
        u.CONSTANTS["INPUT_FILE"] = ARGS.input_yml
        u.CONSTANTS["INPUT_DIR"] = os.path.dirname(ARGS.input_yml)
    if ARGS.output_dir:
        u.CONSTANTS["OUTPUT_DIR"] = ARGS.output_dir
    if not os.path.exists(u.CONSTANTS["OUTPUT_DIR"]):
        os.makedirs(u.CONSTANTS["OUTPUT_DIR"])

    if ARGS.debug:
        run(
            test=ARGS.test,
            debug=ARGS.debug,
            file_format=ARGS.format,
            reduced=ARGS.reduced,
            skip=ARGS.skip_prompt,
        )
    else:
        try:
            run(
                test=ARGS.test,
                debug=ARGS.debug,
                file_format=ARGS.format,
                reduced=ARGS.reduced,
                skip=ARGS.skip_prompt,
            )
        except Exception as err:
            logging.error(err.__repr__())
            print("ERROR: %s" % err)
            print("")
