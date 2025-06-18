from __future__ import print_function
import swacmod.feature_flags as ff
import os
import sys
import time
import random
import logging
import argparse
import multiprocessing as mp
if not ff.disable_multiprocessing:
    from multiprocessing.heap import Arena
else:
    import queue
import swacmod.timer as timer
import mmap
import gc
import numpy as np
from tqdm import tqdm
from swacmod import utils as u
from swacmod import input_output as io
import swacmod.input_files.input_file_reader as input_file_reader
import swacmod.version_information as version_information
from swacmod.input_files.input_files_version_2.default_file_resource import DefaultFileResource
import swacmod.stuff_to_be_named_later as stuff_module
import swacmod.output_functions as output_functions
from swacmod import model as m
import swacmod.model_numpy as model_numpy
import swacmod.historical_solute as historical_solute
import swacmod.solute as solute
from swacmod.environment import Environment
import swacmod.runoff_recharge as runoff_recharge_module

# win fix
sys.maxint = 2**63 - 1

# sentinel for iteration count
SENTINEL = 1

def anonymous_arena_init(self, size, fd=-1):
    "Create Arena using an anonymous memory mapping."
    self.size = size
    self.fd = fd  # still kept but is not used !
    self.buffer = mmap.mmap(-1, self.size)

# monkey patch for anonymous memory mapping python 3
if not ff.disable_multiprocessing:
    if sys.version_info > (3,):
        if mp.get_start_method() == 'fork':
            Arena.__init__ = anonymous_arena_init

class Worker:
    "mp worker"
    def __init__(self, name, result_queue, process, verbose=True):
        self.name = name
        self.result_queue = result_queue
        self.process = process
        self.verbose = verbose

    def start(self):
        "start worker"
        if self.verbose:
            print("starting ", self.name)
        self.process.start()

    def join(self):
        "join worker"
        if self.verbose:
            print("join ", self.name)
        self.process.join()

def aggregate_reporting(reporting):
    """Aggregate reporting zones across processes."""
    logging.info("\tAggregating reporting across processes")
    new_rep = {}
    for key in reporting.keys():
        if key[1] not in new_rep:
            new_rep[key[1]] = reporting[key].copy()
        else:
            for key2 in reporting[key]:
                new_rep[key[1]][key2] += reporting[key][key2]
    return new_rep

def compare_methods(unoptimised, optimised):
    result = lambda data, output, node: _compare_methods(unoptimised, optimised, data, output, node)
    result.__name__ = unoptimised.__name__
    return result

def _compare_methods(unoptimised, optimised, data, output, node):
    comparison_time_switcher = data["comparison_time_switcher"]
    timer.switch_to(comparison_time_switcher, f"{unoptimised.__name__} (unoptimised)")
    unoptimised_result = unoptimised(data, output, node)

    timer.switch_to(comparison_time_switcher, f"{unoptimised.__name__} (optimised)")
    optimised_result = optimised(data, output, node)
    timer.switch_off(comparison_time_switcher)

    np.testing.assert_equal(unoptimised_result, optimised_result)

    return optimised_result

def get_output(data, node, time_switcher):
    """Run the model."""
    logging.debug("\tRunning model for node %d", node)

    start = time.time()

    output = {}

    methods = [
        m.get_precipitation,
        m.get_pe,
        m.get_pefac,
        m.get_canopy_storage,
        m.get_net_pefac,
        m.get_precip_to_ground,
        m.get_snowfall_o,
        m.get_rainfall_o,
        m.get_snow_simple,
        m.get_snow_complex,
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
        model_numpy.get_swabs,
        model_numpy.get_swdis,
        m.get_combined_str,
        m.get_recharge_store_input,
        m.get_recharge,
        m.get_combined_ae,
        m.get_evt,
        m.get_average_in,
        m.get_average_out,
        m.get_change,
        m.get_balance,
        historical_solute.get_historical_solute,
        solute.get_solute,
    ]

    for function in methods:

        timer.switch_to(time_switcher, function.__name__)
        columns = function(data, output, node)
        output.update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()

    logging.debug("\tNode %d done (%dms).", node, (end - start) * 1000)
    return output

def run_process(
        env,
        num,
        ids,
        data,
        test,
        stuff,
        recharge_agg,
        runoff_agg,
        evtr_agg,
        solute_aggregation,
        stream_solute_aggregation,
        solute_mi_aggregation,
        recharge,
        runoff,
        log_path,
        level,
        spatial_index,
        q,
        pbar=None
):
    """Run model for a chunk of nodes."""
    timer_token_run_process = timer.make_time_switcher()
    timer.switch_to(timer_token_run_process, "run_main > run > run_process")
    time_switcher = timer.make_time_switcher()
    comparison_time_switcher = timer.make_time_switcher()
    data["time_switcher"] = time_switcher
    data["comparison_time_switcher"] = comparison_time_switcher

    timer.switch_to(time_switcher, "run_main > run > run_process (preamble)")

    io.start_logging(env, path=log_path, level=level)
    logging.info("mp.Process %d started (%d nodes)", num, len(ids))
    nnodes = data["params"]["num_nodes"]

    timer.switch_to(time_switcher, "run_main > run > for")

    for node in ids:

        q.put(SENTINEL)
        if pbar is not None:
            pbar.update()

        if data["params"]['sw_ponding_process'] == 'enabled':
            zone_sw = data["params"]['sw_zone_mapping'][node]
            pond_area = data["params"]['sw_ponding_area'][zone_sw]
        else:
            pond_area = 0.0

        rep_zone = data["params"]["reporting_zone_mapping"][node]
        if rep_zone != 0:
            timer.switch_to(time_switcher, "run_main > run > run_process (output calculation)")
            output = get_output(data, node, time_switcher)
            timer.switch_to(time_switcher, "run_main > run > run_process (post calc)")

            logging.debug("RAM usage is %.2fMb", u.get_ram_usage_for_process())
            if not test:
                aggregate_output(time_switcher, node, data, output, num, rep_zone, stuff, recharge_agg, nnodes, recharge, runoff, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, spatial_index, pond_area)

    logging.info("mp.Process %d ended", num)

    timer.switch_off(time_switcher)
    timer.switch_off(timer_token_run_process)

    timer.print_time_switcher_report(time_switcher)
    timer.print_time_switcher_report(comparison_time_switcher)
    timer.print_time_switcher_report(timer_token_run_process)

def compare_lambdas(name, time_switcher, unoptimised, optimised):

    timer.switch_to(time_switcher, f"{name} (unoptimised)")
    unoptimised_result = unoptimised()

    timer.switch_to(time_switcher, f"{name} (optimised)")
    optimised_result = optimised()

    timer.switch_to(time_switcher, f"{name} (comparison)")
    np.testing.assert_equal(unoptimised_result, optimised_result)

    return optimised_result

def aggregate_output(time_switcher, node, data, output, num, rep_zone, stuff, recharge_agg, nnodes, recharge, runoff, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, spatial_index, pond_area):
    if node in data["params"]["output_individual"]:
        timer.switch_to(time_switcher, "aggregate_output (output_individual)")
        # if this node for individual output then preserve
        stuff.single_node_output[node] = output.copy()
    timer.switch_to(time_switcher, "aggregate_output > (call m.aggregate)")
    key = (num, rep_zone)
    area = data["params"]["node_areas"][node]
    if key not in stuff.reporting_agg:
        stuff.reporting_agg[key] = m.aggregate(output, area, pond_area)
    else:
        stuff.reporting_agg[key] = m.aggregate(
            output, area, pond_area, reporting=stuff.reporting_agg[key])

    timer.switch_to(time_switcher, "aggregate_output > (output_recharge)")
    if data["params"]["output_recharge"]:
        rech = {"recharge": output["combined_recharge"].copy()}
        for i, p in enumerate(
                u.aggregate_output_col(data,
                                        rech,
                                        "recharge",
                                        method="average")):
            recharge_agg[(nnodes * i) + int(node)] = p
        rech = None

    timer.switch_to(time_switcher, "aggregate_output > (swrecharge_process)")
    if data["params"]["swrecharge_process"] == "enabled":

        rech = output["combined_recharge"].copy()
        for i, p in enumerate(rech):
            recharge[(nnodes * i) + int(node)] = p
        rech = None

        ro = output["combined_str"].copy()
        for i, p in enumerate(ro):
            runoff[(nnodes * i) + int(node)] = p
        ro = None

    timer.switch_to(time_switcher, "aggregate_output > (output_sfr)")
    if (data["params"]["output_sfr"]
            or data["params"]["excess_sw_process"] != "disabled"):
        ro = {"runoff": output["combined_str"].copy()}
        for i, p in enumerate(
                u.aggregate_output_col(data,
                                        ro,
                                        "runoff",
                                        method="average")):
            runoff_agg[(nnodes * i) + int(node)] = p
        ro = None

    timer.switch_to(time_switcher, "aggregate_output > (output_evt)")
    if data["params"]["output_evt"]:
        evt = {"evtr": output["unutilised_pe"].copy()}
        for i, p in enumerate(
                u.aggregate_output_col(data,
                                        evt,
                                        "evtr",
                                        method="average")):
            evtr_agg[(nnodes * i) + int(node)] = p
        evt = None

    timer.switch_to(time_switcher, "aggregate_output > (solute)")
    if data["params"]["solute_process"] == "enabled":
        solute.aggregate_solute(solute_aggregation, data, output, node)
        solute.aggregate_surface_water_solute(stream_solute_aggregation, data, output, node)
        solute.aggregate_mi(solute_mi_aggregation, data, output, node)

    timer.switch_to(time_switcher, "aggregate_output > (spatial_output_date)")
    if data["params"]["spatial_output_date"]:
        stuff.spatial[node] = m.aggregate(output,
                                    area,
                                    pond_area,
                                    index=spatial_index)
    logging.info("mp.Process %d ended", num)

# this stuff stranded here for windows - multiprocessing cannot handle
#  - functions not in the top level
#  - not pickleable objects as arguments to that function :(
def listener(q, total):
    pbar = tqdm(total=total, desc="SWAcMod Parallel        ")
    for item in iter(q.get, None):
        pbar.update()
    pbar.close()

def run(test=False, debug=False, file_format=None, reduced=False, skip=False, env=Environment(), file_opener=DefaultFileResource._default_file_open):
    """Run model for all nodes."""
    total_timer_switcher_for_run = timer.make_time_switcher()
    timer.switch_to(total_timer_switcher_for_run, "run_main > run")
    timer_switcher_for_run = timer.make_time_switcher()
    timer.switch_to(timer_switcher_for_run, "run_main > run (before loading data)")
    times = {"start_of_run": time.time()}

    if ff.disable_multiprocessing:
        stuff = stuff_module.make_single_threaded_stuff()
    else:
        stuff = stuff_module.make_multiprocessing_stuff()

    specs_file = u.CONSTANTS["SPECS_FILE"]
    input_file = u.CONSTANTS["INPUT_FILE"]
    input_dir = u.CONSTANTS["INPUT_DIR"]

    env.print(version_information.format_version_information())

    timer.switch_to(timer_switcher_for_run, "run_main > run (loading data)")
    level, log_path = scrape_run_name_and_start_logging(debug, env, input_file, file_opener)

    data = input_file_reader.read_inputs(specs_file, input_file, input_dir, file_opener)
    params = data["params"]

    check_open_files(file_format, skip, data)

    timer.switch_to(timer_switcher_for_run, "run_main > run (getting ready for multiprocessing)")

    per = len(data["params"]["time_periods"])
    nnodes = data["params"]["num_nodes"]
    len_rch_agg = (nnodes * per) + 1
    if not ff.disable_multiprocessing:
        recharge_agg = mp.Array("f", 1)
        runoff_agg = mp.Array("f", 1)
        runoff_recharge_agg = np.zeros((1))
        evtr_agg = mp.Array("f", 1)
        solute_aggregation = solute.make_aggregation_array(data)
        stream_solute_aggregation = solute.make_aggregation_array(data)
        solute_mi_aggregation = solute.make_mi_aggregation_array(data)
        recharge = mp.Array("f", 1)
        runoff = mp.Array("f", 1)
        if params["swrecharge_process"] == "enabled" or data["params"][
                "output_recharge"]:
            recharge_agg = mp.Array("f",
                                    len_rch_agg)  # recharge by output period (agg)

        if params["swrecharge_process"] == "enabled" or data["params"][
                "output_sfr"]:
            runoff_agg = mp.Array("f", len_rch_agg)

        if params["swrecharge_process"] == "enabled":
            runoff_recharge_agg = np.zeros((len_rch_agg))

        if data["params"]["output_evt"]:
            evtr_agg = mp.Array("f", len_rch_agg)

        days = len(data["series"]["date"])
        len_rch = (nnodes * days) + 1

        if params["swrecharge_process"] == "enabled":
            recharge = mp.sharedctypes.Array("f", len_rch, lock=True)
            runoff = mp.sharedctypes.Array("f", len_rch, lock=True)
    else:
        recharge_agg = np.zeros(1, dtype=np.single)
        runoff_agg = np.zeros(1, dtype=np.single)
        runoff_recharge_agg = np.zeros((1))
        evtr_agg = np.zeros(1, dtype=np.single)
        solute_aggregation = solute.make_aggregation_array(data)
        stream_solute_aggregation = solute.make_aggregation_array(data)
        solute_mi_aggregation = solute.make_mi_aggregation_array(data)
        recharge = np.zeros(1, dtype=np.single)
        runoff = np.zeros(1, dtype=np.single)
        if params["swrecharge_process"] == "enabled" or data["params"][
                "output_recharge"]:
            recharge_agg = np.zeros(len_rch_agg, np.single)  # recharge by output period (agg)

        if params["swrecharge_process"] == "enabled" or data["params"][
                "output_sfr"]:
            runoff_agg = np.zeros(len_rch_agg, dtype=np.single)

        if params["swrecharge_process"] == "enabled":
            runoff_recharge_agg = np.zeros((len_rch_agg))

        if data["params"]["output_evt"]:
            evtr_agg = np.zeros(len_rch_agg, dtype=np.single)

        days = len(data["series"]["date"])
        len_rch = (nnodes * days) + 1

        if params["swrecharge_process"] == "enabled":
            recharge = np.zeros(len_rch, dtype=np.single)
            runoff = np.zeros(len_rch, dtype=np.single)

    chunks = extract_random_chunks_of_ids(data, nnodes)
    times["end_of_input"] = time.time()
    spatial_index = make_spatial_index(data, days)

    if ff.disable_multiprocessing:
        run_single_threaded(test, env, timer_switcher_for_run, stuff, level, log_path, data, nnodes, recharge_agg, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, recharge, runoff, chunks, spatial_index)
    else:
        run_multiprocessing(test, env, timer_switcher_for_run, stuff, level, log_path, data, nnodes, recharge_agg, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, recharge, runoff, chunks, spatial_index)

    timer.switch_to(timer_switcher_for_run, "run_main > run (output)")
    output_timer_token = timer.make_time_switcher()
    timer.switch_to(output_timer_token, "before anything interesting")

    times["end_of_model"] = time.time()

    if not test:

        # aggregate over processes
        timer.switch_to(output_timer_token, "reporting_agg")
        stuff.reporting_agg = aggregate_reporting(stuff.reporting_agg)
        timer.switch_to(output_timer_token, "swrecharge_process")
        runoff_recharge_module.calculate_runoff_recharge(data, days, nnodes, params, recharge, recharge_agg, runoff, runoff_agg, runoff_recharge_agg, stuff)

        env.print("\nWriting output files:")
        timer.switch_to(output_timer_token, "checking open files")
        check_open_files(file_format, skip, data)

        timer.switch_to(output_timer_token, "reporting agg loop")
        output_functions.output_water_balance(file_format, reduced, env, stuff.reporting_agg, data)

        timer.switch_to(output_timer_token, "output_individual")
        output_functions.output_individual(file_format, reduced, env, stuff.single_node_output, data)

        timer.switch_to(output_timer_token, "swrecharge_process")
        if params["swrecharge_process"] == "enabled":
            del runoff, recharge
            gc.collect()
        timer.switch_to(output_timer_token, "output_recharge")
        output_functions.output_recharge(env, data, recharge_agg)

        timer.switch_to(output_timer_token, "spatial_output_date")
        output_functions.output_spatial_output_date(reduced, env, stuff.spatial, data)

        timer.switch_to(output_timer_token, "output_sfr")
        roff_agg = output_functions.output_sfr(env, data, runoff_agg)

        timer.switch_to(output_timer_token, "output_evt")
        output_functions.output_evt(env, data, runoff_agg, evtr_agg, output_timer_token)

        timer.switch_off(output_timer_token)
        timer.print_time_switcher_report(output_timer_token)

        timer.switch_to(output_timer_token, "output_solute")
        output_functions.output_solute(data, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, roff_agg)

        timer.switch_off(output_timer_token)
        timer.print_time_switcher_report(output_timer_token)

    times["end_of_run"] = time.time()

    diff = times["end_of_run"] - times["start_of_run"]
    total = io.format_time(diff)

    per_node = int(round(diff * 1000 / data["params"]["num_nodes"]))
    cores = ("%d cores" % data["params"]["num_cores"]
             if data["params"]["num_cores"] != 1 else "1 core")

    env.print("\nPerformance (%s)" % cores)
    env.print("Input time:  %s" %
          io.format_time(times["end_of_input"] - times["start_of_run"]))
    env.print("Run time:    %s" %
          io.format_time(times["end_of_model"] - times["end_of_input"]))
    env.print("Output time: %s" %
          io.format_time(times["end_of_run"] - times["end_of_model"]))
    env.print("Total time:  %s (%d msec/node)" % (total, per_node))
    env.print("")

    logging.info("End SWAcMod run")

    gc.collect()

    timer.switch_off(timer_switcher_for_run)
    timer.switch_off(total_timer_switcher_for_run)
    timer.print_time_switcher_report(timer_switcher_for_run)
    timer.print_time_switcher_report(total_timer_switcher_for_run)

def extract_random_chunks_of_ids(data, nnodes):
    ids = range(1, nnodes + 1)
    random.shuffle(list(ids))
    chunks = np.array_split(ids, data["params"]["num_cores"])
    return chunks

def make_spatial_index(data, days):
    if data["params"]["spatial_output_date"] == "mean":
        spatial_index = [range(days)] + [u.month_indices(i+1, data)
                                         for i in range(12)]
    elif data["params"]["spatial_output_date"] is not None:
        spatial_index = [(data["params"]["spatial_output_date"] -
                         data["params"]["start_date"]).days]
    else:
        spatial_index = None
    return spatial_index

def run_single_threaded(test, env, timer_switcher_for_run, stuff, level, log_path, data, nnodes, recharge_agg, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, recharge, runoff, chunks, spatial_index):
    q = queue.Queue()
    pbar = tqdm(total=nnodes, desc="SWAcMod Parallel        ")

    timer.switch_to(timer_switcher_for_run, "run_main > run (multiprocessing)")

    for process, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue

        run_process(
                env,
                process,
                chunk,
                data,
                test,
                stuff,
                recharge_agg,
                runoff_agg,
                evtr_agg,
                solute_aggregation,
                stream_solute_aggregation,
                solute_mi_aggregation,
                recharge,
                runoff,
                log_path,
                level,
                spatial_index,
                q,
                pbar
            )

    q.put(None)
    pbar.close()

def run_multiprocessing(test, env, timer_switcher_for_run, stuff, level, log_path, data, nnodes, recharge_agg, runoff_agg, evtr_agg, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, recharge, runoff, chunks, spatial_index):
    workers = []
    q = mp.Queue()
    lproc = mp.Process(target=listener, args=(q, nnodes))
    lproc.start()

    for process, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue

        proc = mp.Process(
                target=run_process,
                args=(
                    env,
                    process,
                    chunk,
                    data,
                    test,
                    stuff,
                    recharge_agg,
                    runoff_agg,
                    evtr_agg,
                    solute_aggregation,
                    stream_solute_aggregation,
                    solute_mi_aggregation,
                    recharge,
                    runoff,
                    log_path,
                    level,
                    spatial_index,
                    q,
                ),
            )

        workers.append(Worker("worker%d" % process, q, proc, verbose=False))

    timer.switch_to(timer_switcher_for_run, "run_main > run (multiprocessing)")

    for p in workers:
        p.start()

    for p in workers:
        p.join()

    q.put(None)
    lproc.join()

def check_open_files(file_format, skip, data):
    if not skip:
        io.check_open_files(data, file_format, u.CONSTANTS["OUTPUT_DIR"])

def scrape_run_name_and_start_logging(debug, env, input_file, file_opener=DefaultFileResource._default_file_open):
    level = logging.DEBUG if debug else logging.INFO
    run_name = input_file_reader.scrape_run_name(input_file, file_opener)
    log_path = io.start_logging(env, level=level, run_name=run_name)

    env.print('\nStart "%s"' % run_name)
    logging.info("Start SWAcMod run")
    return level,log_path

def parse_arguments():
    # Parser for command line arguments
    DESCRIPTION = """
    Invoke this script to run SWAcMod.
    e.g. 'python swacmod_run.py'"""
    FORM = argparse.RawTextHelpFormatter

    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    PARSER.add_argument("-t",
                        "--test",
                        help="run with no output",
                        action="store_true")
    PARSER.add_argument("-d",
                        "--debug",
                        help="verbose log",
                        action="store_true")
    PARSER.add_argument("-r",
                        "--reduced",
                        help="reduced output",
                        action="store_true")
    PARSER.add_argument("-i",
                        "--input_yml",
                        help="path to input yaml file inside input directory")
    PARSER.add_argument("-o", 
                        "--output_dir", 
                        help="path to output directory - default: 'ouput_files/'")
    PARSER.add_argument("-f",
                        "--format",
                        help="output file format",
                        choices=["hdf5", "h5", "csv"],
                        default="csv")
    PARSER.add_argument("-s",
                        "--skip_prompt",
                        help="skip user prompts and warnings",
                        action="store_true")
    PARSER.add_argument("-v",
                        "--version",
                        action="version",
                        version=version_information.format_version_information())

    return PARSER.parse_args()

def check_arguments(ARGS):
    if not ARGS.input_yml:
        raise u.ArgumentError('No input file specified. Use "-i" or "--input_yml" to specify the path to "input.yml"\n')

def run_main():
    ARGS = parse_arguments()
    check_arguments(ARGS)

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

if __name__ == "__main__":
    mp.freeze_support()
    run_main()
