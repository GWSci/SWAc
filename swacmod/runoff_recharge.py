#!/usr/bin/env python
"""SWAcMod main."""

# -*- coding: utf-8 -*-
from __future__ import print_function

import swacmod.feature_flags as ff

# Standard Library
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

import swacmod.performance_logging as performance_logging

import swacmod.timer as timer

import mmap
import gc

# Third Party Libraries
import numpy as np
from tqdm import tqdm

# Internal modules
from swacmod import utils as u
from swacmod import input_output as io
import swacmod.input_files.input_file_reader as input_file_reader
import swacmod.flopy_adaptor as flopy_adaptor
import swacmod.version_information as version_information
from swacmod.input_files.input_files_version_2.default_file_resource import DefaultFileResource
import swacmod.stuff_to_be_named_later as stuff_module
import swacmod.output_functions as output_functions

# Compile and import model
from swacmod import model as m
import swacmod.model_numpy as model_numpy

import swacmod.historical_solute as historical_solute
import swacmod.solute as solute
import swacmod.solute_proportion_reaching_water_table as solute_proportion
from swacmod.environment import Environment

def calculate_runoff_recharge(data, days, nnodes, params, recharge, recharge_agg, runoff, runoff_agg, runoff_recharge_agg, stuff):
    if params["swrecharge_process"] == "enabled":

        for cat in data["params"]["reporting_zone_mapping"].values():
            stuff.reporting_agg2[cat] = {}

        # ended up needing this for catchment output - bit silly
        if ff.disable_multiprocessing:
            if ff.use_natproc:
                runoff_recharge = recharge.copy()
            else:
                runoff_recharge = runoff.copy()
        else:
            if ff.use_natproc:
                runoff_recharge = np.frombuffer(recharge.get_obj(),
                                                dtype=np.float32).copy()
            else:
                runoff_recharge = np.frombuffer(runoff.get_obj(),
                                                dtype=np.float32).copy()

        # do RoR
        runoff, recharge = m.do_swrecharge_mask(data, runoff, recharge)
        # get RoR for cat output purposes
        if ff.use_natproc:
            if ff.disable_multiprocessing:
                runoff_recharge = recharge
            else:
                runoff_recharge = np.frombuffer(recharge.get_obj(),
                                                dtype=np.float32) - runoff_recharge
        else:
            if ff.disable_multiprocessing:
                runoff_recharge -= runoff
            else:
                runoff_recharge -= np.frombuffer(runoff.get_obj(),
                                                 dtype=np.float32)
        # aggregate amended recharge & runoff arrays by output periods
        for node in tqdm(list(m.all_days_mask(data).nodes()),
                         desc="Aggregating Fluxes      "):

            # get indices of output for this node
            idx = range(node, (nnodes * days) + 1, nnodes)

            if params['sw_ponding_process'] == 'enabled':
                zone_sw = data['params']['sw_zone_mapping'][node]
                pond_area = data['params']['sw_ponding_area'][zone_sw]
            else:
                pond_area = 0.0

            if ff.disable_multiprocessing:
                tmp = recharge
            else:
                tmp = np.frombuffer(recharge.get_obj(), dtype=np.float32)
            rch_array = np.array(tmp[idx], dtype=np.float64, copy=True)
            if ff.disable_multiprocessing:
                tmp = runoff
            else:
                tmp = np.frombuffer(runoff.get_obj(), dtype=np.float32)
            ro_array = np.array(tmp[idx], dtype=np.float64, copy=True)
            ror_array = np.array(runoff_recharge[idx],
                                 dtype=np.float64,
                                 copy=True)

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
            if ff.use_natproc:
                do_this_bit = rep_zone > 0
            else:
                do_this_bit = True
            if do_this_bit:
                area = data["params"]["node_areas"][node]
                ror = {"runoff_recharge": ror_array}

                if "runoff_recharge" not in stuff.reporting_agg2[rep_zone]:
                    stuff.reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
                        ror, area, pond_area)
                else:
                    stuff.reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
                        ror,
                        area,
                        pond_area,
                        reporting=stuff.reporting_agg2[rep_zone]["runoff_recharge"])

            # check for single node
            if node in data["params"]["output_individual"]:
                # amend single_node_output with ror values
                # this method required due to upstream bug
                tmp_node = stuff.single_node_output[node]
                tmp_node["runoff_recharge"] = ror_array.copy()
                tmp_node["combined_recharge"] = np.copy(rch_array)
                tmp_node["combined_str"] = np.copy(ro_array)
                stuff.single_node_output[node] = tmp_node

        # copy new bits into cat output
        term = "runoff_recharge"
        for cat in stuff.reporting_agg2:
            if "runoff_recharge" in stuff.reporting_agg2[cat]:
                stuff.reporting_agg[cat]["combined_recharge"] += stuff.reporting_agg2[
                    cat][term][term]
                stuff.reporting_agg[cat]["combined_str"] -= stuff.reporting_agg2[cat][
                    term][term]
                stuff.reporting_agg[cat]["runoff_recharge"] = stuff.reporting_agg2[
                    cat][term][term]
