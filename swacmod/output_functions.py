import swacmod.timer as timer
import gc
import numpy as np
from swacmod import utils as u
from swacmod import input_output as io
import swacmod.flopy_adaptor as flopy_adaptor
from swacmod import model as m
import swacmod.solute as solute

def output_water_balance(file_format, reduced, env, reporting_agg, data):
    for num, key in enumerate(reporting_agg.keys()):
        env.print("\t- Report file (%d of %d)" %
                (num + 1, len(reporting_agg.keys())))
        io.dump_water_balance(
                data,
                reporting_agg[key],
                file_format,
                u.CONSTANTS["OUTPUT_DIR"],
                zone=key,
                reduced=reduced,
            )

def output_individual(file_format, reduced, env, single_node_output, data):
    for node in list(data["params"]["output_individual"]):
        env.print("\t- Node output file")
        io.dump_water_balance(
                data,
                single_node_output[node],
                file_format,
                u.CONSTANTS["OUTPUT_DIR"],
                node=node,
                reduced=reduced,
            )

def output_recharge(env, data, recharge_agg):
    if data["params"]["output_recharge"]:
        env.print("\t- Recharge file")
        if data['params']['gwmodel_type'] == 'mfusg':
            io.dump_recharge_file(data, recharge_agg)
        elif data['params']['gwmodel_type'] == 'mf6':
            flopy_adaptor.write_mf_gwf_rch(m.get_mf6rch_file(data, recharge_agg))
        elif data['params']['gwmodel_type'] == 'mf96':
            io.dump_mf96_recharge_file(data, recharge_agg)

def output_spatial_output_date(reduced, env, spatial, data):
    if data["params"]["spatial_output_date"]:
        env.print("\t- Spatial file")
        io.dump_spatial_output(env,
                                   data,
                                   spatial,
                                   u.CONSTANTS["OUTPUT_DIR"],
                                   reduced=reduced)

def output_sfr(env, data, runoff_agg):
    roff_agg = None
    if data["params"]["output_sfr"]:
        env.print("\t- SFR file")
        if data['params']['gwmodel_type'] == 'mf96':
            roff_agg = np.copy(np.array(runoff_agg))
            strm = m.get_str_file(data, np.copy(np.array(runoff_agg)))
            strm.write_file()
                # remove header from str file
            with open(strm.file_name[0], 'r') as fin:
                    #data = fin.read().splitlines(True)
                lst_strm = fin.readlines()
            with open(strm.file_name[0], 'w') as fout:
                fout.write(lst_strm[1].rstrip() + "        -1\n")
                fout.writelines(lst_strm[2:])
            del strm
        else:
            sfr = m.get_sfr_file(data, np.copy(np.array(runoff_agg)))
            if data['params']['gwmodel_type'] == 'mfusg':
                io.dump_sfr_output(sfr)
            elif data['params']['gwmodel_type'] == 'mf6':
                sfr.write()
            del sfr
            gc.collect()
    return roff_agg

def output_evt(env, data, runoff_agg, evtr_agg, output_timer_token):
    if data["params"]["output_evt"]:
        env.print("\t- EVT file")

        if data["params"]["excess_sw_process"] != "disabled":
            timer.switch_to(output_timer_token, "output_evt (copying arrays)")
            tmp = (np.copy(np.array(evtr_agg)) -
                       np.copy(np.array(runoff_agg)))
            if data["params"]["excess_sw_process"] == "sw_rip":
                timer.switch_to(output_timer_token, "output_evt (sw_rip)")
                evt = m.get_evt_file(data, tmp)
            elif data["params"]["excess_sw_process"] == "sw_ow_evap":
                timer.switch_to(output_timer_token, "output_evt (sw_ow_evap)")
                evt = m.get_evt_file(data, np.where(tmp > 0.0, 0.0, tmp))
            elif data["params"]["excess_sw_process"] == "sw_only":
                timer.switch_to(output_timer_token, "output_evt (sw_only)")
                evt = m.get_evt_file(data, -np.copy(np.array(runoff_agg)))
            else:
                raise Exception("Could not determine evt.")
        else:
            timer.switch_to(output_timer_token, "output_evt (else)")
            evt = m.get_evt_file(data, evtr_agg)

        if data['params']['gwmodel_type'] == 'mfusg':
            timer.switch_to(output_timer_token, "output_evt (mfusg)")
            io.dump_evt_output(evt)
        elif data['params']['gwmodel_type'] == 'mf6':
            timer.switch_to(output_timer_token, "output_evt (mf6)")
            evt.write()
        timer.switch_to(output_timer_token, "output_evt (cleaning up)")
        evt, tmp = None, None
        del evt, tmp
        gc.collect()

def output_solute(data, solute_aggregation, stream_solute_aggregation, solute_mi_aggregation, roff_agg):
    if data["params"]["solute_process"] == "enabled":
        if data['params']['gwmodel_type'] == 'mf96':
            stream_conc = m.get_str_solute(data, roff_agg, stream_solute_aggregation)
            solute.write_stream_solute_csv(data, stream_conc)
        solute.write_solute_csv(data, solute_aggregation)
        solute.write_mi_csv(data, solute_mi_aggregation)
