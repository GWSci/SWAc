import unittest
import swacmod.input_files.input_files_version_2.validation_new as validation_new
import swacmod.input_files.input_files_version_2.specs as specs_module
import datetime

class Test_Validation_New(unittest.TestCase):
    def test_validation_converts_exceptions_to_error_list(self):
        spec_maps = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = make_valid_params()
        params["run_name"] = 5
        actual = validation_new.validate_2(params, spec_maps).errors
        self.assertEqual(
            "---> Validation failed: Parameter \"run_name\" has to be a string, found a <class 'int'> instead",
            actual[0])

    def test_validations_skipped_for_alt_files(self):
        spec_maps = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = make_valid_params()
        params["time_periods"] = "some-alt-file.yml"
        actual = validation_new.validate_2(params, spec_maps).errors
        self.assertEqual(0, len(actual), msg = "\n".join(actual))

    def test_validations_pass_for_valid_input(self):
        spec_maps = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = make_valid_params()
        actual = validation_new.validate_2(params, spec_maps).errors
        self.assertEqual(0, len(actual), msg = "\n".join(actual))

    def test_invalid_values_record_errors(self):
        invalid_pairs = {
            "run_name": 5,
            "num_cores": "x",
            "num_cores": 1000,
            "num_nodes": "x",
            "num_nodes": -1,
            "node_areas": "x",
            "node_areas": {},
            "node_areas": {1: "x", 2: "x", 3: "x"},
            "node_areas": {1: -11, 2: -13, 3: -17},
            "start_date": "x",
            "time_periods": "x",
            "time_periods": [[], [], []],
            "time_periods": [["x", "x"], ["x", "x"], ["x", "x"]],
            "time_periods": [[1, 1], [1, 1], [1, 1]],
            "output_recharge": "x",
            "output_individual": "x",
            "irchcb": "x",
            "nodes_per_line": "x",
            "nodes_per_line": -5,
            "output_fac": "x",
            "output_fac": -5,
            "reporting_zone_names": "x",
            "reporting_zone_names": {1:2, 3:4},
            "rainfall_zone_names": "x",
            "rainfall_zone_names": {1:2, 3:4},
            "rapid_runoff_zone_names": "x",
            "rapid_runoff_zone_names": {1:2, 3:4},
            "pe_zone_names": "x",
            "pe_zone_names": {1:2, 3:4},
            "temperature_zone_names": "x",
            "temperature_zone_names": {1:2, 3:4},
            "tmax_c_zone_names": "x",
            "tmax_c_zone_names": {1:2, 3:4},
            "tmin_c_zone_names": "x",
            "tmin_c_zone_names": {1:2, 3:4},
            "windsp_zone_names": "x",
            "windsp_zone_names": {1:2, 3:4},
            "subroot_zone_names": "x",
            "subroot_zone_names": {1:2, 3:4},
            "rapid_runoff_zone_names": "x",
            "rapid_runoff_zone_names": {1:2, 3:4},
            "swrecharge_zone_names": "x",
            "swrecharge_zone_names": {1:2, 3:4},
            "macropore_zone_names": "x",
            "macropore_zone_names": {1:2, 3:4},
            "soil_zone_names": "x",
            "soil_zone_names": {1:2, 3:4},
            "landuse_zone_names": "x",
            "landuse_zone_names": {1:2, 3:4},
            "canopy_zone_names": "x",
            "canopy_zone_names": {1:2, 3:4},
            "interflow_zone_names": "x",
            "interflow_zone_names": {1:2, 3:4},
            # "sw_zone_names": "x", # TODO: Requires natproc flag.
            # "sw_zone_names": {1:2, 3:4}, # TODO required Natproc flag.
            "canopy_process": "x",
            "canopy_process": 5,
            "snow_process_simple": "x",
            "snow_process_simple": 5,
            "snow_process_complex": "x",
            "snow_process_complex": 5,
            "rapid_runoff_process": "x",
            "rapid_runoff_process": 5,
            "swrecharge_process": "x",
            "swrecharge_process": 5,
            # "single_cell_swrecharge_process": "x", # TODO: Legacy feature (no longer maintained)
            # "single_cell_swrecharge_process": 5, # TODO: Legacy feature (no longer maintained)
            "macropore_process": "x",
            "macropore_process": 5,
            "fao_process": "x",
            "fao_process": 5,
            "subroot_leakage_process": "x",
            "subroot_leakage_process": 5,
            "interflow_process": "x",
            "interflow_process": 5,
            "recharge_attenuation_process": "x",
            "recharge_attenuation_process": 5,
            "historical_solute_process": "x",
            "historical_solute_process": 5,
            "solute_process": "x",
            "solute_process": 5,
            "sw_process": "x",
            "sw_process": 5,
            "sw_ponding_process": "x",
            "sw_ponding_process": 5,
            "routing_process": "x",
            "routing_process": 5,
            "excess_sw_process": "x",
            "excess_sw_process": 5,
            "excess_sw_process": "enabled",
            "recharge_node_mapping": "x",
            "recharge_node_mapping": {"a": "b", "c": "d"},
            "reporting_zone_mapping": "x",
            "reporting_zone_mapping": {"a": "b", "c": "d"},
            "rainfall_zone_mapping": "x",
            "rainfall_zone_mapping": {"a": "b", "c": "d"},
            "pe_zone_mapping": "x",
            "pe_zone_mapping": {"a": "b", "c": "d"},
            "temperature_zone_mapping": "x",
            "temperature_zone_mapping": {"a": "b", "c": "d"},
            "tmax_c_zone_mapping": "x",
            "tmax_c_zone_mapping": {"a": "b", "c": "d"},
            "tmin_c_zone_mapping": "x",
            "tmin_c_zone_mapping": {"a": "b", "c": "d"},
            "windsp_zone_mapping": "x",
            "windsp_zone_mapping": {"a": "b", "c": "d"},
            "subroot_zone_mapping": "x",
            "subroot_zone_mapping": {"a": "b", "c": "d"},
            "rapid_runoff_zone_mapping": "x",
            "rapid_runoff_zone_mapping": {"a": "b", "c": "d"},
            "swrecharge_zone_mapping": "x",
            "swrecharge_zone_mapping": {"a": "b", "c": "d"},
            "macropore_zone_mapping": "x",
            "macropore_zone_mapping": {"a": "b", "c": "d"},
            "canopy_zone_mapping": "x",
            "canopy_zone_mapping": {"a": "b", "c": "d"},
            "interflow_zone_mapping": "x",
            "interflow_zone_mapping": {"a": "b", "c": "d"},
            "sw_zone_mapping": "x",
            "sw_zone_mapping": {"a": "b", "c": "d"},
            "gwmodel_type": 5,
            "solute_calibration_a": "x",
            "solute_calibration_sigma": "x",
            "solute_calibration_alpha": "x",
            "sw_init_ponding": "x",
            "sw_max_ponding": "x",
            "output_sfr": "x",
            "attenuate_sfr_flows": "x",
            "istcb1": "x",
            "istcb2": "x",
            # "mf96_lrc": "x", # TODO no validation
            "output_evt": "x",
            "percolation_rejection_use_timeseries": "x",
            "infiltration_limit_use_timeseries": "x",
            "interflow_decay_use_timeseries": "x",
            "fao_input": 5,
            "fao_input": "x",
            "free_throughfall": "x",
            "free_throughfall": 5,
            "max_canopy_storage": "x",
            "max_canopy_storage": 5,
            "snow_params_simple": 5,
            "snow_params_complex": "x",
            "snow_params_simple": {"a":"b","c":"d","e":"f"},
            "snow_params_simple": {1:[],2:[],3:[]},
            "snow_params_complex": 5,
            "snow_params_complex": "x",
            "snow_params_simple": {"a":"b","c":"d","e":"f"},
            "snow_params_simple": {1:[],2:[],3:[]},
            "rapid_runoff_params": "x",
            "rapid_runoff_params": 5,
            "rapid_runoff_params": {"a":"b","c":"d","e":"f"},
            "rapid_runoff_params": {1:[],2:[],3:[]},
            "rapid_runoff_params": [{"a":"b","c":"d","e":"f"}],
            "swrecharge_proportion": 5,
            "swrecharge_proportion": "x",
            "swrecharge_proportion": [],
            "swrecharge_proportion": {"a":"b","c":"d","e":"f"},
            "swrecharge_proportion": {1:[],2:[],3:[]},
            "swrecharge_limit": 5,
            "swrecharge_limit": "x",
            "swrecharge_limit": [],
            "swrecharge_limit": {"a":"b","c":"d","e":"f"},
            "swrecharge_limit": {1:[],2:[],3:[]},
            "macropore_proportion": 5,
            "macropore_proportion": "x",
            "macropore_proportion": [],
            "macropore_proportion": {"a":"b","c":"d","e":"f"},
            "macropore_proportion": {1:[],2:[],3:[]},
            "macropore_limit": 5,
            "macropore_limit": "x",
            "macropore_limit": [],
            "macropore_limit": {"a":"b","c":"d","e":"f"},
            "macropore_limit": {1:[],2:[],3:[]},
            "macropore_activation": 5,
            "macropore_activation": "x",
            "macropore_activation": [],
            "macropore_activation": {"a":"b","c":"d","e":"f"},
            "macropore_activation": {1:[],2:[],3:[]},
            "macropore_recharge": 5,
            "macropore_recharge": "x",
            "macropore_recharge": [],
            "macropore_recharge": {"a":"b","c":"d","e":"f"},
            "macropore_recharge": {1:[],2:[],3:[]},            
            # "soil_static_params": 5, #TODO validation only runs if fao_process=enabled
            # "soil_static_params": "x", #TODO validation only runs if fao_process=enabled
            # "soil_static_params": [], #TODO validation only runs if fao_process=enabled
            # "soil_static_params": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "soil_static_params": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "smd": 5, #TODO validation only runs if fao_process=enabled
            # "smd": "x", #TODO validation only runs if fao_process=enabled
            # "smd": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "soil_spatial": 5, #TODO validation only runs if fao_process=enabled
            # "soil_spatial": "x", #TODO validation only runs if fao_process=enabled
            # "soil_spatial": [], #TODO validation only runs if fao_process=enabled
            # "soil_spatial": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "soil_spatial": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "lu_spatial": 5, #TODO validation only runs if fao_process=enabled
            # "lu_spatial": "x", #TODO validation only runs if fao_process=enabled
            # "lu_spatial": [], #TODO validation only runs if fao_process=enabled
            # "lu_spatial": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "lu_spatial": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "zr": 5, #TODO validation only runs if fao_process=enabled
            # "zr": "x", #TODO validation only runs if fao_process=enabled
            # "zr": [], #TODO validation only runs if fao_process=enabled
            # "zr": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "zr": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "kc": 5, #TODO validation only runs if fao_process=enabled
            # "kc": "x", #TODO validation only runs if fao_process=enabled
            # "kc": [], #TODO validation only runs if fao_process=enabled
            # "kc": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "kc": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "taw": 5, #TODO validation only runs if fao_process=enabled
            # "taw": "x", #TODO validation only runs if fao_process=enabled
            # "taw": [], #TODO validation only runs if fao_process=enabled
            # "taw": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "taw": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "raw": 5, #TODO validation only runs if fao_process=enabled
            # "raw": "x", #TODO validation only runs if fao_process=enabled
            # "raw": [], #TODO validation only runs if fao_process=enabled
            # "raw": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "raw": {1:[],2:[],3:[]}, #TODO validation only runs if fao_process=enabled
            # "percolation_rejetion": 5, #TODO validation only runs if fao_process=enabled
            # "percolation_rejetion": "x", #TODO validation only runs if fao_process=enabled
            # "percolation_rejetion": {"a":"b","c":"d","e":"f"}, #TODO validation only runs if fao_process=enabled
            # "percolation_rejection_ts": #TODO timeseries
            "subroot_leakage_fraction": 5,
            "subroot_leakage_fraction": "x",
            "init_interflow_store": 5,
            "init_interflow_store": "x",
            "interflow_store_bypass": 5,
            "interflow_store_bypass": "x",
            "infiltration_limit": 5,
            "infiltration_limit": "x",
            "interflow_decay": 5,
            "interflow_decay": "x",
            # "infiltration_limit_ts": #TODO timeseries
            # "interflow_decay_ts": #TODO timeseries
            "recharge_attenuation_params": 5,
            "recharge_attenuation_params": "x",
            "recharge_attenuation_params": [],
            "recharge_attenuation_params": {"a":"b","c":"d","e":"f"},
            "recharge_attenuation_params": {1:[],2:[],3:[]},
            "solute_calibration_mu": 5,
            "solute_calibration_mu": "x",
            "solute_calibration_effective_porosity": 5,
            "solute_calibration_effective_porosity": "x",
            "solute_depth_to_water": 5,
            "solute_depth_to_water": "x",
            "solute_loading": 5,
            "solute_loading": "x",
            "solute_loading": [],
            "solute_loading": {"a":"b","c":"d","e":"f"},
        }
        spec_maps = specs_module.make_specs_dictionary(specs_module.make_specs())
        for k, v in invalid_pairs.items():
            params = make_valid_params()
            params[k] = v
            actual = validation_new.validate_2(params, spec_maps).errors
            message = f"Expected an error for [{k} = {v}]."
            self.assertNotEqual(0, len(actual), msg = message)

# TODO A similar structure is in test_input_data_v2_validation.py. See if these can be merged.
def make_valid_params():
    return {
        "version": 2,
        "run_name": "my_run",
        "temp_file_backed_array_directory": "temp_scratch_files/",
        "num_nodes": 3,
        "node_areas": {1: 11, 2: 13, 3: 17},
        "node_xy": "node_xy.csv",
        "start_date": datetime.datetime(1980, 1, 1),
        "time_periods": [[1, 2], [3, 4], [5, 6]],
        "num_cores": 1,
        "output_recharge": True,
        "irchcb": 50,
        "nodes_per_line": 5,
        "output_fac": 1.0,
        "spatial_output_date": "1980-01-01",
        "reporting_zone_names": {1:"b", 2:"d"},
        "rainfall_zone_names": {1:"b", 2:"d"},
        "recharge_node_mapping": "rch_nodes.csv",
        "reporting_zone_mapping": {1:1, 2:1, 3:2},
        "rainfall_zone_mapping": {1:[1], 2:[1], 3:[2]},
        "pe_zone_mapping": {1:[1], 2:[1], 3:[2]},
        "temperature_zone_mapping": {1:1, 2:1, 3:2},
        "tmax_c_zone_mapping": {1:1, 2:1, 3:2},
        "tmin_c_zone_mapping": {1:1, 2:1, 3:2},
        "windsp_zone_mapping": {1:1, 2:1, 3:2},
        "subroot_zone_mapping": {1:[1], 2:[1], 3:[2]},
        "rapid_runoff_zone_mapping": {1:1, 2:1, 3:2},
        "swrecharge_zone_mapping": {1:1, 2:1, 3:2},
        "macropore_zone_mapping": {1:1, 2:1, 3:2},
        "canopy_zone_mapping": {1:1, 2:1, 3:2},
        "interflow_zone_mapping": {1:1, 2:1, 3:2},
        "sw_zone_mapping": {1:1, 2:1, 3:2},
        "pe_zone_names": {1:"b", 2:"d"},
        "temperature_zone_names": {1:"b", 2:"d"},
        "tmax_c_zone_names": {1:"b", 2:"d"},
        "tmin_c_zone_names": {1:"b", 2:"d"},
        "windsp_zone_names": {1:"b", 2:"d"},
        "subroot_zone_names": {1:"b", 2:"d"},
        "rapid_runoff_zone_names": {1:"b", 2:"d"},
        "swrecharge_zone_names": {1:"b", 2:"d"},
        "macropore_zone_names": {1:"b", 2:"d"},
        "soil_zone_names": {1:"b", 2:"d"},
        "landuse_zone_names": {1:"b", 2:"d"},
        "canopy_zone_names": {1:"b", 2:"d"},
        "interflow_zone_names": {1:"b", 2:"d"},
        "sw_zone_names": {1:"b", 2:"d"},
        "output_individual": set([1, 2, 3]),
        "canopy_process": "enabled",
        "snow_process_simple": "enabled",
        "snow_process_complex": "enabled",
        "rapid_runoff_process": "enabled",
        "swrecharge_process": "enabled",
        "single_cell_swrecharge_process": "enabled",
        "macropore_process": "enabled",
        "fao_process": "enabled",
        "subroot_leakage_process": "enabled",
        "interflow_process": "enabled",
        "recharge_attenuation_process": "enabled",
        "historical_solute_process": "enabled",
        "solute_process": "enabled",
        "sw_process": "enabled",
        "sw_ponding_process": "enabled",
        "routing_process": "enabled",
        "excess_sw_process": "sw_rip",
        "gwmodel_type": "mf6",
        "solute_calibration_a": 1.38,
        "solute_calibration_sigma": 3.96,
        "solute_calibration_alpha": 3906.25,
        "sw_init_ponding": 5.0,
        "sw_max_ponding": 300.0,
        "output_sfr": True,
        "attenuate_sfr_flows": False,
        "istcb1": 50,
        "istcb2": 55,
        "mf96_lrc": [2, 2, 5],
        "output_evt": True,
        "percolation_rejection_use_timeseries": True,
        "infiltration_limit_use_timeseries": True,
        "interflow_decay_use_timeseries": True,
        "fao_input": 'l',
        "free_throughfall": {1:0.99,2: 0.99},
        "max_canopy_storage": {1:1.0,2: 1.0},
        "snow_params_simple": {1:[100, 1, -2], 2:[100, 1, -2], 3:[100, 1, -2]},
        "snow_params_complex": {1:[7.55, 0.05, 4.79, 20.0, 100.0, 0.25, 0.95, 1.0, 0.0, 450.0],2:[7.55, 0.05, 4.79, 20.0, 100.0, 0.25, 0.95, 1.0, 0.0, 450.0],3:[7.55, 0.05, 4.79, 20.0, 100.0, 0.25, 0.95, 1.0, 0.0, 450.0]},
        "rapid_runoff_params": [{'class_smd':[5],'class_ri':[5],'values':[[0.16]]},{'class_smd':[5],'class_ri':[5],'values':[[0.16]]}],
        "swrecharge_proportion": {1:[0.1,0.],2:[0.1,0.],3:[0.1,0.],4:[0.1,0.],5:[0.1,0.],6:[0.1,0.],7:[0.1,0.],8:[0.1,0.],9:[0.1,0.],10:[0.1,0.],11:[0.1,0.],12:[0.1,0.]},
        "swrecharge_limit": {1:[0.,2.],2:[0.,2.],3:[0.,2.],4:[0.,2.],5:[0.,2.],6:[0.,2.],7:[0.,2.],8:[0.,2.],9:[0.,2.],10:[0.,2.],11:[0.,2.],12:[0.,2.]},
        "macropore_proportion": {1:[0.05,0.05],2:[0.05,0.05],3:[0.05,0.05],4:[0.05,0.05],5:[0.05,0.05],6:[0.05,0.05],7:[0.05,0.05],8:[0.05,0.05],9:[0.05,0.05],10:[0.05,0.05],11:[0.05,0.05],12:[0.05,0.05]},
        "macropore_limit": {1:[0.05,0.05],2:[0.05,0.05],3:[0.05,0.05],4:[0.05,0.05],5:[0.05,0.05],6:[0.05,0.05],7:[0.05,0.05],8:[0.05,0.05],9:[0.05,0.05],10:[0.05,0.05],11:[0.05,0.05],12:[0.05,0.05]},
        "macropore_activation": {1:[0.1,0.1],2:[0.1,0.1],3:[0.1,0.1],4:[0.1,0.1],5:[0.1,0.1],6:[0.1,0.1],7:[0.1,0.1],8:[0.1,0.1],9:[0.1,0.1],10:[0.1,0.1],11:[0.1,0.1],12:[0.1,0.1]},
        "macropore_recharge": {1:[0.4,0.4],2:[0.4,0.4],3:[0.4,0.4],4:[0.4,0.4],5:[0.4,0.4],6:[0.4,0.4],7:[0.4,0.4],8:[0.4,0.4],9:[0.4,0.4],10:[0.4,0.4],11:[0.4,0.4],12:[0.4,0.4]},
        "soil_static_params": {'FC':[0.36,0.36],'WP':[0.15,0.15],'p':[0.55,0.55]},
        "smd": {'starting_SMD':[100,100]},
        "soil_spatial": {1:[0.5, 0.5],2:[0.5, 0.5],3:[0.5, 0.5]},
        "lu_spatial": {1:[0.5, 0.5],2:[0.5, 0.5],3:[0.5, 0.5]},
        "zr": {1:[0.4,0.4],2:[0.4,0.4],3:[0.4,0.4],4:[0.4,0.4],5:[0.4,0.4],6:[0.4,0.4],7:[0.4,0.4],8:[0.4,0.4],9:[0.4,0.4],10:[0.4,0.4],11:[0.4,0.4],12:[0.4,0.4]},
        "kc": {1:[0.4,0.4],2:[0.4,0.4],3:[0.4,0.4],4:[0.4,0.4],5:[0.4,0.4],6:[0.4,0.4],7:[0.4,0.4],8:[0.4,0.4],9:[0.4,0.4],10:[0.4,0.4],11:[0.4,0.4],12:[0.4,0.4]},
        "taw": {1:[231,231],2:[231,231],3:[231,231],4:[231,231],5:[231,231],6:[231,231],7:[231,231],8:[231,231],9:[231,231],10:[231,231],11:[231,231],12:[231,231]},
        "raw": {1:[127.05,127.05],2:[127.05,127.05],3:[127.05,127.05],4:[127.05,127.05],5:[127.05,127.05],6:[127.05,127.05],7:[127.05,127.05],8:[127.05,127.05],9:[127.05,127.05],10:[127.05,127.05],11:[127.05,127.05],12:[127.05,127.05]},
        "percolation_rejection": {'percolation_rejection': [10.0, 10.0]},

        "percolation_rejection_ts": "percolation_rejection_ts.yml", #TODO timeseries
        
        "subroot_leakage_fraction": {1:0.5,2:0.5,3:0.5},
        "init_interflow_store": {1:10.0,2:10.0},
        "interflow_store_bypass": {1:0.05,2:0.05},
        "infiltration_limit": {1:0.7,2:0.7},
        "interflow_decay": {1:0.05,2:0.05},
        "infiltration_limit_ts": "infiltration_limit_ts.yml", #TODO timeseries
        "interflow_decay_ts": "interflow_decay_ts.yml", #TODO timeseries
        "recharge_attenuation_params": {1:[1.0,0.05,2.0],2:[1.0,0.05,2.0],3:[1.0,0.05,2.0]},
        "solute_calibration_mu": {1:1.58,2:1.58,3:1.58},
        "solute_calibration_effective_porosity": {1:0.15,2:0.15,3:0.15},
        "solute_depth_to_water": {1:10.0,2:10.0,3:10.0},
        "solute_loading": {1:[156132,200000,118000,361.4217178,11.75727306,146.4513318,351.6890755,18.07108589,180.7108589,343.350632,90.35542945],
                           2:[156132,200000,119000,361.4217178,11.75727306,146.4513318,351.6890755,18.07108589,180.7108589,343.350632,90.35542945],
                           3:[156132,200000,120000,361.4217178,11.75727306,146.4513318,351.6890755,18.07108589,180.7108589,343.350632,90.35542945]},

        "sw_params": "sw_params.yml", #TODO
        "sw_downstream": "sw_downstream.yml", #TODO
        "sw_activation": "sw_activation.yml", #TODO
        "sw_bed_infiltration": "sw_bed_infiltration.yml", #TODO
        "sw_direct_recharge": "sw_direct_recharge.yml", #TODO
        "sw_pe_to_open_water": "sw_pe_to_open_water.yml", #TODO
        "sw_ponding_area": "sw_ponding_area.yml", #TODO
        "routing_topology": "routing_parameters.csv", #TODO
        "sfr_obs": "gauges_sfr.obs", #TODO
        "swdis_ts": "swdis_ts.csv", #TODO timeseries
        "swdis_locs": "swdis_locs.yml", #TODO
        "swabs_ts": "swabs_ts.csv", #TODO
        "swabs_locs": "swabs_locs.csv", #TODO
        "rainfall_ts": "rainfall_ts.yml", #TODO timeseries
        "pe_ts": "pe_ts.yml", #TODO timeseries
        "temperature_ts": "temperature_ts.yml", #TODO timeseries
        "tmax_c_ts": "tmax_c_ts.yml", #TODO timeseries
        "tmin_c_ts": "tmin_c_ts.yml", #TODO timeseries
        "windsp_ts": "windsp_ts.yml", #TODO timeseries
        "subroot_leakage_ts": "subroot_leakage_ts.yml", #TODO timeseries
        "evt_parameters": "evt_params.csv", #TODO
    }
