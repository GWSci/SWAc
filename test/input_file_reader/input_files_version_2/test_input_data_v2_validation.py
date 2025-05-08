import unittest
import io
import swacmod.input_files.input_files_version_2.input_data as input_data
import swacmod.input_files.input_files_version_2.validator as validator
import swacmod.input_files.input_files_version_2.specs as specs_module
import swacmod.input_files.input_file_reader as input_file_reader

class Printer_Spy:
    def __init__(self):
        self.result = ""
    
    def do_print(self, x):
        self.result += x

class Test_Input_Data_v2_Validation_Through_Input_File_Reader(unittest.TestCase):
    def test_reading_input_data_with_an_invalid_file_throws_an_exception(self):
        input_file = "some_file.yml"
        input_dir = "some_dir/"
        params = make_sample_valid_input_file()
        del params["num_nodes"]
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = make_mock_file_opener({input_file: input_file_contents})
        printer = lambda x: None
        with self.assertRaisesRegex(Exception, "Run has exited with errors."):
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener = file_opener, printer=printer)

    def test_reading_input_data_with_an_invalid_file_records_the_error(self):
        input_file = "some_file.yml"
        input_dir = "some_dir/"
        params = make_sample_valid_input_file()
        del params["num_nodes"]
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = make_mock_file_opener({input_file: input_file_contents})
        printer_spy = Printer_Spy()
        try:
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener=file_opener, printer=printer_spy.do_print)
        except:
            pass
        self.assertEqual('Error: The file "some_file.yml" is missing the required field "num_nodes".', printer_spy.result)

def make_mock_file_opener(filenames_to_contents):
    return lambda filename: io.StringIO(filenames_to_contents[filename])

class Test_Input_Data_v2_Validation(unittest.TestCase):
    def test_a_valid_file_has_no_errors(self):
        specs = specs_module.make_specs()
        params = make_sample_valid_input_file()
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(0, len(validation_result.errors))

    def test_a_valid_file_has_no_warnings(self):
        specs = specs_module.make_specs()
        params = make_sample_valid_input_file()
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(0, len(validation_result.warnings))

    def test_validating_a_file_missing_a_required_field_reports_an_error(self):
        specs = specs_module.make_specs()
        params = make_sample_valid_input_file()
        del params["version"]
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(1, len(validation_result.errors))

    def test_validating_a_file_missing_a_required_field_has_text_describing_the_problem(self):
        specs = specs_module.make_specs()
        params = make_sample_valid_input_file()
        del params["version"]
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        expected = 'Error: The file "some_input_file.yml" is missing the required field "version".'
        actual = validation_result.errors[0]
        self.assertEqual(expected, actual)

    def test_validating_empty_params_reports_all_missing_required_parameters(self):
        specs = specs_module.make_specs()
        params = {}
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        all_errors_string = "\n".join(validation_result.errors)
        self.assertIn('"version"', all_errors_string)
        self.assertIn('"run_name"', all_errors_string)

def make_sample_valid_input_file():
    return {
        "version": 2,
        "run_name": "my_run",
        "temp_file_backed_array_directory": "temp_scratch_files/",
        "num_nodes": 10,
        "node_areas": "node_areas.yml",
        "node_xy": "node_xy.csv",
        "start_date": "1980-01-01",
        "time_periods": "time_periods.csv",
        "num_cores": 1,
        "output_recharge": True,
        "irchcb": 50,
        "nodes_per_line": 5,
        "recharge_node_mapping": "rch_nodes.csv",
        "output_fac": 1.0,
        "spatial_output_date": "1980-01-01",
        "reporting_zone_mapping": "reporting_zone_mapping.yml",
        "reporting_zone_names": "reporting_zone_names.yml",
        "rainfall_zone_mapping": "rainfall_zone_mapping.yml",
        "rainfall_zone_names": "rainfall_zone_names.yml",
        "pe_zone_mapping": "pe_zone_mapping.yml",
        "pe_zone_names": "pe_zone_names.yml",
        "temperature_zone_mapping": "temperature_zone_mapping.yml",
        "temperature_zone_names": "temperature_zone_names.yml",
        "tmax_c_zone_mapping": "tmax_c_zone_mapping.yml",
        "tmax_c_zone_names": "tmax_c_zone_names.yml",
        "tmin_c_zone_mapping": "tmin_c_zone_mapping.yml",
        "tmin_c_zone_names": "tmin_c_zone_names.yml",
        "windsp_zone_mapping": "windsp_zone_mapping.yml",
        "windsp_zone_names": "windsp_zone_names.yml",
        "subroot_zone_mapping": "subroot_zone_mapping.yml",
        "subroot_zone_names": "subroot_zone_names.yml",
        "rapid_runoff_zone_mapping": "rapid_runoff_zone_mapping.yml",
        "rapid_runoff_zone_names": "rapid_runoff_zone_names.yml",
        "swrecharge_zone_mapping": "swrecharge_zone_mapping.yml",
        "swrecharge_zone_names": "swrecharge_zone_names.yml",
        "macropore_zone_mapping": "macropore_zone_mapping.yml",
        "macropore_zone_names": "macropore_zone_names.yml",
        "soil_zone_names": "soil_zone_names.yml",
        "landuse_zone_names": "landuse_zone_names.yml",
        "canopy_process": "enabled",
        "canopy_zone_mapping": "canopy_zone_mapping.yml",
        "canopy_zone_names": "canopy_zone_names.yml",
        "free_throughfall": "free_throughfall.yml",
        "max_canopy_storage": "max_canopy_storage.yml",
        "snow_process_simple": "enabled",
        "snow_params_simple": "snow_params.yml",
        "snow_process_complex": "disabled",
        "snow_params_complex": "snow_params_complex.yml",
        "rapid_runoff_process": "enabled",
        "rapid_runoff_params": "rapid_runoff_params.yml",
        "swrecharge_process": "enabled",
        "swrecharge_proportion": "swrecharge_proportion.yml",
        "swrecharge_limit": "swrecharge_limit.yml",
        "single_cell_swrecharge_process": "disabled",
        "macropore_process": "enabled",
        "macropore_proportion": "macropore_proportion.yml",
        "macropore_limit": "macropore_limit.yml",
        "macropore_activation": "macropore_activation.yml",
        "macropore_recharge": "macropore_recharge.yml",
        "fao_process": "enabled",
        "fao_input": 'l',
        "soil_static_params": "soil_static_params.yml",
        "smd": "smd.yml",
        "soil_spatial": "soil_spatial.yml",
        "lu_spatial": "lu_spatial.yml",
        "zr": "zr.yml",
        "kc": "kc.yml",
        "taw": "taw.yml",
        "raw": "raw.yml",
        "percolation_rejection": "percolation_rejection.yml",
        "percolation_rejection_use_timeseries": "true",
        "percolation_rejection_ts": "percolation_rejection_ts.yml",
        "subroot_leakage_process": "enabled",
        "subroot_leakage_fraction": "subsoilzone_leakage_fraction.yml",
        "interflow_process": "enabled",
        "interflow_zone_mapping": "interflow_zone_mapping.yml",
        "interflow_zone_names": "interflow_zone_names.yml",
        "init_interflow_store": "init_interflow_store.yml",
        "interflow_store_bypass": "interflow_store_bypass.yml",
        "infiltration_limit": "infiltration_limit.yml",
        "interflow_decay": "interflow_decay.yml",
        "infiltration_limit_use_timeseries": "true",
        "interflow_decay_use_timeseries": "true",
        "infiltration_limit_ts": "infiltration_limit_ts.yml",
        "interflow_decay_ts": "interflow_decay_ts.yml",
        "recharge_attenuation_process": "enabled",
        "recharge_attenuation_params": "recharge_attenuation_params.yml",
        "historical_solute_process": "disabled",
        "solute_process": "disabled",
        "solute_calibration_a": 1.38,
        "solute_calibration_mu": "mu.csv",
        "nitrate_calibration_sigma": 3.96,
        "nitrate_calibration_alpha": 3906.25,
        "nitrate_calibration_effective_porosity": "effective_porosity.csv",
        "nitrate_depth_to_water": "Average_DTW.csv",
        "nitrate_loading": "NO3_loading.csv",
        "sw_process": "enabled",
        "sw_params": "sw_params.yml",
        "sw_ponding_process": "disabled",
        "sw_zone_mapping": "sw_zone_mapping.yml",
        "sw_zone_names": "sw_zone_names.yml",
        "sw_downstream": "sw_downstream.yml",
        "sw_activation": "sw_activation.yml",
        "sw_bed_infiltration": "sw_bed_infiltration.yml",
        "sw_direct_recharge": "sw_direct_recharge.yml",
        "sw_pe_to_open_water": "sw_pe_to_open_water.yml",
        "sw_init_ponding": 5.0,
        "sw_max_ponding": 300.0,
        "sw_ponding_area": "sw_ponding_area.yml",
        "routing_process": "enabled",
        "routing_topology": "routing_parameters.csv",
        "output_sfr": True,
        "attenuate_sfr_flows": False,
        "sfr_obs": "gauges_sfr.obs",
        "istcb1": 50,
        "istcb2": 55,
        "swdis_ts": "swdis_ts.csv",
        "swdis_locs": "swdis_locs.yml",
        "swabs_ts": "swabs_ts.csv",
        "swabs_locs": "swabs_locs.csv",
        "rainfall_ts": "rainfall_ts.yml",
        "pe_ts": "pe_ts.yml",
        "temperature_ts": "temperature_ts.yml",
        "tmax_c_ts": "tmax_c_ts.yml",
        "tmin_c_ts": "tmin_c_ts.yml",
        "windsp_ts": "windsp_ts.yml",
        "subroot_leakage_ts": "subroot_leakage_ts.yml",
        "gwmodel_type": "mf6",
        "mf96_lrc": [2, 2, 5],
        "output_evt": True,
        "excess_sw_process": "disabled",
        "evt_parameters": "evt_params.csv",
    }
