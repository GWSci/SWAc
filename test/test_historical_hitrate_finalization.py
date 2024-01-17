import datetime
import unittest
import swacmod.finalization as finalization
import swacmod.utils as utils

class Test_Historical_Nitrate_Finalization(unittest.TestCase):
	def test_finalize_minimal_data_completes_without_error(self):
		data = make_minimal_data()
		finalize_params_and_series(data)
		# This assertion-free test is important to ensure the data
		# is in a fit state for the other tests. It will fail when
		# a value required for any finalisation is omitted from
		# the test dataset, for example when a finalisation is
		# added for a new input parameter.

	def test_finalize_param_converts_start_date_when_valid(self):
		data = make_minimal_data()
		data["params"]["start_date"] = "2024-01-16"

		finalize_params_and_series(data)

		actual = data["params"]["start_date"]
		expected = datetime.datetime(2024, 1, 16)
		self.assertEqual(expected, actual)

	def test_finalize_param_raises_Validation_Error_when_start_date_is_invalid(self):
		data = make_minimal_data()
		data["params"]["start_date"] = "aardvark"

		with self.assertRaises(utils.FinalizationError):
			finalize_params_and_series(data)

	def test_finalize_param_historical_start_date_is_ignored_when_historical_nitrate_process_is_disabled(self):
		data = make_minimal_data()
		data["params"]["historical_nitrate_process"] = "disabled"
		data["params"]["historical_start_date"] = "aardvark"

		finalize_params_and_series(data)

		actual = data["params"]["historical_start_date"]
		expected = "aardvark"
		self.assertEqual(expected, actual)

	def test_finalize_param_converts_historical_start_date_when_valid(self):
		data = make_minimal_data()
		data["params"]["historical_nitrate_process"] = "enabled"
		data["params"]["historical_start_date"] = "2024-01-16"

		finalize_params_and_series(data)

		actual = data["params"]["historical_start_date"]
		expected = datetime.datetime(2024, 1, 16)
		self.assertEqual(expected, actual)

	def test_finalize_param_raises_Validation_Error_when_historical_start_date_is_invalid(self):
		data = make_minimal_data()
		data["params"]["historical_nitrate_process"] = "enabled"
		data["params"]["historical_start_date"] = "aardvark"

		with self.assertRaises(utils.FinalizationError):
			finalize_params_and_series(data)

def finalize_params_and_series(data):
	finalization.finalize_params(data)

def make_minimal_data():
	return {
		"params" : {
			"canopy_zone_mapping" : None,
			"canopy_zone_names" : None,
			"disv" : None,
			"evt_parameters" : None,
			"excess_sw_process" : None,
			"fao_input" : "l",
			"fao_process" : "disabled",
			"free_throughfall" : None,
			"gwmodel_type" : None,
			"historical_nitrate_process" : None,
			"historical_start_date" : None, 
			"ievtcb" : None,
			"infiltration_limit_ts" : None,
			"infiltration_limit_use_timeseries" : None,
			"infiltration_limit_use_timeseries" : None,
			"infiltration_limit" : None,
			"infiltration_limit" : None,
			"init_interflow_store" : None,
			"init_interflow_store" : None,
			"interflow_decay_ts" : None,
			"interflow_decay_use_timeseries" : None,
			"interflow_decay" : None,
			"interflow_decay" : None,
			"interflow_store_bypass" : None,
			"interflow_zone_mapping" : None,
			"interflow_zone_names" : None,
			"irchcb" : None,
			"istcb1" : None,
			"istcb1" : None,
			"istcb2" : None,
			"istcb2" : None,
			"kc" : None,
			"landuse_zone_names" : {},
			"lu_spatial" : {1 : []},
			"macropore_activation_option" : None,
			"macropore_activation" : None,
			"macropore_limit" : None,
			"macropore_proportion" : None,
			"macropore_recharge" : None,
			"macropore_zone_mapping" : None,
			"macropore_zone_names" : None,
			"macropore_zone_names" : None,
			"max_canopy_storage" : None,
			"nevtopt" : None,
			"nodes_per_line" : None,
			"num_cores" : None,
			"num_nodes" : 1,
			"output_evt" : None,
			"output_fac" : None,
			"output_individual" : None,
			"output_recharge" : None,
			"output_sfr" : None,
			"output_sfr" : None,
			"pe_zone_mapping" : None,
			"pe_zone_mapping" : {},
			"pe_zone_names" : None,
			"percolation_rejection_use_timeseries" : None,
			"percolation_rejection" : None,
			"ponding_area" : None,
			"rainfall_zone_mapping" : {},
			"rainfall_zone_names" : None,
			"rapid_runoff_params" : None,
			"rapid_runoff_zone_mapping" : None,
			"rapid_runoff_zone_names" : None,
			"raw" : {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], },
			"recharge_attenuation_params" : None,
			"reporting_zone_mapping" : None,
			"reporting_zone_names" : None,
			"routing_process" : None,
			"routing_process" : None,
			"routing_topology" : None,
			"routing_topology" : None,
			"run_name" : "aardvark",
			"sfr_obs" : None,
			"sfr_obs" : None,
			"single_cell_swrecharge_activation" : None,
			"single_cell_swrecharge_limit" : None,
			"single_cell_swrecharge_proportion" : None,
			"single_cell_swrecharge_zone_mapping" : None,
			"single_cell_swrecharge_zone_names" : None,
			"snow_params_complex" : None,
			"soil_spatial" : {1 : [0]},
			"soil_static_params" : None,
			"soil_zone_names" : None,
			"soil_zone_names" : None,
			"spatial_output_date" : None,
			"start_date" : "2000-01-01",
			"subroot_zone_mapping" : None,
			"subroot_zone_names" : None,
			"subsoilzone_leakage_fraction" : None,
			"sw_activation" : None,
			"sw_activation" : None,
			"sw_bed_infiltration" : None,
			"sw_bed_infiltration" : None,
			"sw_direct_recharge" : None,
			"sw_direct_recharge" : None,
			"sw_downstream" : None,
			"sw_downstream" : None,
			"sw_init_ponding" : None,
			"sw_init_ponding" : None,
			"sw_params" : None,
			"sw_pe_to_open_water" : None,
			"sw_pe_to_open_water" : None,
			"sw_ponding_area" : None,
			"sw_zone_mapping" : None,
			"sw_zone_names" : None,
			"sw_zone_names" : None,
			"swabs_f" : None,
			"swabs_f" : None,
			"swabs_locs" : None,
			"swabs_locs" : None,
			"swdis_f" : None,
			"swdis_f" : None,
			"swdis_locs" : None,
			"swdis_locs" : None,
			"swrecharge_limit" : None,
			"swrecharge_process" : "disabled",
			"swrecharge_proportion" : None,
			"swrecharge_zone_mapping" : None,
			"swrecharge_zone_names" : None,
			"taw" : {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []},
			"temperature_zone_mapping" : None,
			"temperature_zone_names" : None,
			"tmax_c_zone_mapping" : None,
			"tmax_c_zone_names" : None,
			"tmin_c_zone_mapping" : None,
			"tmin_c_zone_names" : None,
			"windsp_zone_mapping" : None,
			"windsp_zone_names" : None,
			"zr" : None,
		}
	}