import unittest
from swacmod_run import run
import swacmod.utils as u
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource
from test.dummy_environment import Dummy_Environment

class Test_Inmput_File_With_Missing_SMD(unittest.TestCase):
    def test_x(self):
        filename = 'aaa.yml'
        input_dir = ''
        input_file = make_input_file()
        input_file_contents = ""
        for k, v in input_file.items():
            input_file_contents += f"{k}: {v}\n"
        mock_file_open = MockFileResource.make_mock_file_opener({filename: input_file_contents})

        default_input_file = u.CONSTANTS["INPUT_FILE"]
        default_input_dir = u.CONSTANTS["INPUT_DIR"]
        default_output_dir = u.CONSTANTS["OUTPUT_DIR"]
        try:
            u.CONSTANTS["INPUT_FILE"] = filename
            u.CONSTANTS["INPUT_DIR"] = input_dir
            run(test=False, skip=True, env=Dummy_Environment(), file_opener=mock_file_open)
        finally:
             u.CONSTANTS["INPUT_FILE"] = default_input_file
             u.CONSTANTS["INPUT_DIR"] = default_input_dir

def make_input_file():
        return {
            "version": 2,
            "run_name": "my_run",
            "temp_file_backed_array_directory": "temp_scratch_files/",
            "num_nodes": 2,
            "node_areas": {1: 10.0, 2:10.0},
            "start_date": "1980-01-01",
            "time_periods": [[1, 2]],
            "num_cores": 1,
            "output_recharge": False,
            "reporting_zone_mapping": {1: 1, 2: 1},
            "reporting_zone_names": {1:'name'},
            "rainfall_zone_mapping": {1: [1, 1.0]},
            "rainfall_zone_names": {1: 'name'},
            "pe_zone_mapping": {1: [1, 1.0]},
            "pe_zone_names": {1: 'name'},
            "soil_zone_names": {1: 'name'},
            "canopy_process": "enabled",
            "canopy_zone_mapping": {1: 1},
            "canopy_zone_names": {1: 'name'},
            "free_throughfall": {1: 0.99},
            "max_canopy_storage": {1: 1.0},
            "snow_process_simple": "disabled",
            "snow_process_complex": "disabled",
            "rapid_runoff_process": "disabled",
            "swrecharge_process": "disabled",
            "single_cell_swrecharge_process": "disabled",
            "macropore_process": "disabled",
            "fao_process": "disabled",

            "smd": {'starting_SMD': [100]},
            "soil_spatial": {1: [0.5]},

            
            # "percolation_rejection": "percolation_rejection.yml",
            # "percolation_rejection_use_timeseries": "true",
            # "percolation_rejection_ts": "percolation_rejection_ts.yml",

            "subroot_leakage_process": "disabled",
            "interflow_process": "disabled",
            "recharge_attenuation_process": "disabled",
            "historical_solute_process": "disabled",
            "solute_process": "disabled",
            "sw_process": "disabled",
            "sw_ponding_process": "disabled",
            "routing_process": "disabled",
            "output_sfr": False,
            "attenuate_sfr_flows": False,
            "rainfall_ts": [[7.19625], [0.09595]],
            "pe_ts": [[3.774], [3.774]],
            
            "gwmodel_type": "mf6",
            "mf96_lrc": [2, 2, 5],
            "output_evt": False,
            "excess_sw_process": "disabled"
        }


