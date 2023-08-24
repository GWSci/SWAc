import os

_include_cython_in_perf_features = True

use_perf_features = "True" == os.environ.get("SWAc_use_perf_features", "False")

use_cython = _include_cython_in_perf_features if use_perf_features else True

# max_node_count_override = 4181
max_node_count_override = 1000000 #46368
