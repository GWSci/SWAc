import os

use_perf_features = "True" == os.environ.get("SWAc_use_perf_features", "False")

_node_count_override = True
use_node_count_override = _node_count_override if use_perf_features else False

max_node_count_override = 4181
# max_node_count_override = 1000000
