import os

use_perf_features = "True" == os.environ.get("SWAc_use_perf_features", "False")

max_node_count_override = 4181
# max_node_count_override = 1000000
