import os

use_perf_features = "True" == os.environ.get("SWAc_use_perf_features", "False")

_node_count_override = False
use_node_count_override = _node_count_override if use_perf_features else False
max_node_count_override = 4181 if use_node_count_override else 10000000

_disable_multiprocessing = True
disable_multiprocessing = _disable_multiprocessing if use_perf_features else False
