import os

# Flags values when performance features are enabled.

_node_count_override = False
_max_node_count_override = 4181
_disable_multiprocessing = True
_skip_validation = False

# Set master flag from environment variable

_use_perf_features = "True" == os.environ.get("SWAc_use_perf_features", "False")

# Flags to query in code

use_node_count_override = _node_count_override and _use_perf_features
max_node_count_override = _max_node_count_override if use_node_count_override else 10000000
disable_multiprocessing = _disable_multiprocessing and _use_perf_features
skip_validation = _skip_validation and _use_perf_features 
