source env/bin/activate
export SWAc_use_perf_features=True
python3 swacmod_run.py "$@"
deactivate
