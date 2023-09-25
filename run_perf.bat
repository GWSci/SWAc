call env/Scripts/activate.bat
set SWAc_use_perf_features=True
python swacmod_run.py %*
deactivate
