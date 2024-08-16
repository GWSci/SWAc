source env/bin/activate
python3 "compile_model.py"
python3 swacmod_run.py "$@"
deactivate
