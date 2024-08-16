source env/bin/activate
python3 -c "import compile_model"
python3 swacmod_run.py "$@"
deactivate
