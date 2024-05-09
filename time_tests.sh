source env/bin/activate
python3.11 -c "import swacmod.compile_model"
env TQDM_DISABLE=true python test/time_tests.py
exit_status=$?
deactivate
exit $exit_status
