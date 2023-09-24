source env/bin/activate
python3 -m unittest discover -s test_full_runs
exit_status=$?
deactivate
exit $exit_status
