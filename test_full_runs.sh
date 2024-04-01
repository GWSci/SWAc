source env/bin/activate
coverage run -m unittest discover -s test_full_runs
exit_status=$?
coverage report -m
deactivate
exit $exit_status
