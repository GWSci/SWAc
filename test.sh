source env/bin/activate
python3 -m unittest discover -s test
exit_status=$?
deactivate
exit $exit_status
