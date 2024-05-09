discovery_root="test"

for arg in "$@"
do
	if [[ "$arg" == "--all" ]]; then
		discovery_root="."
	else
		echo "Arg not recognised: $arg"
		exit 1
	fi
done

source env/bin/activate
python3.11 -c "import swacmod.compile_model"
env TQDM_DISABLE=true coverage run -m unittest discover -s $discovery_root
exit_status=$?
coverage report -m
deactivate
exit $exit_status
