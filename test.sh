discovery_root="test"
use_coverage=false

for arg in "$@"
do
	if [[ "$arg" == "--all" ]]; then
		discovery_root="."
	elif [[ "$arg" == "--coverage" ]]; then
		use_coverage=true
	else
		echo "Arg not recognised: $arg"
		exit 1
	fi
done

source env/bin/activate
python3.11 -c "import swacmod.compile_model"

if [ "$use_coverage" = true ]; then
	env TQDM_DISABLE=true coverage run -m unittest discover -s $discovery_root
else
	env TQDM_DISABLE=true python -m unittest discover -s $discovery_root
fi

exit_status=$?

if [ "$use_coverage" = true ]; then
	coverage report -m
fi
deactivate
exit $exit_status
