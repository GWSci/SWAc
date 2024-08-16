discovery_root="test"
use_coverage=false
show_outdated_dependencies=false

for arg in "$@"
do
	if [[ "$arg" == "--all" ]]; then
		discovery_root="."
	elif [[ "$arg" == "--coverage" ]]; then
		use_coverage=true
	elif [[ "$arg" == "--dependencies" ]]; then
		show_outdated_dependencies=true
	elif [[ "$arg" == "--full" ]]; then
		discovery_root="."
		use_coverage=true
		show_outdated_dependencies=true
	else
		echo "Arg not recognised: $arg"
		exit 1
	fi
done

source env/bin/activate
python3 -c "import compile_model"

if [ "$use_coverage" = true ]; then
	env TQDM_DISABLE=true coverage run -m unittest discover --durations 10 -s $discovery_root
else
	env TQDM_DISABLE=true python3 -m unittest discover --durations 10 --s $discovery_root
fi

exit_status=$?

if [ "$use_coverage" = true ]; then
	coverage report -m
	coverage html
fi

if [ "$show_outdated_dependencies" = true ]; then
	pip list --outdated
fi
deactivate
exit $exit_status
