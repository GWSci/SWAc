discovery_root="test"
use_coverage=false
show_outdated_dependencies=false
run_linter=false

for arg in "$@"
do
	if [[ "$arg" == "--all" ]]; then
		discovery_root="."
	elif [[ "$arg" == "--coverage" ]]; then
		use_coverage=true
	elif [[ "$arg" == "--dependencies" ]]; then
		show_outdated_dependencies=true
	elif [[ "$arg" == "--lint" ]]; then
		run_linter=true
	elif [[ "$arg" == "--full" ]]; then
		discovery_root="."
		use_coverage=true
		show_outdated_dependencies=true
		run_linter=true
	else
		echo "Arg not recognised: $arg"
		exit 1
	fi
done

source env/bin/activate
python3 "compile_model.py"

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

if [ "$run_linter" = true ]; then
	deactivate
	source env-lint/bin/activate

	Pylint \
		--extension-pkg-allow-list=swacmod.model \
		--disable=R,C,W \
		--ignore=env,env-lint \
		--generated-member=flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr.obs \
		.

	deactivate
	source env/bin/activate
fi

if [ "$show_outdated_dependencies" = true ]; then
	pip list --outdated
fi
deactivate
exit $exit_status
