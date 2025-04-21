import sys
import subprocess

args = sys.argv[1:]
python_binary = "env/bin/python3"
coverage_binary = "env/bin/coverage"

discovery_root = "test"
use_coverage = False
show_outdated_dependencies = False
run_linter = False

for arg in args:
    if arg == "--all":
        discovery_root="."
    elif arg == "--coverage":
        use_coverage = True
    elif arg == "--dependencies":
        show_outdated_dependencies = True
    elif arg == "--lint":
        run_linter = True
    elif arg == "--full":
        discovery_root="."
        use_coverage = True
        show_outdated_dependencies = True
        run_linter = True
    else:
        print(f"Arg not recognised: {arg}")
        sys.exit(1)

subprocess.run([python_binary, "compile_model.py"])

if use_coverage:
    pass
    # env TQDM_DISABLE=true coverage run -m unittest discover --durations 10 -s $discovery_root
else:
    subprocess.run(
        [python_binary, "-m", "unittest", "discover", "--durations", "10", "--s", discovery_root],
        env={"TQDM_DISABLE": "true"})
