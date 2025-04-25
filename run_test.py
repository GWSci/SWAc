import sys
import subprocess
import os

args = sys.argv[1:]

if sys.platform == "win32":
    python_binary = "env/Scripts/python"
    coverage_binary = "env/Scripts/coverage"
    linter_binary = "env-lint/Scripts/Pylint"
    pip_binary = "env/Scripts/pip"
    linter_pip_binary = "env-lint/Scripts/pip"
else:
    python_binary = "env/bin/python3"
    coverage_binary = "env/bin/coverage"
    linter_binary = "env-lint/bin/Pylint"
    pip_binary = "env/bin/pip"
    linter_pip_binary = "env-lint/bin/pip"

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

if sys.platform == "win32":
    env = dict(os.environ)
    env["TQDM_DISABLE"] = "true"
    if use_coverage:
        result = subprocess.run(
            [coverage_binary, "run", "-m", "unittest", "discover", "-s", discovery_root],
            env=env)
    else:
        result = subprocess.run(
            [python_binary, "-m", "unittest", "discover", "--s", discovery_root],
            env=env)
else:
    if use_coverage:
        result = subprocess.run(
            [coverage_binary, "run", "-m", "unittest", "discover", "--durations", "10", "-s", discovery_root],
            env={"TQDM_DISABLE": "true"})
    else:
        result = subprocess.run(
            [python_binary, "-m", "unittest", "discover", "--durations", "10", "--s", discovery_root],
            env={"TQDM_DISABLE": "true"})

exit_status = result.returncode

if use_coverage:
    subprocess.run([coverage_binary, "report", "-m"])
    subprocess.run([coverage_binary, "html"])

if run_linter:
    subprocess.run([
        linter_binary,
        "--extension-pkg-allow-list=swacmod.model",
        "--disable=R,C,W",
        "--ignore=env,env-lint",
        "--generated-member=flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr.obs",
        ".",
    ])

if show_outdated_dependencies:
    print("\nOutdated dependencies from env:\n")
    subprocess.run([pip_binary, "list", "--outdated"])

    print("\nOutdated dependencies from env-lint:\n")
    subprocess.run([linter_pip_binary, "list", "--outdated"])

sys.exit(exit_status)
