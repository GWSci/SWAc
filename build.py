import sys
import shutil
import subprocess
import os
import datetime

version_filename = '_version.py'
commit_id_filename = '_commit_id.py'
build_time_filename = '_build_time.py'

def _default_file_open(filename, how='r'):
        return open(filename, how)

def get_old_version_string(filename=version_filename, file_open=_default_file_open):
    with file_open(filename, 'r') as file:
        old_version = file.readlines()[0]
    old_version = old_version.replace('\n', '')
    old_version = old_version.replace(' ', '')
    old_version = old_version.replace('[', '')
    old_version = old_version.replace(']', '')
    old_version = old_version.replace(',', '.')
    old_version = old_version.replace('version', '')
    old_version = old_version.replace('=', '')
    return old_version

def write_new_commit_id(sha, filename=commit_id_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'commit_id = "{sha}"')

def format_daytime(date):
    return date.strftime("%d %b %Y %H:%M:%S")

def format_daytime_for_filename(date):
    return date.strftime("%Y-%m-%dT%H-%M-%S")

def write_new_build_time(formated_datetime, filename=build_time_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'build_time = "{formated_datetime}"')

def build(sha, date):
    if (os.path.exists("build/")):
        shutil.rmtree("build/")
    if (os.path.exists("dist/")):
        shutil.rmtree("dist/")
    if (os.path.exists("release/")):
        shutil.rmtree("release/")

    os.mkdir("release/")

    release_filename = conjure_release_filename(sha, date)

    if sys.platform == "win32":
        python_binary = "env/Scripts/python"
        zip_v2_csv_input_files_command = "powershell Compress-Archive input_files_v2_csv/*.* dist/input_files_v2_csv.zip"
        zip_v2_yml_input_files_command = "powershell Compress-Archive input_files_v2_yml/*.* dist/input_files_v2_yml.zip"
        zip_release_command = f"powershell Compress-Archive dist/*.* release/{release_filename}"
    else:
        python_binary = "env/bin/python3"
        zip_v2_csv_input_files_command = "zip --quiet --recurse-paths dist/input_files_v2_csv.zip input_files_v2_csv/"
        zip_v2_yml_input_files_command = "zip --quiet --recurse-paths dist/input_files_v2_yml.zip input_files_v2_yml/"
        zip_release_command = f"zip --quiet --recurse-paths release/{release_filename} dist/"

    subprocess.run([python_binary, "compile_model.py"])

    subprocess.run([
        python_binary,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--add-data",
        "./swacmod/input_files/input_files_version_1/specs.yml:./swacmod/input_files/input_files_version_1/",
        "--hidden-import",
        "swacmod.snow_melt",
        "--hidden-import",
        "swacmod.networkx_adaptor",
        "--onefile",
        "swacmod_run.py",
    ])

    subprocess.run(["pandoc", "doc/getting-started.md", "-o", "dist/getting-started.html"])
    shutil.copy("doc/SWAcUserGuide.pdf", "dist/SWAcUserGuide.pdf")
    shutil.copy("doc/SWAcFlowChart.png", "dist/SWAcFlowChart.png")

    subprocess.run(zip_v2_csv_input_files_command, shell=True)
    subprocess.run(zip_v2_yml_input_files_command, shell=True)
    subprocess.run(zip_release_command, shell=True)

def conjure_release_filename(sha, date):
    build_time = format_daytime_for_filename(date)
    version = get_old_version_string()
    commit_id = sha[:8]
    platform = sys.platform
    return f"SWAcMod-v{version}.{commit_id}-{platform}-{build_time}.zip"

def convert_sys_platform_to_platform(sys_platform):
    return "aardvark"

def set_build_details():
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
    sha = sha.rstrip()
    write_new_commit_id(sha)

    date = datetime.datetime.now()
    formated_datetime = format_daytime(date)
    write_new_build_time(formated_datetime)

    return sha, date

def restore_build_details():
    subprocess.run(['git', 'restore', commit_id_filename])
    subprocess.run(['git', 'restore', build_time_filename])

def main():
    sha, date = set_build_details()
    build(sha, date)
    restore_build_details()

if __name__ == '__main__':
    main()
