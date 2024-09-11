rm -rf build/
rm -rf dist/

source env/bin/activate
python3 "compile_model.py"
pyinstaller --clean --noconfirm --add-data ./swacmod/specs.yml:./swacmod/ --hidden-import swacmod.snow_melt --hidden-import swacmod.networkx_adaptor --onefile swacmod_run.py
deactivate

pandoc doc/getting-started.md -o dist/getting-started.html

zip --quiet --recurse-paths dist/input_files.zip input_files/
