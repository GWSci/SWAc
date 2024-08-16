source env/bin/activate
python3 "compile_model.py"
pyinstaller --add-data ./swacmod/specs.yml:./swacmod/ --hidden-import swacmod.snow_melt --hidden-import swacmod.networkx_adaptor swacmod_run.py
deactivate
