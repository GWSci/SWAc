rmdir build /s /q
rmdir dist /s /q

call env/Scripts/activate.bat
python "compile_model.py"
pyinstaller --clean --noconfirm --add-data .\swacmod\specs.yml:.\swacmod\ --hidden-import swacmod.snow_melt --hidden-import swacmod.networkx_adaptor --onefile swacmod_run.py
call env\Scripts\deactivate.bat
