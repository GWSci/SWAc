@echo off
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Cannot find Python. Please install it from www.python.org and rerun this script.
    exit /b
)

@echo on
python -m venv env || exit /b

call env/Scripts/activate.bat || exit /b
pip install -r requirements3.txt || exit /b
deactivate || exit /b