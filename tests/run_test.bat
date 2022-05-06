@echo off
python -m venv venv

venv\\Scripts\\pip.exe install -r requirements.txt

venv\\Scripts\\python.exe tests/test_clean.py