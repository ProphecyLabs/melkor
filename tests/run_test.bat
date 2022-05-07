@echo off
python -m venv venv

venv\\Scripts\\pip.exe install pandas

venv\\Scripts\\python.exe tests/test_clean.py
