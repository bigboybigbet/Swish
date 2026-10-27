@echo off
rem Double-click to start Swish (installs dependencies on the first run only).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 start.py %*) else (python start.py %*)
if errorlevel 1 pause
