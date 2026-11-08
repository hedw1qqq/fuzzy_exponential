@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check that Python 3.10+ and internet access are available.
  pause
  exit /b 1
)
echo.
echo Installation completed.
pause
