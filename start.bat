@echo off
chcp 65001 >nul
cd /d "%~dp0"
py app.py
if errorlevel 1 (
  echo.
  echo Launch failed. Run install.bat first.
  pause
)
