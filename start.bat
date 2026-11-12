@echo off
rem Starts Puppet to Pilot. Double-click this file.
cd /d "%~dp0"
where pixi >nul 2>nul
if errorlevel 1 (
    "%LOCALAPPDATA%\pixi\bin\pixi.exe" run launch
) else (
    pixi run launch
)
pause
