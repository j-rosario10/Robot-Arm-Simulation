@echo off
title Puppet to Pilot launcher
echo Starting Puppet to Pilot (4 windows will open)...
echo.
echo   1. CAMERA window: follow the calibration prompts
echo      (hold hand OPEN, then PINCHED). It retries until it sees you.
echo.
start "1 - camera (Windows)" powershell -NoExit -Command "cd 'C:\Robot Arm Project\windows_side'; .venv\Scripts\python pose_publisher.py"
timeout /t 3 >nul
start "2 - bridge (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/bridge.sh"
start "3 - simulator + recorder (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/session.sh"
timeout /t 5 >nul
start "4 - teleop control (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/teleop.sh"
echo All four started. The robot window appears once the simulator loads.
echo Close this window - or all of them - to stop.
timeout /t 15 >nul
