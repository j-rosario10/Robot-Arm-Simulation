@echo off
title Puppet to Pilot (no camera demo)
echo Starting the NO-CAMERA demo: a synthetic operator drives the arm.
start "1 - synthetic operator (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/synthetic.sh"
timeout /t 2 >nul
start "2 - bridge (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/bridge.sh" --host 127.0.0.1
start "3 - simulator (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/session.sh"
timeout /t 5 >nul
start "4 - teleop control (WSL)" wsl.exe -d Ubuntu-24.04 -- bash "/mnt/c/Robot Arm Project/launch/teleop.sh"
echo Watch the robot window: the arm moves in a figure-eight on its own.
timeout /t 15 >nul
