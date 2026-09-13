@echo off
setlocal
title Publish Puppet to Pilot to GitHub
cd /d "%~dp0"

set "GH=gh"
where gh >nul 2>&1 || set "GH=%ProgramFiles%\GitHub CLI\gh.exe"

echo ============================================================
echo   Publishing "Puppet to Pilot" to GitHub
echo ============================================================
echo.

"%GH%" auth status >nul 2>&1
if errorlevel 1 (
    echo You are not signed in to GitHub yet.
    echo.
    echo A sign-in will start now. Answer like this:
    echo    What account?      GitHub.com
    echo    Protocol?          HTTPS
    echo    Authenticate Git?  Yes
    echo    How to login?      Login with a web browser
    echo.
    echo It shows you a one-time code, then opens your browser.
    echo Paste the code there and approve.
    echo.
    pause
    "%GH%" auth login
    if errorlevel 1 (
        echo.
        echo Sign-in did not complete. Run this file again when you are ready.
        pause
        exit /b 1
    )
)

echo.
echo Signed in as:
"%GH%" api user --jq ".login"
echo.

echo Creating the repository and pushing...
echo.
"%GH%" repo create puppet-to-pilot --public --source=. --remote=origin --push ^
    --description "Vision-teleoperated UR5e in simulation, with a behaviour-cloning policy trained on the recorded demonstrations. From-scratch kinematics, MediaPipe teleop over a Windows/WSL socket, ROS 2 + PyBullet."

if errorlevel 1 (
    echo.
    echo That did not work. Most likely you already have a repo with this name.
    echo Try a different name, for example:
    echo    "%GH%" repo create puppet-to-pilot-ur5e --public --source=. --remote=origin --push
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done. Your repo:
"%GH%" repo view --json url --jq ".url"
echo ============================================================
echo.
echo Opening it in your browser...
"%GH%" repo view --web
pause
