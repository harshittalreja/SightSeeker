@echo off
setlocal EnableExtensions
title SightSeeker - Demo Launcher
color 0B
rem Always run from the folder this file lives in (so models/HTML files are found)
cd /d "%~dp0"

rem ---- Find Python: your Python 3.12 install first, then the "py" launcher, then PATH ----
set "PYEXE="
set "PYARGS="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :menu
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=py"
    set "PYARGS=-3"
    goto :menu
)
where python >nul 2>nul
if not errorlevel 1 set "PYEXE=python"

:menu
cls
echo.
echo  ============================================
echo   SIGHTSEEKER - Vision-Based Autonomous Robot
echo   Hack in Hills '26
echo  ============================================
echo.
echo   [1] Pitch             the interactive pitch (F fullscreen, H speaker notes)
echo   [2] 3D Simulator      MAIN DEMO - opens in your browser
echo   [3] 2D Simulator      Pygame
echo   [4] Vision Brain      real YOLO26 on the webcam -^> steering decisions
echo   [5] Mission Control   dashboard (Pygame)
echo   [6] Install / check   packages, NVIDIA GPU, YOLO26 + open-vocab models
echo   [7] Exit
echo.
if not defined PYEXE echo   NOTE: Python not found - options 3-6 need Python 3.10+ (python.org)
if not defined PYEXE echo.
set "choice="
set /p "choice=  Enter choice (1-7): "

if "%choice%"=="1" goto :slides
if "%choice%"=="2" goto :sim3d
if "%choice%"=="3" goto :sim2d
if "%choice%"=="4" goto :vision
if "%choice%"=="5" goto :dashboard
if "%choice%"=="6" goto :install
if "%choice%"=="7" goto :end
goto :menu

:slides
echo   Opening the slides in your browser...
start "" "%~dp0SightSeeker_Pitch.html"
goto :menu

:sim3d
echo   Opening the 3D simulator in your browser...
start "" "%~dp0robot_simulation_3d.html"
goto :menu

:sim2d
call :need_python || goto :menu
echo   Starting the 2D simulator (close the window or press Q to return)...
"%PYEXE%" %PYARGS% robot_simulation.py
if %errorlevel% neq 0 pause
goto :menu

:vision
call :need_python || goto :menu
echo.
set "GOAL="
set /p "GOAL=  What should the robot look for? (Enter = bottle): "
if not defined GOAL set "GOAL=bottle"
set "OV="
set /p "OV=  Open-vocabulary mode (any object, slower)? y/N: "
set "OVFLAG="
if /i "%OV%"=="y" set "OVFLAG=--open-vocab"
set "MX="
set /p "MX=  MAXIMUM mode (best accuracy, uses the whole laptop, runs hot)? y/N: "
set "MXFLAG="
if /i "%MX%"=="y" set "MXFLAG=--max"
set "CAM="
set /p "CAM=  Camera: Enter = laptop webcam, or your phone's IP Webcam address (e.g. 192.168.1.5:8080): "
set "CAMFLAG="
if defined CAM set "CAMFLAG=--camera %CAM%"
set "WK="
set /p "WK=  Walk mode (it speaks directions while you carry the camera)? y/N: "
set "WKFLAG="
if /i "%WK%"=="y" set "WKFLAG=--walk"
set "TW="
set /p "TW=  Digital twin view (a virtual rover follows the wheel commands)? y/N: "
set "TWFLAG="
if /i "%TW%"=="y" set "TWFLAG=--twin"
echo   Starting Vision Brain with goal "%GOAL%" (press Q in the video window to quit)...
"%PYEXE%" %PYARGS% yolo_detection.py --goal "%GOAL%" %OVFLAG% %MXFLAG% %CAMFLAG% %WKFLAG% %TWFLAG%
if %errorlevel% neq 0 pause
goto :menu

:dashboard
call :need_python || goto :menu
echo   Starting Mission Control...
"%PYEXE%" %PYARGS% dashboard.py
if %errorlevel% neq 0 pause
goto :menu

:install
call :need_python || goto :menu
"%PYEXE%" %PYARGS% -m pip install -r requirements.txt
echo.
echo   Checking for an NVIDIA GPU...
"%PYEXE%" %PYARGS% check_gpu.py
if not errorlevel 2 goto :models
set "GPUQ="
set /p "GPUQ=  Install the NVIDIA GPU build of PyTorch now (about 3 GB download)? Y/n: "
if /i "%GPUQ%"=="n" goto :models
"%PYEXE%" %PYARGS% check_gpu.py --install
:models
echo.
echo   Downloading the YOLO26 and open-vocabulary models (one time)...
"%PYEXE%" %PYARGS% download_models.py
echo.
"%PYEXE%" %PYARGS% -c "import cv2, numpy, pygame, ultralytics; print('  All demo packages are installed.')"
pause
goto :menu

:need_python
if defined PYEXE exit /b 0
echo.
echo   Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
echo   (tick "Add python.exe to PATH"), then run option 6 to install the packages.
echo.
pause
exit /b 1

:end
endlocal
exit /b 0
