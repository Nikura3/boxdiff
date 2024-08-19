@echo off

title Dowloading script
echo Downloading your files from the cluster

setlocal enabledelayedexpansion

REM Variables
set user=dvezzaro
set source_dir=/home/dvezzaro/hf_gligen/dev/
set destination_dir=C:/Users/david/Desktop/Diffusers/gligen-test/results

REM Get the current date in YYYY-MM-DD format
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c-%%a-%%b

REM Get the current time with seconds
for /f "tokens=1-2 delims=:" %%a in ('time /t') do (
    set hour=%%a
    set minute=%%b
)

REM Remove leading spaces from hour and minute
set hour=%hour: =%
set minute=%minute: =%

REM Get current seconds
for /f "tokens=1,2 delims=:" %%a in ('echo %time%') do set second=%%b

REM Remove leading spaces from seconds
set second=%second: =%

REM Convert 12-hour format to 24-hour format if necessary
if "%time:~9,2%"=="PM" if "%hour%" neq "12" set /a hour+=12
if "%time:~9,2%"=="AM" if "%hour%"=="12" set hour=00

REM Format time with seconds
set mytime=%hour%-%minute%-%second%

REM Combine date and time
set destination_dir=%destination_dir%/%mydate%_%mytime%

mkdir "%destination_dir%"

echo - source folder:%source_dir%
echo - destination folder:%destination_dir%

scp -r -J %user%@labta.math.unipd.it %user%@labsrv8.math.unipd.it:%source_dir% %destination_dir%