@echo off
setlocal enabledelayedexpansion

REM Get script directory to establish absolute paths
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..

REM Get current date and time in format YYYY-MM-DD_HH-MM-SS
for /f "tokens=2 delims==" %%G in ('wmic os get localdatetime /value') do set datetime=%%G
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set hour=%datetime:~8,2%
set minute=%datetime:~10,2%
set second=%datetime:~12,2%
set timestamp=%year%-%month%-%day%_%hour%-%minute%-%second%

REM Define backup directory with absolute path
set backup_dir=%SCRIPT_DIR%backups\%timestamp%

REM Create backup directory
mkdir "%backup_dir%"

REM Copy all Python files to backup directory
echo Creating backup of all Python scripts in %backup_dir%...
xcopy "%ROOT_DIR%\*.py" "%backup_dir%\" /Y

REM Create a version info file
echo # Backup created on %timestamp% > "%backup_dir%\version_info.txt"
echo. >> "%backup_dir%\version_info.txt"
echo ## Files included: >> "%backup_dir%\version_info.txt"
dir /B "%ROOT_DIR%\*.py" >> "%backup_dir%\version_info.txt"

echo.
echo Backup completed successfully!
echo Backup location: %backup_dir%
echo.
echo To restore from this backup, use tools\restore_backup.bat
