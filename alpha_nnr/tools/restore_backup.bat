@echo off
setlocal enabledelayedexpansion

REM Get script directory to establish absolute paths
set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..

echo GNSS-InSAR Scripts - Restore from Backup
echo ======================================
echo.

REM Check if backups directory exists
if not exist "%SCRIPT_DIR%backups\" (
    echo No backups found. Please create a backup first using tools\create_backup.bat
    pause
    exit /b 1
)

REM List available backups
echo Available backups:
echo.
set count=1
for /d %%D in ("%SCRIPT_DIR%backups\*") do (
    set "backups[!count!]=%%~nxD"
    echo !count!: %%~nxD
    set /a count+=1
)

if %count% leq 1 (
    echo No backups found. Please create a backup first using tools\create_backup.bat
    pause
    exit /b 1
)

echo.
set /p choice=Enter the number of the backup to restore: 

REM Validate input
if %choice% lss 1 (
    echo Invalid selection.
    pause
    exit /b 1
)
if %choice% geq %count% (
    echo Invalid selection.
    pause
    exit /b 1
)

set backup_timestamp=!backups[%choice%]!
set backup_dir=%SCRIPT_DIR%backups\%backup_timestamp%

REM Confirm restoration
echo.
echo You are about to restore from backup: %backup_timestamp%
echo This will overwrite your current Python scripts.
echo.
set /p confirm=Are you sure you want to continue? (y/n): 

if /i not "%confirm%"=="y" (
    echo Restore cancelled.
    pause
    exit /b 0
)

REM Create a backup of current state before restoring
call %SCRIPT_DIR%create_backup.bat

REM Restore from the selected backup
echo.
echo Restoring from backup %backup_timestamp%...
xcopy "%backup_dir%\*.py" "%ROOT_DIR%\" /Y

echo.
echo Restore completed successfully!
pause
