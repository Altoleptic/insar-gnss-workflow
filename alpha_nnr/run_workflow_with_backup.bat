@echo off
echo GNSS-InSAR Workflow with Automatic Backup
echo ========================================
echo.

REM Get current date and time in format YYYY-MM-DD_HH-MM-SS
for /f "tokens=2 delims==" %%G in ('wmic os get localdatetime /value') do set datetime=%%G
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set hour=%datetime:~8,2%
set minute=%datetime:~10,2%
set second=%datetime:~12,2%
set timestamp=%year%-%month%-%day%_%hour%-%minute%-%second%

REM Create a backup before running the workflow
echo Creating backup before running workflow...
call create_backup.bat

REM Run the master script
echo.
echo Running workflow...
python master.py

REM Check if workflow completed successfully
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Workflow completed successfully!
    echo.
    echo You can find log information in workflow.log
    echo.
) else (
    echo.
    echo ========================================
    echo Workflow encountered errors.
    echo.
    echo Check workflow.log for details.
    echo.
    echo You can restore your code from the backup if needed using restore_backup.bat
)

pause
