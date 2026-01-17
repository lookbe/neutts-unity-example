@echo off
:: --- Configuration Section ---

:: Set your JAVA_HOME path
set "JAVA_HOME=C:\Program Files\Unity\Hub\Editor\6000.0.58f2\Editor\Data\PlaybackEngines\AndroidPlayer\OpenJDK"

:: Set the path to your Android SDK build-tools (where jobb.bat lives)
set "JOBB_PATH=C:\Program Files\Unity\Hub\Editor\6000.0.58f2\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\tools\bin"

:: Set your Android Package Name and Version
set "PACKAGE_NAME=com.lookbe.neutts"
set "VERSION=1"

:: Set the output filename
set "OUTPUT_NAME=main.%VERSION%.%PACKAGE_NAME%.obb"

:: --- Execution Section ---

:: Add JAVA_HOME and JOBB to the temporary PATH
set "PATH=%JAVA_HOME%\bin;%JOBB_PATH%;%PATH%"

echo Creating OBB from folder 'obb'...
echo Package: %PACKAGE_NAME%
echo Version: %VERSION%

:: Run the jobb tool
:: -d: input directory, -o: output file, -pn: package name, -pv: version
call jobb.bat -d .\Assets\StreamingAssets -o %OUTPUT_NAME% -pn %PACKAGE_NAME% -pv %VERSION%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Success! Created %OUTPUT_NAME%
) else (
    echo.
    echo Error: Failed to create OBB file.
)

pause