@echo off
REM use_cuda12.cmd - Use CUDA 12.x for GTX 1050 Ti (sm_61), then build/test
REM Usage: use_cuda12.cmd [all|test|bench|validate]

setlocal EnableDelayedExpansion

set "ACTION=all"
if not "%~1"=="" set "ACTION=%~1"

set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
set "CUDA12_BIN="

if not exist "%CUDA_ROOT%" (
  echo ERROR: CUDA root not found at "%CUDA_ROOT%".
  echo Install CUDA 12.x (recommended: CUDA 12.9) and run this script again.
  exit /b 1
)

REM Prefer an explicit 12.9 installation if present (recommended for Pascal/GTX 1050 Ti)
if exist "%CUDA_ROOT%\v12.9" (
  set "CUDA12_BIN=%CUDA_ROOT%\v12.9\bin"
) else (
  for /f "delims=" %%d in ('dir /b /ad "%CUDA_ROOT%\v12*" 2^>nul') do (
    set "CUDA12_BIN=%CUDA_ROOT%\%%d\bin"
  )
)

if not defined CUDA12_BIN (
  echo ERROR: No CUDA 12.x installation found under "%CUDA_ROOT%".
  echo Install CUDA 12.x ^(Pascal-compatible^) and re-run.
  exit /b 1
)

if not exist "!CUDA12_BIN!\nvcc.exe" (
  echo ERROR: nvcc.exe not found in "!CUDA12_BIN!".
  exit /b 1
)

set "CUDA_BIN=!CUDA12_BIN!"
set "PATH=!CUDA_BIN!;!PATH!"

echo Using CUDA_BIN: !CUDA_BIN!
"!CUDA_BIN!\nvcc.exe" --version

REM Validate environment (also initializes MSVC)
call "%~dp0validate_env.cmd"
if errorlevel 1 (
  echo ERROR: validate_env.cmd failed.
  exit /b 1
)

if /I "%ACTION%"=="validate" (
  echo Validation only completed.
  exit /b 0
)

if /I "%ACTION%"=="all" (
  echo Running build_with_make.cmd ...
  call "%~dp0build_with_make.cmd"
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="test" (
  echo Running make test ...
  call "%~dp0build_with_make.cmd"
  if errorlevel 1 exit /b 1
  where make >nul 2>&1
  if errorlevel 1 (
    if exist "%ProgramFiles(x86)%\GnuWin32\bin\make.exe" (
      "%ProgramFiles(x86)%\GnuWin32\bin\make.exe" test
    ) else if exist "%ProgramFiles%\GnuWin32\bin\make.exe" (
      "%ProgramFiles%\GnuWin32\bin\make.exe" test
    ) else (
      echo ERROR: make.exe not found.
      exit /b 1
    )
  ) else (
    make test
  )
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="bench" (
  echo Running make bench ...
  call "%~dp0build_with_make.cmd"
  if errorlevel 1 exit /b 1
  where make >nul 2>&1
  if errorlevel 1 (
    if exist "%ProgramFiles(x86)%\GnuWin32\bin\make.exe" (
      "%ProgramFiles(x86)%\GnuWin32\bin\make.exe" bench
    ) else if exist "%ProgramFiles%\GnuWin32\bin\make.exe" (
      "%ProgramFiles%\GnuWin32\bin\make.exe" bench
    ) else (
      echo ERROR: make.exe not found.
      exit /b 1
    )
  ) else (
    make bench
  )
  exit /b %ERRORLEVEL%
)

echo Unknown action: %ACTION%
echo Usage: use_cuda12.cmd [all^|test^|bench^|validate]
exit /b 1
