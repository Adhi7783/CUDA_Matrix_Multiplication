@echo off
REM validate_env.cmd - Simplified validator for MSVC, CUDA, and Make
REM Usage: validate_env.cmd [make]

setlocal EnableDelayedExpansion
echo === Environment validation for CUDA build ===

REM Attempt to locate vswhere
set "PF86=%ProgramFiles(x86)%"
set "PF=%ProgramFiles%"
set "VSWHERE=!PF86!\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
  echo Running vswhere to locate Visual Studio installation...
  "!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\vswhere_out.txt" 2>nul
  if exist "%TEMP%\vswhere_out.txt" (
    set /p VSINSTALL=<"%TEMP%\vswhere_out.txt"
    del "%TEMP%\vswhere_out.txt" 2>nul
  )
)

if defined VSINSTALL (
  echo Found Visual Studio at: !VSINSTALL!
  echo Calling vcvars64.bat to set MSVC environment...
  call "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
  if errorlevel 1 echo WARNING: vcvars64.bat returned an error
) else (
  echo WARNING: Visual Studio installation not found via vswhere.
  echo If you have Visual Studio installed, run this from a Developer Command Prompt.
)

REM vcvars can override PATH; prepend CUDA bin
REM Priority: existing CUDA_BIN env var (user override) -> auto-detect latest installed
if not defined CUDA_BIN (
  for /f "delims=" %%d in ('dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do set "CUDA_VER=%%d"
  if defined CUDA_VER set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\!CUDA_VER!\bin"
)
if defined CUDA_BIN (
  if exist "!CUDA_BIN!\nvcc.exe" (
    set "PATH=!CUDA_BIN!;!PATH!"
  )
)

echo.
echo -- Checking MSVC compiler (cl.exe)
where cl.exe >nul 2>&1
if errorlevel 1 (
  echo cl.exe not found on PATH. Run the Developer Command Prompt or call vcvars64.bat first.
) else (
  where cl.exe
  cl.exe
)

echo.
echo -- Checking CUDA (nvcc)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo nvcc not found on PATH
  if defined CUDA_BIN echo Tried CUDA bin: !CUDA_BIN!
) else (
  nvcc --version
  echo Checking support for GTX 1050 Ti target ^(compute_61^) ...
  nvcc --list-gpu-arch | findstr /C:"compute_61" >nul
  if errorlevel 1 (
    echo WARNING: This nvcc does NOT support compute_61 ^(sm_61^).
    echo Your project target GPU GTX 1050 Ti requires sm_61.
    echo Install CUDA 12.x for Pascal support, or build for a newer GPU arch.
  ) else (
    echo OK: nvcc supports compute_61.
  )
)

echo.
echo -- Checking GNU Make
where make >nul 2>&1
if errorlevel 1 (
  echo make not found on PATH
  echo Checking common locations...
  if exist "!PF86!\GnuWin32\bin\make.exe" (
    echo Found make at !PF86!\GnuWin32\bin\make.exe
    "!PF86!\GnuWin32\bin\make.exe" --version
  ) else if exist "!PF!\GnuWin32\bin\make.exe" (
    echo Found make at !PF!\GnuWin32\bin\make.exe
    "!PF!\GnuWin32\bin\make.exe" --version
  ) else (
    echo To install Make, use winget or Chocolatey as documented in SETUP_WINDOWS.md
  )
) else (
  where make
  make --version
)

echo.
echo === Validation complete ===

REM Optionally run make if first arg is 'make'
if "%~1"=="make" (
  echo Running make all...
  where make >nul 2>&1
  if errorlevel 1 (
    if exist "!PF86!\GnuWin32\bin\make.exe" (
      "!PF86!\GnuWin32\bin\make.exe" all
    ) else if exist "!PF!\GnuWin32\bin\make.exe" (
      "!PF!\GnuWin32\bin\make.exe" all
    ) else (
      echo Cannot run make all: make.exe not found.
      exit /b 1
    )
  ) else (
    make all
  )
)

endlocal
exit /b 0
