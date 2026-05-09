@echo off
REM build_with_make.cmd - Initialize MSVC environment and run Make
REM Usage: build_with_make.cmd

setlocal EnableDelayedExpansion
echo Initializing Visual Studio environment and running Make...

set "PF86=%ProgramFiles(x86)%"
set "PF=%ProgramFiles%"
set "VSWHERE=!PF86!\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
  "!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\vswhere_out.txt" 2>nul
  if exist "%TEMP%\vswhere_out.txt" (
    set /p VSINSTALL=<"%TEMP%\vswhere_out.txt"
    del "%TEMP%\vswhere_out.txt" 2>nul
  )
)

if defined VSINSTALL (
  echo Using Visual Studio at: !VSINSTALL!
  call "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
) else (
  echo WARNING: Visual Studio installation not found. Open Developer Command Prompt instead.
)

REM vcvars can override PATH; prepend CUDA bin
REM Priority: existing CUDA_BIN env var (user override) -> auto-detect latest installed
if not defined CUDA_BIN (
  for /f "delims=" %%d in ('dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do set "CUDA_VER=%%d"
  if defined CUDA_VER set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\!CUDA_VER!\bin"
)
if defined CUDA_BIN (
  if exist "!CUDA_BIN!\nvcc.exe" (
    echo Using CUDA bin: !CUDA_BIN!
    set "PATH=!CUDA_BIN!;!PATH!"
  )
)

cd /d "%~dp0"
echo Running make all in %CD%

where nvcc >nul 2>&1
if errorlevel 1 (
  echo ERROR: nvcc not found on PATH.
  exit /b 1
)

nvcc --list-gpu-arch | findstr /C:"compute_61" >nul
if errorlevel 1 (
  echo ERROR: Current CUDA compiler does not support sm_61 ^(GTX 1050 Ti^).
  echo Installed toolkit is likely CUDA 13.x where Pascal support is removed.
  echo Install CUDA 12.x, then run this script again.
  exit /b 1
)

where make >nul 2>&1
if errorlevel 1 (
  if exist "!PF86!\GnuWin32\bin\make.exe" (
    "!PF86!\GnuWin32\bin\make.exe" all
  ) else if exist "!PF!\GnuWin32\bin\make.exe" (
    "!PF!\GnuWin32\bin\make.exe" all
  ) else (
    echo ERROR: make.exe not found. Install GNU Make or add it to PATH.
    exit /b 1
  )
) else (
  make all
)

endlocal
exit /b %ERRORLEVEL%
