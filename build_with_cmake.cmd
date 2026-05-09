@echo off
REM build_with_cmake.cmd - Initialize MSVC environment and run CMake build with MSVC
REM Usage: build_with_cmake.cmd [Release|Debug]

setlocal EnableDelayedExpansion
set CONFIG=Release
if not "%~1"=="" set CONFIG=%~1

set "PF86=%ProgramFiles(x86)%"
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
if not exist build mkdir build
cd build
echo Running CMake configure...
cmake -G "Visual Studio 17 2022" -A x64 ..
if errorlevel 1 (
  echo CMake configure failed
  exit /b 1
)
echo Building configuration: %CONFIG%
cmake --build . --config %CONFIG%

endlocal
exit /b %ERRORLEVEL%
