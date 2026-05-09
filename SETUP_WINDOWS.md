# Windows Build Environment Setup (Visual Build Tools, CUDA Toolkit, Make)

This guide installs the required tools to build and run the CUDA Matrix Multiplication project on Windows. It targets modern Windows and recommends Visual Studio 2022 for IDE or Build Tools.

Prerequisites
- Windows 10/11 with admin privileges
- Internet connection for downloads

Overview
- Visual C++ Build Tools (MSVC): provides `cl.exe` used by CUDA's host compiler
- NVIDIA CUDA Toolkit: provides `nvcc`, `cuBLAS`, headers, and libraries
- GNU Make (optional, used by the provided `Makefile`) or use CMake + MSVC

1) Install Visual Studio (or Build Tools) (MSVC) — recommended: 2022

-- Download: https://visualstudio.microsoft.com/downloads/
-- Choose: Visual Studio 2022 (Community/Professional/Enterprise) or Build Tools for Visual Studio 2022.
- During installation check: **Desktop development with C++** workload.
- Optional but recommended components:
  - MSVC v143 (or latest) toolset
  - Windows 10/11 SDK
  - CMake tools for Windows

Notes:
- Using Build Tools 2026 is fine — CUDA toolkits support current MSVC toolsets. If you encounter an error, try the 2022 toolset.

2) Install NVIDIA CUDA Toolkit (includes `nvcc`, `cuBLAS`)

- Download: https://developer.nvidia.com/cuda-downloads
-- Choose: OS = Windows, Architecture = x86_64, Installer Type = exe (local)
-- Recommended CUDA Toolkit version for GTX 1050 Ti (Pascal / sm_61): **CUDA 12.9**. Install CUDA 12.9 to ensure offline compile support for `sm_61`.
- Run the installer as Administrator. Recommended options:
  - Install CUDA Toolkit (required)
  - Install cuBLAS (required)
  - You can skip the NVIDIA driver step if your driver is current
- After install, restart your machine if prompted.

Verify CUDA installation in PowerShell:

```powershell
nvcc --version
nvidia-smi
```

If `nvcc` is not found, add to PATH:

```powershell
# $PS> setx PATH "$Env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
```

Replace `v12.0` with the installed CUDA version directory (for this project prefer `v12.9`).

3) Install GNU Make (if you prefer `Makefile`)

Option A — Chocolatey (recommended if you have Chocolatey):

```powershell
choco install make
```

Option B — winget:

```powershell
winget install GnuWin32.Make
```

Option C — Manual (less recommended):
- Download GNU Make for Windows from GnuWin32 and add `bin` to PATH.

Verify:

```powershell
make --version
```

4) (Optional) Install CMake (recommended for cross-platform builds)

Download: https://cmake.org/download/

Verify:

```powershell
cmake --version
```

5) Build the project (Make or CMake)

Using Make (requires `make` and `nvcc` on PATH):

```powershell
cd c:\personal\CUDA\CUDA_Matrix_Multiplication
make all
make test
make bench
```

Using CMake (MSVC generator):

```powershell
cd c:\personal\CUDA\CUDA_Matrix_Multiplication
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"   # use 2026 generator name if available
cmake --build . --config Release
```

6) Troubleshooting

- `nvcc: command not found` — CUDA Toolkit not added to PATH. Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\bin` to PATH.
- `cl.exe: command not found` — Visual Build Tools not installed or Developer Command Prompt not used. Ensure MSVC installed and PATH includes MSVC `bin` (re-open PowerShell after install).
- `cublas_v2.h: No such file or directory` — Reinstall CUDA Toolkit, ensure cuBLAS selected.
- Driver mismatches/crashes — update NVIDIA GPU driver from NVIDIA website.
- Make problems — use CMake + MSVC if `make` is unavailable.

7) Verifying a successful build and run

After building, run the tests and benchmark executables (paths depend on build system):

```powershell
# From the project root after a Make build
./bin/test_correctness.exe
./bin/matmul_bench.exe
```

8) Notes about WSL2

- CUDA on WSL requires NVIDIA drivers for WSL and appropriate CUDA WSL packages. See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- If you plan to compile in WSL, prefer the Linux instructions for CUDA and use the `Makefile` directly.

9) Where to get help

- NVIDIA Developer Forums: https://forums.developer.nvidia.com/
- Microsoft Visual Studio docs for MSVC
- Your local IT if corporate restrictions prevent installs

Commit these files to the repository so contributors have exact steps to prepare their machine.

## Quick validation: make sure `cl.exe`, `nvcc`, and `make` are available

If you installed Visual Studio Build Tools but `cl.exe` is not found in a normal PowerShell session, run the MSVC environment setup script (`vcvars64.bat`) and then re-check. Use the following PowerShell commands to locate your Visual Studio installation and run the environment script:

```powershell
# locate Visual Studio installation that has the MSVC toolset
& 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

# if the above prints a path, store it and run vcvars64.bat (adjust path if different)
$vsPath = (& 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
& "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

# After the above, verify the compiler is on PATH
cl.exe
where cl.exe

# verify CUDA toolchain and make
nvcc --version
where nvcc
make --version
where make
```

Notes:
- If `vswhere.exe` is not present, check `C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe` manually — it is normally installed with Visual Studio or the Build Tools installer.
- If `cl.exe` works only after running `vcvars64.bat`, always run builds from a Developer Command Prompt or include the `vcvars` call in CI/build scripts so the MSVC environment variables are set.

Important note about PowerShell vs cmd.exe
-----------------------------------------
- Calling `vcvars64.bat` directly from PowerShell using the `&` operator does not always propagate MSVC environment variables into the PowerShell session in the same way as the Visual Studio Developer Command Prompt. To ensure the environment is set for subsequent build commands, either:

- Open **Developer Command Prompt for VS 2022** or **Developer PowerShell for VS 2022** from the Start Menu and run your build there, or
- Use `cmd.exe` to call the batch script and run your build in the same `cmd` process.

Example (run in a normal PowerShell window):

```powershell
REM Replace the VS path with your installed VS 2022 path if different
cmd.exe /c '"C:\Program Files (x86)\Microsoft Visual Studio\17\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && where cl.exe && cl.exe && "C:\Program Files (x86)\GnuWin32\bin\make.exe" --version'
```

Or open the Developer Command Prompt and run:

```cmd
"C:\>" cd c:\personal\CUDA\CUDA_Matrix_Multiplication
"C:\>" make all
```

If you prefer PowerShell, launch **Developer PowerShell for VS 2026** from Visual Studio (it runs the setup script and configures the environment correctly for PowerShell sessions).

Convenience scripts
-------------------
To simplify validation and building on Windows `cmd.exe`, the repository includes helper scripts at the project root:

- `validate_env.cmd` — Runs `vcvars64.bat` (if found), checks for `cl.exe`, `nvcc`, and `make`. Usage: `validate_env.cmd` or `validate_env.cmd make` to also run `make all`.
- `build_with_make.cmd` — Sets the MSVC environment and runs `make all`.
- `build_with_cmake.cmd` — Sets the MSVC environment and runs CMake configure + build (default Release).
- `use_cuda12.cmd` — Auto-detects CUDA 12.x, sets `CUDA_BIN`, validates toolchain, and runs build actions for Pascal GPUs like GTX 1050 Ti. Usage: `use_cuda12.cmd validate|all|test|bench`.

Example usage from Command Prompt (recommended):

```cmd
C:\> cd C:\personal\CUDA\CUDA_Matrix_Multiplication
C:\personal\CUDA\CUDA_Matrix_Multiplication> validate_env.cmd
C:\personal\CUDA\CUDA_Matrix_Multiplication> use_cuda12.cmd validate
C:\personal\CUDA\CUDA_Matrix_Multiplication> use_cuda12.cmd all
C:\personal\CUDA\CUDA_Matrix_Multiplication> build_with_make.cmd
C:\personal\CUDA\CUDA_Matrix_Multiplication> build_with_cmake.cmd Release
```

These scripts are provided so contributors can quickly validate and build without manually starting a Developer Command Prompt.

## Nsight / Visual Studio integration note

The CUDA installer displays a message about Nsight "Not Installed" when it cannot find a Visual Studio IDE to integrate with. This is informational only. If you only need to compile and run CUDA code, the MSVC Build Tools are sufficient and you can ignore the Nsight message. If you want full Nsight integration inside Visual Studio, install the full Visual Studio IDE (Community/Professional/Enterprise) and re-run the CUDA installer or install Nsight from the Visual Studio Marketplace.
# Windows Setup: CUDA Toolkit, Build Tools, and Make

This document provides complete, reproducible setup steps for building this project on Windows.

## Required Tools

1. NVIDIA GPU driver (latest stable)
2. Microsoft Visual Studio 2022 Build Tools with C++ workload
3. NVIDIA CUDA Toolkit (12.x recommended)
4. GNU Make (or use CMake build flow)
5. Python 3.9+ (for plotting scripts)

## 1) Install Visual Studio C++ Build Tools

CUDA on Windows requires MSVC.

1. Download installer:
- https://visualstudio.microsoft.com/downloads/
- Under Tools for Visual Studio 2022, choose Build Tools for Visual Studio 2022

2. In installer, select workload:
- Desktop development with C++

3. Ensure these components are included:
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows (optional but recommended)

4. Install and restart Windows.

## 2) Install CUDA Toolkit

1. Download from:
- https://developer.nvidia.com/cuda-downloads

2. Choose:
- OS: Windows
- Architecture: x86_64
- Version: Windows 10/11
- Installer Type: exe (local)

3. During install:
- Keep CUDA Toolkit and cuBLAS selected
- If your display driver is already up to date, you can skip driver installation

4. Restart Windows after installation.

## 3) Install GNU Make

Choose one option.

### Option A: winget

```powershell
winget install GnuWin32.Make
```

### Option B: Chocolatey

```powershell
choco install make
```

### Option C: Manual

1. Install via GnuWin32 package.
2. Add make binary path to PATH.

## 4) Verify Installation

Open a new PowerShell terminal and run:

```powershell
nvcc --version
make --version
```

Check MSVC compiler from a Developer Command Prompt, or ensure VS tools are on PATH:

```powershell
cl.exe
```

If `cl.exe` is not recognized in regular PowerShell, launch builds from one of these:
- x64 Native Tools Command Prompt for VS 2022
- Developer PowerShell for VS 2022

## 5) Build This Project

From repository root:

```powershell
cd c:\personal\CUDA\CUDA_Matrix_Multiplication
make all
make test
make bench
```

If you prefer CMake:

```powershell
cd c:\personal\CUDA\CUDA_Matrix_Multiplication
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

## 6) Generate Plots

```powershell
cd c:\personal\CUDA\CUDA_Matrix_Multiplication\python
python benchmark_plot.py
python roofline.py
```

Output files are written under `results/`.

## Troubleshooting

### `nvcc` not found

- Reopen terminal after install.
- Verify CUDA bin path exists, e.g.:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
- Add CUDA bin to PATH and reopen terminal.

### `make` not found

- Reopen terminal.
- Confirm make installation path is in PATH.
- Use CMake workflow if make is unavailable.

### `cl.exe` not found

- Install VS Build Tools C++ workload.
- Use Developer PowerShell/Prompt for VS 2022.

### Link or include errors for cuBLAS

- Ensure CUDA Toolkit installation included cuBLAS.
- Verify CUDA install is intact and environment variables are refreshed.

## Notes for Contributors

- These steps are required for local Windows builds.
- Please include your tool versions when reporting issues:

```powershell
nvcc --version
make --version
python --version
```