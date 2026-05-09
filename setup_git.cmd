@echo off
REM Setup Git configuration and initialize repository

echo Configuring Git user...
"C:\Program Files\Git\bin\git.exe" config --global user.name "CUDA Developer"
"C:\Program Files\Git\bin\git.exe" config --global user.email "developer@example.com"

echo Initializing Git repository...
"C:\Program Files\Git\bin\git.exe" init

echo Adding all files to staging...
"C:\Program Files\Git\bin\git.exe" add -A

echo Creating initial commit...
"C:\Program Files\Git\bin\git.exe" commit -m "Initial commit: CUDA matrix multiplication optimization study

- 4 kernel variants: naive, tiled(16), tiled(32), register-blocked
- Correctness validation: 12/12 tests passing
- Performance: register-blocked achieves 18%% of peak on GTX 1050 Ti
- Measurement: warm-up iterations, median aggregation, CUDA events
- Fixed: memory access bugs, UTF-8 console output, cuBLAS timing
- Toolchain: CUDA 12.9, MSVC 2022, GNU Make 3.81
- MIT License"

echo.
echo Checking Git status...
"C:\Program Files\Git\bin\git.exe" status

echo.
echo ✓ Git repository initialized and ready!
echo.
echo Next steps:
echo 1. Add remote: git remote add origin https://github.com/USERNAME/CUDA_Matrix_Multiplication.git
echo 2. Push: git push -u origin main
