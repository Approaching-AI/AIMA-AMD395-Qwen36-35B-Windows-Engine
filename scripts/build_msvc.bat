@echo off
setlocal enabledelayedexpansion

if not exist build mkdir build

cl /nologo /std:c11 /W4 /O2 /I native\src native\src\qrt.c native\src\qwen36_baseline.c native\src\main.c /Fe:build\qrt-c-smoke.exe
if errorlevel 1 exit /b 1

build\qrt-c-smoke.exe
if errorlevel 1 exit /b 1

cargo test --workspace
if errorlevel 1 exit /b 1

cargo run -p qrt-cli -- self-test
if errorlevel 1 exit /b 1

cargo run -p qrt-cli -- slice-self-test
if errorlevel 1 exit /b 1

cargo run -p qrt-cli -- micro-self-test
if errorlevel 1 exit /b 1

echo msvc_verify=pass
