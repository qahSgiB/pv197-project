param (
    [Alias("b")][switch]$build,
    [Alias("r")][switch]$run,
    [Alias("br")][switch]$all,
    [Alias("a")][string]$runArgs
)

$buildDir = "build"
$exePath = Join-Path $buildDir "framework.exe"

if ($build -or $all) {
    nvcc -std=c++17 -O3 -use_fast_math -o $exePath .\framework.cu
}

if ($run -or $all) {
    Invoke-Expression "$exePath $runArgs" # [todo]
}