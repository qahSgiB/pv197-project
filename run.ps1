param (
    [Alias("b")][switch]$build,
    [Alias("r")][switch]$run,
    [Alias("br")][switch]$all
)

$buildDir = "build"
$exePath = Join-Path $buildDir "framework.exe"

if ($build -or $all) {
    nvcc -O3 -use_fast_math -o $exePath .\framework.cu
}

if ($run -or $all) {
    .$exePath
}