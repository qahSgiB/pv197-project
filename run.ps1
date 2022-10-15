param (
    [Alias("b")][switch]$build,
    [Alias("r")][switch]$run,
    [Alias("br")][switch]$all,
    [Alias("f")][ValidateSet("14", "17", "orig")][string]$frameworkVersion = "17",
    [Alias("a")][string]$runArgs
)

$cppStd = $null
$frameworkMainFile = "framework.cu"

if ($frameworkVersion -eq "14") {
    $frameworkMainFile = "framework_14.cu"
    $cppStd = "c++14"
} elseif ($frameworkVersion -eq "17") {
    $frameworkMainFile = "framework_17.cu"
    $cppStd = "c++17"
}

$buildDir = "build"
$exePath = Join-Path $buildDir "framework.exe"

if ($build -or $all) {
    if ($null -eq $cppStd) {
        nvcc -O3 -use_fast_math -o $exePath $frameworkMainFile
    } else {
        nvcc -std="$cppStd" -O3 -use_fast_math -o $exePath $frameworkMainFile
    }
}

if ($run -or $all) {
    Invoke-Expression "$exePath $runArgs" # [todo]
}