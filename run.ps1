param (
    [Alias("b")][switch]$build,
    [Alias("r")][switch]$run,
    [Alias("br")][switch]$all,
    [Alias("f")][ValidateSet("14", "17", "orig")][string]$frameworkVersion = "17",
    [Alias("a")][string]$runArgs
)

$cppStd = $null
$useOriginalFramework = $true;

if ($frameworkVersion -eq "14") {
    $cppStd = "14"
    $useOriginalFramework = $false;
} elseif ($frameworkVersion -eq "17") {
    $cppStd = "17"
    $useOriginalFramework = $false;
}

$buildDir = "build"
$exePath = Join-Path $buildDir "framework.exe"

if ($build -or $all) {
    if ($useOriginalFramework) {
        nvcc -O3 -use_fast_math -o $exePath "framework.cu"
    } else {
        nvcc -std="c++$cppStd" -D "FSTD=$cppStd" -I "./a/$cppStd" -O3 -use_fast_math -o $exePath "framework_plus.cu"
    }
}

if ($run -or $all) {
    Invoke-Expression "$exePath $runArgs" # [todo]
}