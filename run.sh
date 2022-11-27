args=$(getopt -a -n run -o brf:a: --long build,run,all,framework-version:,run-args: -- "$@")
args_valid=$?
if [ $args_valid != 0 ]; then
	exit 1
fi

eval set -- $args # [todo]

build=0
run=0
framework_version="14"
run_args=""

while true; do
	case "$1" in
		-b | --build) build=1; shift ;;
		-r | --run)   run=1;   shift ;;
		--all) build=1; run=1; shift ;;
		-f | --frameworkversion)
			shift
			if [ "$1" != "14" -a "$1" != "17" -a "$1" != "orig" ]; then
				echo "-f / --framework-version > invalid arguement (valid values : 14, 17, orig)"
				exit 1
			fi
			framework_version="$1"
			shift
			;;
		-a | --run-args)
			shift
			run_args="$1"
			shift
			;;
		--) shift; break ;;
		*) echo "dafuk $1"; exit 1 ;;
	esac
done

# [todo] check if some arguments remains -> error

cpp_std=""
use_original_framework=1

if [ "$framework_version" = "14" ]; then
	cpp_std="14"
	use_original_framework=0
elif [ "$framework_version" = "17" ]; then
	cpp_std="17"
	use_original_framework=0
fi

build_dir="build"
bin_path="${build_dir%%/}/framework"

mkdir -p "$build_dir"

if [ $build = 1 ]; then
	if [ $use_original_framework = 1 ]; then
		nvcc -O3 -use_fast_math -o "$bin_path" "framework.cu"
	else
		nvcc -std="c++$cpp_std" -D "FSTD=$cpp_std" -I "./a" -O3 -use_fast_math -o "$bin_path" "framework_plus.cu"
	fi
fi

if [ $run = 1 ]; then
	if [ -z "$run_args" ]; then
		$bin_path
	else
		$bin_path $run_args
	fi
fi
