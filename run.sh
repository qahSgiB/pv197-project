args=$(getopt -a -n run -o br --long build,run,all -- "$@")
args_valid=$?
if [ $args_valid != 0 ]; then
	exit 1
fi

set -- $args # [todo]

build=0
run=0

while true; do
	case "$1" in
		-b | --build) build=1; shift ;;
		-r | --run)   run=1;   shift ;;
		--all) build=1; run=1; shift ;;
		--) shift; break ;;
		*) echo "dafuk $1"; exit 1 ;;
	esac
done

# [todo] check if some arguments remains -> error

build_dir="build"
bin_path="${build_dir%%/}/framework"

mkdir -p "$build_dir"

if [ $build = 1 ]; then
	nvcc -O3 -use_fast_math -o "$bin_path" ./framework.cu
fi

if [ $run = 1 ]; then
	$bin_path
fi
