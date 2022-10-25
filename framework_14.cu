#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <system_error>
#include <memory>
#include <limits>

#include <cuda_runtime.h>




// galaxy is stored as cartesian coordinates of its stars, each dimmension is in separate array
struct sGalaxy
{
    float* x;
    float* y;
    float* z;
};


class cuda_exception : public std::exception
{
    const char* msg;

public:
    explicit cuda_exception(const char* msg_) : msg(msg_) {}

    const char* what() const noexcept override
    {
        return msg;
    }
};



#include "kernel.cu"
#include "kernel_CPU.C"



void generateGalaxies(sGalaxy A, sGalaxy B, int n) {
    for (int i = 0; i < n; i++) {
        // create star in A at random position first
        A.x[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
        A.y[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
        A.z[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
        // create star in B near star A
        // in small probability, create more displaced star
        if ((float)rand() / (float)RAND_MAX < 0.01f) {
            B.x[i] = A.x[i] + 10.0f * (float)rand() / (float)RAND_MAX;
            B.y[i] = A.y[i] + 10.0f * (float)rand() / (float)RAND_MAX;
            B.z[i] = A.z[i] + 10.0f * (float)rand() / (float)RAND_MAX;
        }
        else {
            B.x[i] = A.x[i] + 1.0f * (float)rand() / (float)RAND_MAX;
            B.y[i] = A.y[i] + 1.0f * (float)rand() / (float)RAND_MAX;
            B.z[i] = A.z[i] + 1.0f * (float)rand() / (float)RAND_MAX;
        }
    }
}


template<typename T>
std::enable_if_t<std::is_signed<T>::value, bool> str_to_num(const char* s, T& value)
{
    char* s_last;
    long long value_ll = std::strtoll(s, &s_last, 10); // not optimized for smaller types

    if (*s_last != '\0' || value_ll < static_cast<long long>(std::numeric_limits<T>::lowest()) || value_ll > static_cast<long long>(std::numeric_limits<T>::max())) {
        return false;
    }

    value = static_cast<T>(value_ll);
    return true;
}

template<typename T>
std::enable_if_t<std::is_unsigned<T>::value, bool> str_to_num(const char* s, T& value)
{
    char* s_last;
    unsigned long long value_ll = std::strtoull(s, &s_last, 10); // not optimized for smaller types

    if (*s_last != '\0' || value_ll > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        return false;
    }

    value = static_cast<T>(value_ll);
    return true;
}

template<typename ... Args>
std::string format_string(const char* format, Args ... args)
{
    int size = snprintf(nullptr, 0, format, args ...);
    if (size < 0) {
        throw std::runtime_error("formatting error");
    }

    std::unique_ptr<char[]> sc = std::make_unique<char[]>(size + 1);
    int size_n = snprintf(sc.get(), size + 1, format, args ...);
    if (size_n < 0 || size_n > size) {
        throw std::runtime_error("formatting error");
    }

    return std::string(sc.get()); // [todo] unnecessary char[] copying - use something like string + reserve / stringstream
}



class arg_parser
{
    std::vector<char*> args;
    std::vector<char*>::iterator args_it;

    void end_throw(const char* msg) const
    {
        if (end()) {
            throw std::invalid_argument(msg);
        }
    }

    void next()
    {
        args_it++;
    }

    char* get_arg() const
    {
        return *args_it;
    }

    char* get_arg_next()
    {
        return *(args_it++);
    }

    bool parse_arg(const char* short_s, const char* long_s)
    {
        char* arg = get_arg();
        if (strcmp(arg, short_s) != 0 && strcmp(arg, long_s) != 0) {
            return false;
        }

        next();
        return true;
    }

public:
    bool end() const
    {
        return args_it == args.end();
    }

    size_t size() const
    {
        return args.size();
    }

    arg_parser(int argc, char** argv) : args(argv + 1, argv + argc)
    {
        args_it = args.begin();
    }

    bool load_arg_switch(bool& b, const char* short_s, const char* long_s, bool b_value = true)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        b = b_value;
        return true;
    }

    template<typename T>
    bool load_num(T& n)
    {
        end_throw("unexpected end of arguments");

        if (!str_to_num<T>(get_arg(), n)) {
            return false;
        }

        next();
        return true;
    }

    template<typename T>
    bool load_arg_num(T& n, const char* short_s, const char* long_s, const char* type_str = nullptr)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        std::string arg_error_msg = format_string("%s / %s > %s argument expected", short_s, long_s, type_str == nullptr ? "numerical" : type_str);
        end_throw(arg_error_msg.data());

        if (!str_to_num<T>(get_arg_next(), n)) {
            throw std::invalid_argument(arg_error_msg.data());
        }

        return true;
    }

    void throw_unknown_arg() const
    {
        throw std::invalid_argument(format_string("unknown argument - %s", get_arg()));
    }
};



int main_exc(int argc, char** argv)
{
    std::cout << "using framework 14\n\n";

    // params
    int device = 0;

    bool run_cpu = true;
    bool run_gpu = true;

    bool seed_time = false;
    bool seed_number = false;
    unsigned int seed = 314159;

    size_t stars_count = 2000;

    size_t grid_dim_x = 64;
    size_t grid_dim_y = 64;
    size_t block_dim_x = 16;
    size_t block_dim_y = 16;

    size_t gpu_repeat = 10;

    // parse command line params
    arg_parser ap(argc, argv);

    if (ap.size() != 1 || !ap.load_num(device)) { // for compatibility with original framework
        while (!ap.end()) {
            if (ap.load_arg_switch(run_cpu, "-c", "--cpu")) { continue; }
            if (ap.load_arg_switch(run_gpu, "-g", "--gpu")) { continue; }
            if (ap.load_arg_switch(run_cpu, "-nc", "--no-cpu", false)) { continue; }
            if (ap.load_arg_switch(run_gpu, "-ng", "--no-gpu", false)) { continue; }
            if (ap.load_arg_switch(seed_time, "-t", "--seed-time")) { continue; }
            if (ap.load_arg_num(seed, "-n", "--seed-number", "unsigned int")) { seed_number = true; continue; }
            if (ap.load_arg_num(device, "-d", "--device", "int")) { continue; }
            if (ap.load_arg_num(stars_count, "-s", "--stars", "long long")) { continue; }
            if (ap.load_arg_num(grid_dim_x, "-gx", "--grid-dim-x", "size_t")) { continue; }
            if (ap.load_arg_num(grid_dim_y, "-gy", "--grid-dim-y-", "size_t")) { continue; }
            if (ap.load_arg_num(block_dim_x, "-bx", "--block-dim-x", "size_t")) { continue; }
            if (ap.load_arg_num(block_dim_y, "-by", "--block-dim-y", "size_t")) { continue; }
            if (ap.load_arg_num(gpu_repeat, "-gr", "--gpu-repeat", "size_t")) { continue; }

            ap.throw_unknown_arg();
        }
    }

    // setup seed
    if (seed_number) {
        srand(seed);
    } else if (seed_time) {
        srand(time(NULL));
    }

    // setup device
    if (cudaSetDevice(device) != cudaSuccess) {
        throw cuda_exception("cannot set CUDA device");
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "        [using device]\n";
    std::cout << "index : " << device << "\n";
    std::cout << "name : " << deviceProp.name << "\n";
    std::cout << "\n";

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    sGalaxy A, B;
    A.x = A.y = A.z = B.x = B.y = B.z = NULL;
    sGalaxy dA, dB;
    dA.x = dA.y = dA.z = dB.x = dB.y = dB.z = NULL;

    // allocate and set host memory
    A.x = (float*)malloc(stars_count*sizeof(A.x[0]));
    A.y = (float*)malloc(stars_count*sizeof(A.y[0]));
    A.z = (float*)malloc(stars_count*sizeof(A.z[0]));
    B.x = (float*)malloc(stars_count*sizeof(B.x[0]));
    B.y = (float*)malloc(stars_count*sizeof(B.y[0]));
    B.z = (float*)malloc(stars_count*sizeof(B.z[0]));
    generateGalaxies(A, B, stars_count);      
 
    // allocate and set device memory
    if (cudaMalloc((void**)&dA.x, stars_count*sizeof(dA.x[0])) != cudaSuccess
    || cudaMalloc((void**)&dA.y, stars_count*sizeof(dA.y[0])) != cudaSuccess
    || cudaMalloc((void**)&dA.z, stars_count*sizeof(dA.z[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.x, stars_count*sizeof(dB.x[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.y, stars_count*sizeof(dB.y[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.z, stars_count*sizeof(dB.z[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(dA.x, A.x, stars_count*sizeof(dA.x[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dA.y, A.y, stars_count*sizeof(dA.y[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dA.z, A.z, stars_count*sizeof(dA.z[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.x, B.x, stars_count*sizeof(dB.x[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.y, B.y, stars_count*sizeof(dB.y[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.z, B.z, stars_count*sizeof(dB.z[0]), cudaMemcpyHostToDevice);
    
    float time;
    float diff_CPU, diff_GPU;

    if (run_cpu) {
        std::cout << "        [CPU solve]\n";

        cudaEventRecord(start, 0);
        diff_CPU = solveCPU(A, B, stars_count);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "CPU performance: " << static_cast<float>(stars_count) * static_cast<float>(stars_count - 1) / (2.0f * time * 1e3f) << " megapairs/s\n";
        std::cout << "CPU result: " << diff_CPU << "\n";
        std::cout << "\n";
    }

    if (run_gpu) {
        std::cout << "        [GPU solve]\n";

        cudaEventRecord(start, 0);

        // run it 10x for more accurately timing results
        for (size_t i = 0; i < gpu_repeat; i++) {
            diff_GPU = solve_gpu_param(dA, dB, stars_count, grid_dim_x, grid_dim_y, block_dim_x, block_dim_y);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "GPU performance: " << static_cast<float>(stars_count) * static_cast<float>(stars_count - 1) / (2.0f * time * 1e2f) << " megapairs/s\n";
        std::cout << "GPU result: " << diff_GPU << "\n";
        std::cout << "\n";
    }

    if (run_cpu && run_gpu) {
        std::cout << "        [result check]\n";
        std::cout << "CPU result: " << diff_CPU << "\n";
        std::cout << "GPU result: " << diff_GPU << "\n";

        // check GPU results
        if (std::abs((diff_CPU - diff_GPU) / ((diff_CPU + diff_GPU) / 2.0f)) < 0.01f) { // ???
            std::cout << "Test OK :)\n";
        } else {
            std::cout << "Test FAILED :(\n";
        }

        std::cout << "\n";
    }

cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (dA.x) cudaFree(dA.x);
    if (dA.y) cudaFree(dA.y);
    if (dA.z) cudaFree(dA.z);
    if (dB.x) cudaFree(dB.x);
    if (dB.y) cudaFree(dB.y);
    if (dB.z) cudaFree(dB.z);
    if (A.x) free(A.x);
    if (A.y) free(A.y);
    if (A.z) free(A.z);
    if (B.x) free(B.x);
    if (B.y) free(B.y);
    if (B.z) free(B.z);

    return 0;
}


int main(int argc, char** argv)
{
    try {
        main_exc(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "ERROR: unknown exception\n";
    }
}
