#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#include <vector>
#include <string_view>
#include <charconv>
#include <stdexcept>
#include <system_error>
#include <optional>

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
    std::string_view msg;

public:
    explicit cuda_exception(std::string_view msg_) : msg(msg_) {}

    const char* what() const noexcept override
    {
        return msg.data();
    }
};


template<typename T>
class cuda_deleter
{
    void operator()(T* obj) const noexcept
    {
        cudaFree(obj);
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
std::optional<T> str_to_num(std::string_view s)
{
    T value;
    const char* s_last = s.data() + s.size();
    auto [ptr, err] = std::from_chars(s.data(), s_last, value);

    if (ptr != s_last || err != std::errc()) {
        return std::nullopt;
    }

    return std::optional<T>(value);
}


int main_exc(int argc, char** argv)
{
    std::cout << "using framework 17\n\n";

    // params
    int device = 0;

    bool run_cpu = true;
    bool run_gpu = true;

    bool seed_time = false;
    bool seed_number = false;
    unsigned int seed = 314159;

    size_t stars_count = 2000;

    // parse command line params
    std::vector<std::string_view> args(argv + 1, argv + argc);

    bool old_args_style = false;
    if (args.size() == 1) {
        // for compatibility with original framework
        std::string_view device_str = args[0];

        unsigned int device_try = 0;
        const char* device_str_last = device_str.data() + device_str.size();
        auto [ptr, err] = std::from_chars(device_str.data(), device_str_last, device_try);

        if (ptr == device_str_last && err == std::errc()) {
            device = device_try;
            old_args_style = true;
        }
    }

    if (!old_args_style) {
        auto args_it = args.begin();

        while (args_it != args.end()) {
            std::string_view arg = *args_it;
            args_it++;

            if (arg == "-c" || arg == "--cpu") {
                run_cpu = true;
            } else if (arg == "-g" || arg == "--gpu") {
                run_gpu = true;
            } else if (arg == "-nc" || arg == "--no-cpu") {
                run_cpu = false;
            } else if (arg == "-ng" || arg == "--no-gpu") {
                run_gpu = false;
            } else if (arg == "-t" || arg == "--seed-time") {
                seed_time = true;
            } else if (arg == "-n" || arg == "--seed-number") {
                if (args_it == args.end()) {
                    throw std::invalid_argument("-n / --seed-number > argument expected");
                }

                seed_number = true;
                std::optional<unsigned int> seed_try = str_to_num<unsigned int>(*args_it);
                args_it++;

                if (!seed_try.has_value()) {
                    throw std::invalid_argument("-n / --seed-number > unsigned int expected");
                }

                seed = seed_try.value();
            } else if (arg == "-d" || arg == "--device") {
                if (args_it == args.end()) {
                    throw std::invalid_argument("-d / --device > argument expected");
                }
                
                std::optional<int> device_try = str_to_num<int>(*args_it);
                args_it++;

                if (!device_try.has_value()) {
                    throw std::invalid_argument("-d / --device > int expected");
                }

                device = device_try.value();
            } else if (arg == "-s" || arg == "--stars") {
                if (args_it == args.end()) {
                    throw std::invalid_argument("-s / --stars > argument expected");
                }
                
                std::optional<long long> stars_count_try = str_to_num<long long>(*args_it);
                args_it++;

                if (!stars_count_try.has_value()) {
                    throw std::invalid_argument("-s / --stars > long long expected");
                }

                stars_count = stars_count_try.value();
            } else {
                throw std::invalid_argument("unknown argument");
            }
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

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    std::cout << "        [using device]\n";
    std::cout << "index : " << device << "\n";
    std::cout << "name : " << device_prop.name << "\n";
    std::cout << "\n";

    // create events for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
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
    float diff_cpu;
    float diff_gpu;

    if (run_cpu) {
        std::cout << "        [CPU solve]\n";

        cudaEventRecord(start, 0);
        diff_cpu = solveCPU(A, B, stars_count);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "CPU performance: " << static_cast<float>(stars_count) * static_cast<float>(stars_count - 1) / (2.0f * time * 1e3f) << " megapairs/s\n";
        std::cout << "CPU result: " << diff_cpu << "\n";
        std::cout << "\n";
    }

    if (run_gpu) {
        std::cout << "        [GPU solve]\n";

        cudaEventRecord(start, 0);

        // run it 10x for more accurately timing results
        for (int i = 0; i < 10; i++) {
            diff_gpu = solveGPU(dA, dB, stars_count);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "GPU performance: " << static_cast<float>(stars_count) * static_cast<float>(stars_count - 1) / (2.0f * time * 1e2f) << " megapairs/s\n";
        std::cout << "GPU result: " << diff_gpu << "\n";
        std::cout << "\n";
    }

    if (run_cpu && run_gpu) {
        std::cout << "        [result check]\n";
        std::cout << "CPU result: " << diff_cpu << "\n";
        std::cout << "GPU result: " << diff_gpu << "\n";

        // check GPU results
        if (std::abs((diff_cpu - diff_gpu) / ((diff_cpu + diff_gpu) / 2.0f)) < 0.01f) { // ???
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
