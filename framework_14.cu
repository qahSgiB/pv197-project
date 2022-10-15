#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <system_error>

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

    const char* what() const override
    {
        return msg;
    }
};



#include "kernel.cu"
#include "kernel_CPU.C"



// the size of the gallaxy can be arbitrary changed
#define N 2000 // [todo] use as command line parameter
// #define N 10



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

    // parse command line params
    std::vector<char*> args(argv + 1, argv + argc);

    bool old_args_style = false;
    if (args.size() == 1) {
        // for compatibility with original framework
        char* device_str = args[0];

        char* device_str_last;
        int device_try = std::strtol(device_str, &device_str_last, 10); // ignoring out of range

        if (*device_str_last == '\0') {
            device = device_try;
            old_args_style = true;
        }
    }

    if (!old_args_style) {
        auto args_it = args.begin();

        while (args_it != args.end()) {
            char* arg = *args_it;
            args_it++;

            if (std::strcmp(arg, "-c") == 0 || std::strcmp(arg, "--cpu") == 0) {
                run_cpu = true;
            } else if (std::strcmp(arg, "-g") == 0 || std::strcmp(arg, "--gpu") == 0) {
                run_gpu = true;
            } else if (std::strcmp(arg, "-nc") == 0 || std::strcmp(arg, "--no-cpu") == 0) {
                run_cpu = false;
            } else if (std::strcmp(arg, "-ng") == 0 || std::strcmp(arg, "--no-gpu") == 0) {
                run_gpu = false;
            } else if (std::strcmp(arg, "-t") == 0 || std::strcmp(arg, "--seed-time") == 0) {
                seed_time = true;
            } else if (std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--seed-number") == 0) {
                seed_number = true;

                if (args_it == args.end()) {
                    throw std::invalid_argument("-n / --seed-number > unsigned int expected");
                }
                char* seed_str = *args_it;
                args_it++;

                char* seed_str_last;
                unsigned int seed_try = std::strtol(seed_str, &seed_str_last, 10); // ignoring out of range

                if (*seed_str_last != '\0') {
                    throw std::invalid_argument("-n / --seed-number > unsigned int expected");
                }

                seed = seed_try;
            } else if (std::strcmp(arg, "-d") == 0 || std::strcmp(arg, "--device") == 0) {
                if (args_it == args.end()) {
                    throw std::invalid_argument("-d / --device > int expected");
                }
                char* device_str = *args_it;
                args_it++;

                char* device_str_last;
                int device_try = std::strtol(device_str, &device_str_last, 10); // ignoring out of range

                if (*device_str_last != '\0') {
                    throw std::invalid_argument("-d / --device > int expected");
                }

                device = device_try;
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
    A.x = (float*)malloc(N*sizeof(A.x[0]));
    A.y = (float*)malloc(N*sizeof(A.y[0]));
    A.z = (float*)malloc(N*sizeof(A.z[0]));
    B.x = (float*)malloc(N*sizeof(B.x[0]));
    B.y = (float*)malloc(N*sizeof(B.y[0]));
    B.z = (float*)malloc(N*sizeof(B.z[0]));
    generateGalaxies(A, B, N);      
 
    // allocate and set device memory
    if (cudaMalloc((void**)&dA.x, N*sizeof(dA.x[0])) != cudaSuccess
    || cudaMalloc((void**)&dA.y, N*sizeof(dA.y[0])) != cudaSuccess
    || cudaMalloc((void**)&dA.z, N*sizeof(dA.z[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.x, N*sizeof(dB.x[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.y, N*sizeof(dB.y[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.z, N*sizeof(dB.z[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(dA.x, A.x, N*sizeof(dA.x[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dA.y, A.y, N*sizeof(dA.y[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dA.z, A.z, N*sizeof(dA.z[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.x, B.x, N*sizeof(dB.x[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.y, B.y, N*sizeof(dB.y[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.z, B.z, N*sizeof(dB.z[0]), cudaMemcpyHostToDevice);
    
    float time;
    float diff_CPU, diff_GPU;

    if (run_cpu) {
        std::cout << "        [CPU solve]\n";

        cudaEventRecord(start, 0);
        diff_CPU = solveCPU(A, B, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "CPU performance: " << static_cast<float>(N) * static_cast<float>(N - 1) / (2.0f * time * 1e3f) << " megapairs/s\n";
        std::cout << "CPU result: " << diff_CPU << "\n";
        std::cout << "\n";
    }

    if (run_gpu) {
        std::cout << "        [GPU solve]\n";

        cudaEventRecord(start, 0);

        // run it 10x for more accurately timing results
        for (int i = 0; i < 10; i++) {
            diff_GPU = solveGPU(dA, dB, N);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "    [result]\n";
        std::cout << "GPU performance: " << static_cast<float>(N) * static_cast<float>(N - 1) / (2.0f * time * 1e2f) << " megapairs/s\n";
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