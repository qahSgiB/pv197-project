#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#include <vector>
#include <string>
#include <exception>

#include <cuda_runtime.h>

#if (!defined(FSTD))
#define FSTD 14
#endif

#include "argparser.cpp"



class cuda_exception : public std::exception
{
    std::string msg; // [todo] is this needed?

public:
    explicit cuda_exception(std::string msg_) : msg(std::move(msg_)) {}

    const char* what() const noexcept override
    {
        return msg.c_str();
    }
};



struct sGalaxy
{
    float* x;
    float* y;
    float* z;
};



#include "kernel.cu"
#include "kernel_CPU.C"



void generateGalaxies(sGalaxy A, sGalaxy B, int n, bool test = false) {
    for (int i = 0; i < n; i++) {
        A.x[i] = 1000.0f * (float) rand() / (float) RAND_MAX;
        A.y[i] = 1000.0f * (float) rand() / (float) RAND_MAX;
        A.z[i] = 1000.0f * (float) rand() / (float) RAND_MAX;
        
        if (test) {
            B.x[i] = A.x[i];
            B.y[i] = A.y[i];
            B.z[i] = A.z[i];
        } else {
            float mult = (float) rand() / (float) RAND_MAX < 0.01f ? 10.0f : 1.0f;
            B.x[i] = A.x[i] + mult * (float) rand() / (float) RAND_MAX;
            B.y[i] = A.y[i] + mult * (float) rand() / (float) RAND_MAX;
            B.z[i] = A.z[i] + mult * (float) rand() / (float) RAND_MAX;
        }
    }

    if (test) {
        B.x[n - 1] += 1000.0f;
    }
}



enum class gpu_output_type : size_t { DISABLE, ONCE, ENABLE };


int main_exc(int argc, char** argv)
{
    std::cout << "FSTD : " << FSTD << "\n\n";

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

    gpu_output_type gpu_output = gpu_output_type::ONCE;

    // parse command line params
    std::vector<std::string> gpu_output_enum_str{"disable", "once", "enable"};

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
            if (ap.load_arg_string_enum(gpu_output, "-go", "--gpu-output", gpu_output_enum_str.cbegin(), gpu_output_enum_str.cend(), true, "disable, once, enable")) { continue; }

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

    if (run_gpu && gpu_repeat > 0) {
        std::cout << "        [GPU solve]\n";

        cudaEventRecord(start, 0);

        if (gpu_output == gpu_output_type::ONCE) {
            diff_gpu = solve_gpu_param(dA, dB, stars_count, grid_dim_x, grid_dim_y, block_dim_x, block_dim_y, true);
            gpu_repeat -= 1;
        }

        bool gpu_output_b = gpu_output == gpu_output_type::ENABLE;

        for (size_t i = 0; i < gpu_repeat; i++) {
            diff_gpu = solve_gpu_param(dA, dB, stars_count, grid_dim_x, grid_dim_y, block_dim_x, block_dim_y, gpu_output_b);
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

    return EXIT_SUCCESS;
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
