#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <cmath>



// galaxy is stored as cartesian coordinates of its stars, each dimmension is in separate array
struct sGalaxy {
    float* x;
    float* y;
    float* z;
};



#include "kernel.cu"
#include "kernel_CPU.C"



// the size of the gallaxy can be arbitrary changed
#define N 50000
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


int main(int argc, char **argv)
{
    int device = 0;
    bool run_cpu = true; // [todo] command line options
    bool run_gpu = true;

    // parse command line
    if (argc == 2) {
        device = atoi(argv[1]);
    }

    // setup device
    if (cudaSetDevice(device) != cudaSuccess) {
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

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

