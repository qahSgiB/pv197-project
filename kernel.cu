#include <iostream>



__device__ float total_diff = 0.0f;



template<unsigned int block_size, bool force_fsqrt = false>
__global__ void kernel_main_simple_testing(sGalaxy galaxy_a, sGalaxy galaxy_b, int n)
{
    //  ----------  setup  ----------
    float nf = n;

    float k_total_diff = 0.0f;

    __shared__ float shared_galaxy_a_i_x[block_size];
    __shared__ float shared_galaxy_a_i_y[block_size];
    __shared__ float shared_galaxy_a_i_z[block_size];
    __shared__ float shared_galaxy_b_i_x[block_size];
    __shared__ float shared_galaxy_b_i_y[block_size];
    __shared__ float shared_galaxy_b_i_z[block_size];

    //  ----------  computing  ----------
    for (int start_x = blockIdx.x * block_size; start_x < n - 1; start_x += gridDim.x * block_size) {
        shared_galaxy_a_i_x[threadIdx.x] = galaxy_a.x[start_x + threadIdx.x];
        shared_galaxy_a_i_y[threadIdx.x] = galaxy_a.y[start_x + threadIdx.x];
        shared_galaxy_a_i_z[threadIdx.x] = galaxy_a.z[start_x + threadIdx.x];
        shared_galaxy_b_i_x[threadIdx.x] = galaxy_b.x[start_x + threadIdx.x];
        shared_galaxy_b_i_y[threadIdx.x] = galaxy_b.y[start_x + threadIdx.x];
        shared_galaxy_b_i_z[threadIdx.x] = galaxy_b.z[start_x + threadIdx.x];
        __syncthreads();

        for (int start_y = n - 1 - blockIdx.y * block_size; start_y > start_x; start_y -= gridDim.y * block_size) {
            int rev_start_y = start_y - block_size + 1;

            if (rev_start_y + ((int) threadIdx.x) < 0 || rev_start_y + threadIdx.x >= n) {
                continue;
            }

            float galaxy_a_j_x = galaxy_a.x[rev_start_y + threadIdx.x];
            float galaxy_a_j_y = galaxy_a.y[rev_start_y + threadIdx.x];
            float galaxy_a_j_z = galaxy_a.z[rev_start_y + threadIdx.x];
            float galaxy_b_j_x = galaxy_b.x[rev_start_y + threadIdx.x];
            float galaxy_b_j_y = galaxy_b.y[rev_start_y + threadIdx.x];
            float galaxy_b_j_z = galaxy_b.z[rev_start_y + threadIdx.x];

            int max_k = min(rev_start_y + threadIdx.x - start_x, block_size);
            for (int k = 0; k < max_k; k++) {
                float dx_a = galaxy_a_j_x - shared_galaxy_a_i_x[k];
                float dy_a = galaxy_a_j_y - shared_galaxy_a_i_y[k];
                float dz_a = galaxy_a_j_z - shared_galaxy_a_i_z[k];

                float dx_b = galaxy_b_j_x - shared_galaxy_b_i_x[k];
                float dy_b = galaxy_b_j_y - shared_galaxy_b_i_y[k];
                float dz_b = galaxy_b_j_z - shared_galaxy_b_i_z[k];

                float diff;
                if (force_fsqrt) {
                    diff = __fsqrt_rd(dx_a * dx_a + dy_a * dy_a + dz_a * dz_a) - __fsqrt_rd(dx_b * dx_b + dy_b * dy_b + dz_b * dz_b);
                } else {
                    diff = sqrtf(dx_a * dx_a + dy_a * dy_a + dz_a * dz_a) - sqrtf(dx_b * dx_b + dy_b * dy_b + dz_b * dz_b);
                }
                k_total_diff += diff * diff;
            }
        }
        __syncthreads();
    }

    // if (b_x == 0 && count == 0) {
    //     printf("[%d %d] [%d %d] prrrrrrrrrrrrr count = %d\n", g_x, g_y, b_x, 0, count);
    // }
    // printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, 0, k_total_diff);

    //  ----------  summing  ----------
    // if (threadIdx.x == 0) {
        atomicAdd(&total_diff, k_total_diff / nf);
    // }

    // __shared__ float block_sum[block_size];

    // block_sum[threadIdx.x] = k_total_diff;
    // __syncthreads();
    
    // float sumator3000 = k_total_diff;

    // if (block_size >= 1024) {
    //     if (threadIdx.x < 512) {
    //         block_sum[threadIdx.x] = sumator3000 = sumator3000 + block_sum[threadIdx.x + 512];
    //     };
    //     __syncthreads();
    // }
    // if (block_size >= 512) {
    //     if (threadIdx.x < 256) {
    //         block_sum[threadIdx.x] = sumator3000 = sumator3000 + block_sum[threadIdx.x + 256];
    //     };
    //     __syncthreads();
    // }
    // if (block_size >= 256) {
    //     if (threadIdx.x < 128) {
    //         block_sum[threadIdx.x] = sumator3000 = sumator3000 + block_sum[threadIdx.x + 128];
    //     };
    //     __syncthreads();
    // }
    // if (block_size >= 128) {
    //     if (threadIdx.x < 64) {
    //         block_sum[threadIdx.x] = sumator3000 = sumator3000 + block_sum[threadIdx.x + 64];
    //     };
    //     __syncthreads();
    // }

    // if (threadIdx.x < 32) {
    //     sumator3000 += block_sum[threadIdx.x + 32];
    //     // for (unsigned int offset = 16; offset > 0; offset >>= 1) {
    //     //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, offset);
    //     // }
    //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, 16);
    //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, 8);
    //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, 4);
    //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, 2);
    //     sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, 1);

    //     if (threadIdx.x == 0) {
    //         atomicAdd(&total_diff, sumator3000 / nf);
    //         // grid_sum[g_index] = sumator3000 / nf;
    //         // __threadfence();
    //     }
    // }

    /*__syncthreads();

    if (g_index == 0) {
        sumator3000 = 0.0f;

        unsigned int step = block_size * 2;
        for (int grid_sum_index = b_x; grid_sum_index < gs_x * gs_y; grid_sum_index += step) {
            block_sum[b_x] = sumator3000 = sumator3000 + grid_sum[grid_sum_index] + grid_sum[grid_sum_index + block_size];
        }
        __syncthreads();

        if (b_x < 512) { block_sum[b_x] = sumator3000 = sumator3000 + block_sum[b_x + 512]; }; __syncthreads();
        if (b_x < 256) { block_sum[b_x] = sumator3000 = sumator3000 + block_sum[b_x + 256]; }; __syncthreads();
        if (b_x < 128) { block_sum[b_x] = sumator3000 = sumator3000 + block_sum[b_x + 128]; }; __syncthreads();
        if (bx < 64) { block_sum[b_x] = sumator3000 = sumator3000 + block_sum[b_x + 64]; }; __syncthreads();

        if (b_x < 32) {
            sumator3000 += block_sum[b_x + 32];
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, offset);
            }

            if (b_x == 0) {
                atomicAdd(&total_diff, sumator3000);
            }
        }
    }*/

    // printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, b_y, k_total_diff);
}



float solve_gpu_param(sGalaxy A, sGalaxy B, int n, size_t grid_dim_x, size_t grid_dim_y, size_t block_dim_x, size_t block_dim_y, bool enable_output)
{
    size_t total_dim_x = grid_dim_x * block_dim_x;
    size_t total_dim_y = grid_dim_y * block_dim_y;

    size_t k_x = (n - 2) / total_dim_x + 1; // round up
    size_t k_y = (n - 2) / total_dim_y + 1;

    if (enable_output) {
        std::cout << "    [kernel params]\n";
        std::cout << "grid size : " << grid_dim_x << " x " << grid_dim_y << "\n";
        std::cout << "block size : " << block_dim_x << " x " << block_dim_y << "\n";
        std::cout << "k : " << k_x << " x " << k_y << "\n";
        std::cout << "total size : " << total_dim_x * k_x << " x " << total_dim_y * k_y << "\n";
    }

    float diff = 0.0f;
    cudaMemcpyToSymbol(total_diff, &diff, sizeof(float));

    dim3 grid_size(grid_dim_x, grid_dim_y);
    dim3 block_size(block_dim_x, block_dim_y);

    size_t block_size_total = block_dim_x * block_dim_y;
    if (block_size_total == 1024) {
        kernel_main_simple_testing<1024><<<grid_size, block_size>>>(A, B, n);
    } else if (block_size_total == 512) {
        kernel_main_simple_testing<512><<<grid_size, block_size>>>(A, B, n);
    } else if (block_size_total == 256) {
        kernel_main_simple_testing<256><<<grid_size, block_size>>>(A, B, n);
    } else if (block_size_total == 128) {
        kernel_main_simple_testing<128><<<grid_size, block_size>>>(A, B, n);
    } else if (block_size_total == 64) {
        kernel_main_simple_testing<64><<<grid_size, block_size>>>(A, B, n);
    } else {
        std::cerr << "unsupported block size" << '\n';
        return 0.0f;
    }

    cudaError_t error_code = cudaMemcpyFromSymbol(&diff, total_diff, sizeof(float));
    if (error_code != cudaSuccess) {
        std::cerr << "kernel error (cudaMemcpyFromSymbol) : " << cudaGetErrorString(error_code) << '\n';
        return 0.0f;
    }

    // if (enable_output) { std::cout << "teeeeeeeeeeeeeeeeeeeeest " << diff << "\n"; }

    float nf = n;

    diff = std::sqrt(diff * (1 / (nf - 1)));

    if (enable_output) { std::cout << "\n"; }

    return diff;
}


float solveGPU(sGalaxy A, sGalaxy B, int n)
{
    return solve_gpu_param(A, B, n, 256, 32, 64, 1, false);
}
