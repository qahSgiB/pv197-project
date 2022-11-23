//TODO kernel implementation

#include <iostream>



__device__ float total_diff = 0.0f;



// printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, b_y, k_total_diff);

__global__ void kernel_main_simple(sGalaxy galaxy_a, sGalaxy galaxy_b, int n, float* grid_sum)
{
    // extern __shared__ float shared_dynamic[];
    // __shared__ float shared_dynamic[6 * 32];

    unsigned int gs_x = gridDim.x;
    unsigned int gs_y = gridDim.y;
    unsigned int bs_x = blockDim.x;
    unsigned int bs_y = blockDim.y;
    unsigned int ts_x = gs_x * bs_x;
    unsigned int ts_y = gs_y * bs_y;

    unsigned int g_x = blockIdx.x;
    unsigned int g_y = blockIdx.y;
    unsigned int b_x = threadIdx.x;
    unsigned int b_y = threadIdx.y;
    unsigned int t_x = g_x * bs_x + b_x;
    unsigned int t_y = g_y * bs_y + b_y;

    // int ks_x = (n - 2) / ts_x + 1; // [opt?] param
    // int ks_y = (n - 2) / ts_y + 1;

    float nf = n;

    float k_total_diff = 0.0f;

    unsigned int b_x_start_0 = g_x * bs_x;
    unsigned int b_y_start_0 = g_y * bs_y;

    unsigned int count = n - 1;

    // float* shared_galaxy_a_0_x = &shared_dynamic[0];                   // size = bs_x
    // float* shared_galaxy_a_0_y = &shared_dynamic[bs_x];                // size = bs_x
    // float* shared_galaxy_a_0_z = &shared_dynamic[2 * bs_x];            // size = bs_x
    // float* shared_galaxy_b_0_x = &shared_dynamic[3 * bs_x];            // size = bs_x
    // float* shared_galaxy_b_0_y = &shared_dynamic[4 * bs_x];            // size = bs_x
    // float* shared_galaxy_b_0_z = &shared_dynamic[5 * bs_x];            // size = bs_x
    // __shared__ float shared_galaxy_a_0_x[32];
    // __shared__ float shared_galaxy_a_0_y[32];
    // __shared__ float shared_galaxy_a_0_z[32];
    // __shared__ float shared_galaxy_b_0_x[32];
    // __shared__ float shared_galaxy_b_0_y[32];
    // __shared__ float shared_galaxy_b_0_z[32];
    // __shared__ float shared_galaxy_a_1_x[32];
    // __shared__ float shared_galaxy_a_1_y[32];
    // __shared__ float shared_galaxy_a_1_z[32];
    // __shared__ float shared_galaxy_b_1_x[32];
    // __shared__ float shared_galaxy_b_1_y[32];
    // __shared__ float shared_galaxy_b_1_z[32];
    // __shared__ float shared_galaxy[12][32];
    // float* shared_galaxy_a_1_x = &shared_dynamic[6 * bs_x];            // size = bs_y
    // float* shared_galaxy_a_1_y = &shared_dynamic[6 * bs_x + bs_y];        // size = bs_y
    // float* shared_galaxy_a_1_z = &shared_dynamic[6 * bs_x];        // size = bs_y
    // float* shared_galaxy_b_1 = &shared_dynamic[2 * bs_x + bs_y]; // size = bs_y

    for (int b_y_start = b_y_start_0; b_y_start + b_x_start_0 < count; b_y_start += ts_y) { // block
        for (int b_x_start = b_x_start_0; b_x_start + b_y_start < count; b_x_start += ts_x) { // block
            int x = b_x_start + b_x;
            int y = b_y_start + b_y;

            int galaxy0_index = x;
            int galaxy1_index = count - y;
            int galaxy1_index_x = count - b_y_start - bs_x + b_x + 1;

            // if (b_y == 0) {
            //     shared_galaxy_a_0_x[b_x] = galaxy_a.x[galaxy0_index];
            // } else if (b_y == 1) {
            //     shared_galaxy_a_0_y[b_x] = galaxy_a.y[galaxy0_index];
            // } else if (b_y == 2) {
            //     shared_galaxy_a_0_z[b_x] = galaxy_a.z[galaxy0_index];
            // } else if (b_y == 3) {
            //     shared_galaxy_b_0_x[b_x] = galaxy_b.x[galaxy0_index];
            // } else if (b_y == 4) {
            //     shared_galaxy_b_0_y[b_x] = galaxy_b.y[galaxy0_index];
            // } else if (b_y == 5) {
            //     shared_galaxy_b_0_z[b_x] = galaxy_b.z[galaxy0_index];
            // }
            // if (b_y == 6) {
            //     shared_galaxy_a_1_x[b_x] = galaxy_a.x[galaxy1_index_x];
            // } else if (b_y == 7) {
            //     shared_galaxy_a_1_y[b_x] = galaxy_a.y[galaxy1_index_x];
            // } else if (b_y == 8) {
            //     shared_galaxy_a_1_z[b_x] = galaxy_a.z[galaxy1_index_x];
            // } else if (b_y == 9) {
            //     shared_galaxy_b_1_x[b_x] = galaxy_b.x[galaxy1_index_x];
            // } else if (b_y == 10) {
            //     shared_galaxy_b_1_y[b_x] = galaxy_b.y[galaxy1_index_x];
            // } else if (b_y == 11) {
            //     shared_galaxy_b_1_z[b_x] = galaxy_b.z[galaxy1_index_x];
            // }
            // if (b_y == 0) {
            //     shared_galaxy[0][b_x] = galaxy_a.x[galaxy0_index];
            // } else if (b_y == 1) {
            //     shared_galaxy[1][b_x] = galaxy_a.y[galaxy0_index];
            // } else if (b_y == 2) {
            //     shared_galaxy[2][b_x] = galaxy_a.z[galaxy0_index];
            // } else if (b_y == 3) {
            //     shared_galaxy[3][b_x] = galaxy_b.x[galaxy0_index];
            // } else if (b_y == 4) {
            //     shared_galaxy[4][b_x] = galaxy_b.y[galaxy0_index];
            // } else if (b_y == 5) {
            //     shared_galaxy[5][b_x] = galaxy_b.z[galaxy0_index];
            // }
            // if (b_y == 6) {
            //     shared_galaxy[6][b_x] = galaxy_a.x[galaxy1_index_x];
            // } else if (b_y == 7) {
            //     shared_galaxy[7][b_x] = galaxy_a.y[galaxy1_index_x];
            // } else if (b_y == 8) {
            //     shared_galaxy[8][b_x] = galaxy_a.z[galaxy1_index_x];
            // } else if (b_y == 9) {
            //     shared_galaxy[9][b_x] = galaxy_b.x[galaxy1_index_x];
            // } else if (b_y == 10) {
            //     shared_galaxy[10][b_x] = galaxy_b.y[galaxy1_index_x];
            // } else if (b_y == 11) {
            //     shared_galaxy[11][b_x] = galaxy_b.z[galaxy1_index_x];
            // }

            // __syncthreads();

            if (x + y < count) {
                // printf("(%d,%d,%d,%d,%d,%d,%d,%d),", galaxy0_index, galaxy1_index, b_x, b_y, g_x, g_y, b_x_start, b_y_start);
                float galaxy0a_x = galaxy_a.x[galaxy0_index];
                float galaxy0a_y = galaxy_a.y[galaxy0_index];
                float galaxy0a_z = galaxy_a.z[galaxy0_index];
                float galaxy0b_x = galaxy_b.x[galaxy0_index];
                float galaxy0b_y = galaxy_b.y[galaxy0_index];
                float galaxy0b_z = galaxy_b.z[galaxy0_index];

                // float galaxy0a_x = shared_galaxy_a_0_x[b_x];
                // float galaxy0a_y = shared_galaxy_a_0_y[b_x];
                // float galaxy0a_z = shared_galaxy_a_0_z[b_x];
                // float galaxy0b_x = shared_galaxy_b_0_x[b_x];
                // float galaxy0b_y = shared_galaxy_b_0_y[b_x];
                // float galaxy0b_z = shared_galaxy_b_0_z[b_x];

                // float galaxy0a_x = shared_galaxy[0][b_x];
                // float galaxy0a_y = shared_galaxy[1][b_x];
                // float galaxy0a_z = shared_galaxy[2][b_x];
                // float galaxy0b_x = shared_galaxy[3][b_x];
                // float galaxy0b_y = shared_galaxy[4][b_x];
                // float galaxy0b_z = shared_galaxy[5][b_x];

                float galaxy1a_x = galaxy_a.x[galaxy1_index];
                float galaxy1a_y = galaxy_a.y[galaxy1_index];
                float galaxy1a_z = galaxy_a.z[galaxy1_index];
                float galaxy1b_x = galaxy_b.x[galaxy1_index];
                float galaxy1b_y = galaxy_b.y[galaxy1_index];
                float galaxy1b_z = galaxy_b.z[galaxy1_index];

                // float galaxy1a_x = shared_galaxy_a_1_x[bs_y - 1 - b_y];
                // float galaxy1a_y = shared_galaxy_a_1_y[bs_y - 1 - b_y];
                // float galaxy1a_z = shared_galaxy_a_1_z[bs_y - 1 - b_y];
                // float galaxy1b_x = shared_galaxy_b_1_x[bs_y - 1 - b_y];
                // float galaxy1b_y = shared_galaxy_b_1_y[bs_y - 1 - b_y];
                // float galaxy1b_z = shared_galaxy_b_1_z[bs_y - 1 - b_y];

                // float galaxy1a_x = shared_galaxy[6][bs_y - 1 - b_y];
                // float galaxy1a_y = shared_galaxy[7][bs_y - 1 - b_y];
                // float galaxy1a_z = shared_galaxy[8][bs_y - 1 - b_y];
                // float galaxy1b_x = shared_galaxy[9][bs_y - 1 - b_y];
                // float galaxy1b_y = shared_galaxy[10][bs_y - 1 - b_y];
                // float galaxy1b_z = shared_galaxy[11][bs_y - 1 - b_y];

                // float galaxy1a_x = shared_galaxy_a_1_x[b_y];
                // float galaxy1a_y = shared_galaxy_a_1_y[b_y];
                // float galaxy1a_z = shared_galaxy_a_1_z[b_y];
                // float galaxy1b_x = shared_galaxy_b_1_x[b_y];
                // float galaxy1b_y = shared_galaxy_b_1_y[b_y];
                // float galaxy1b_z = shared_galaxy_b_1_z[b_y];

                float dx = galaxy0a_x - galaxy1a_x;
                float dy = galaxy0a_y - galaxy1a_y;
                float dz = galaxy0a_z - galaxy1a_z;

                float diff_a2 = dx * dx + dy * dy + dz * dz;

                dx = galaxy0b_x - galaxy1b_x;
                dy = galaxy0b_y - galaxy1b_y;
                dz = galaxy0b_z - galaxy1b_z;

                float diff_b2 = dx * dx + dy * dy + dz * dz;

                // k_total_diff -= 2 * sqrtf(diff_a2 * diff_b2);
                // k_total_diff += diff_a2 + diff_b2;
                float diff = sqrtf(diff_a2) - sqrtf(diff_b2);
                k_total_diff += diff * diff;
            }

            // __syncthreads();
        }
    }

    __shared__ float block_sum[16 * 16];

    unsigned int g_index = g_x + g_y * gs_x;
    unsigned int b_index = b_x + b_y * bs_x;
    unsigned int bs = bs_x * bs_y;

    block_sum[b_index] = k_total_diff;
    __syncthreads();
    
    float sumator3000 = k_total_diff;

    // for (unsigned int s = bs_x * bs_y / 2; s > 32; s >>= 1) {
    //     if (b_index < s) {
    //         block_sum[b_index] = sumator3000 = sumator3000 + sum[b_index + s];
    //     }
    //     __syncthreads();
    // }

    // if (b_index < 512) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 512]; }; __syncthreads();
    // if (b_index < 256) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 256]; }; __syncthreads();
    if (b_index < 128) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 128]; }; __syncthreads();
    if (b_index < 64) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 64]; }; __syncthreads();

    if (b_index < 32) {
        sumator3000 += block_sum[b_index + 32];
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, offset);
        }

        // if (b_index == 0) {
        //     atomicAdd(&total_diff, sumator3000 / nf);
        // }
        grid_sum[g_index] = sumator3000 / nf;
        __threadfence();
    }

    __syncthreads();

    if (g_index == 0) {
        sumator3000 = 0.0f;

        unsigned int step = bs * 2;
        for (int grid_sum_index = b_index; grid_sum_index < gs_x * gs_y; grid_sum_index += step) {
            block_sum[b_index] = sumator3000 = sumator3000 + grid_sum[grid_sum_index] + grid_sum[grid_sum_index + bs];
        }
        __syncthreads();

        // if (b_index < 512) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 512]; }; __syncthreads();
        // if (b_index < 256) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 256]; }; __syncthreads();
        if (b_index < 128) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 128]; }; __syncthreads();
        if (b_index < 64) { block_sum[b_index] = sumator3000 = sumator3000 + block_sum[b_index + 64]; }; __syncthreads();

        if (b_index < 32) {
            sumator3000 += block_sum[b_index + 32];
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                sumator3000 += __shfl_down_sync(0xffffffff, sumator3000, offset);
            }

            if (b_index == 0) {
                atomicAdd(&total_diff, sumator3000);
            }
        }
    }

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

    float* grid_sum;
    cudaMalloc(&grid_sum, sizeof(float) * grid_dim_x * grid_dim_y);

    float diff = 0.0f;
    cudaMemcpyToSymbol(total_diff, &diff, sizeof(float));

    dim3 grid_size(grid_dim_x, grid_dim_y);
    dim3 block_size(block_dim_x, block_dim_y);

    kernel_main_simple<<<grid_size, block_size>>>(A, B, n, grid_sum);
    // kernel_main_simple<<<grid_size, block_size, 6 * (block_dim_x + block_dim_y)>>>(A, B, n);

    cudaMemcpyFromSymbol(&diff, total_diff, sizeof(float));

    cudaFree(grid_sum);

    // if (enable_output) { std::cout << "teeeeeeeeeeeeeeeeeeeeest " << diff << "\n"; }

    float nf = n;

    diff = std::sqrt(diff * (1 / (nf - 1)));

    if (enable_output) { std::cout << "\n"; }

    return diff;
}

float solveGPU(sGalaxy A, sGalaxy B, int n)
{
    //TODO kernel call and data manipulation
    return solve_gpu_param(A, B, n, 128, 64, 16, 16, false);
}
