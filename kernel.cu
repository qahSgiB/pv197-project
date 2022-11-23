//TODO kernel implementation

#include <iostream>



__device__ float total_diff = 0.0f;



// printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, b_y, k_total_diff);

__global__ void kernel_main_simple(sGalaxy galaxy_a, sGalaxy galaxy_b, int n)
{
    // extern __shared__ float shared_dynamic[];
    // __shared__ float shared_dynamic[6 * 32];

    int gs_x = gridDim.x;
    int gs_y = gridDim.y;
    int bs_x = blockDim.x;
    int bs_y = blockDim.y;
    int ts_x = gs_x * bs_x;
    int ts_y = gs_y * bs_y;

    int g_x = blockIdx.x;
    int g_y = blockIdx.y;
    int b_x = threadIdx.x;
    int b_y = threadIdx.y;
    int t_x = g_x * bs_x + b_x;
    int t_y = g_y * bs_y + b_y;

    // int ks_x = (n - 2) / ts_x + 1; // [opt?] param
    // int ks_y = (n - 2) / ts_y + 1;

    float nf = n;

    float k_total_diff = 0.0f;

    int b_x_start_0 = g_x * bs_x;
    int b_y_start_0 = g_y * bs_y;

    int count = n - 1;

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
    __shared__ float shared_galaxy[12][32];
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
            if (b_y == 0) {
                shared_galaxy[0][b_x] = galaxy_a.x[galaxy0_index];
            } else if (b_y == 1) {
                shared_galaxy[1][b_x] = galaxy_a.y[galaxy0_index];
            } else if (b_y == 2) {
                shared_galaxy[2][b_x] = galaxy_a.z[galaxy0_index];
            } else if (b_y == 3) {
                shared_galaxy[3][b_x] = galaxy_b.x[galaxy0_index];
            } else if (b_y == 4) {
                shared_galaxy[4][b_x] = galaxy_b.y[galaxy0_index];
            } else if (b_y == 5) {
                shared_galaxy[5][b_x] = galaxy_b.z[galaxy0_index];
            }
            if (b_y == 6) {
                shared_galaxy[6][b_x] = galaxy_a.x[galaxy1_index_x];
            } else if (b_y == 7) {
                shared_galaxy[7][b_x] = galaxy_a.y[galaxy1_index_x];
            } else if (b_y == 8) {
                shared_galaxy[8][b_x] = galaxy_a.z[galaxy1_index_x];
            } else if (b_y == 9) {
                shared_galaxy[9][b_x] = galaxy_b.x[galaxy1_index_x];
            } else if (b_y == 10) {
                shared_galaxy[10][b_x] = galaxy_b.y[galaxy1_index_x];
            } else if (b_y == 11) {
                shared_galaxy[11][b_x] = galaxy_b.z[galaxy1_index_x];
            }

            __syncthreads();

            if (x + y < count) {
                // printf("(%d,%d,%d,%d,%d,%d,%d,%d),", galaxy0_index, galaxy1_index, b_x, b_y, g_x, g_y, b_x_start, b_y_start);
                // float galaxy0a_x = galaxy_a.x[galaxy0_index];
                // float galaxy0a_y = galaxy_a.y[galaxy0_index];
                // float galaxy0a_z = galaxy_a.z[galaxy0_index];
                // float galaxy0b_x = galaxy_b.x[galaxy0_index];
                // float galaxy0b_y = galaxy_b.y[galaxy0_index];
                // float galaxy0b_z = galaxy_b.z[galaxy0_index];

                // float galaxy0a_x = shared_galaxy_a_0_x[b_x];
                // float galaxy0a_y = shared_galaxy_a_0_y[b_x];
                // float galaxy0a_z = shared_galaxy_a_0_z[b_x];
                // float galaxy0b_x = shared_galaxy_b_0_x[b_x];
                // float galaxy0b_y = shared_galaxy_b_0_y[b_x];
                // float galaxy0b_z = shared_galaxy_b_0_z[b_x];

                float galaxy0a_x = shared_galaxy[0][b_x];
                float galaxy0a_y = shared_galaxy[1][b_x];
                float galaxy0a_z = shared_galaxy[2][b_x];
                float galaxy0b_x = shared_galaxy[3][b_x];
                float galaxy0b_y = shared_galaxy[4][b_x];
                float galaxy0b_z = shared_galaxy[5][b_x];

                // float galaxy1a_x = galaxy_a.x[galaxy1_index];
                // float galaxy1a_y = galaxy_a.y[galaxy1_index];
                // float galaxy1a_z = galaxy_a.z[galaxy1_index];
                // float galaxy1b_x = galaxy_b.x[galaxy1_index];
                // float galaxy1b_y = galaxy_b.y[galaxy1_index];
                // float galaxy1b_z = galaxy_b.z[galaxy1_index];

                // float galaxy1a_x = shared_galaxy_a_1_x[bs_y - 1 - b_y];
                // float galaxy1a_y = shared_galaxy_a_1_y[bs_y - 1 - b_y];
                // float galaxy1a_z = shared_galaxy_a_1_z[bs_y - 1 - b_y];
                // float galaxy1b_x = shared_galaxy_b_1_x[bs_y - 1 - b_y];
                // float galaxy1b_y = shared_galaxy_b_1_y[bs_y - 1 - b_y];
                // float galaxy1b_z = shared_galaxy_b_1_z[bs_y - 1 - b_y];
                float galaxy1a_x = shared_galaxy[6][bs_y - 1 - b_y];
                float galaxy1a_y = shared_galaxy[7][bs_y - 1 - b_y];
                float galaxy1a_z = shared_galaxy[8][bs_y - 1 - b_y];
                float galaxy1b_x = shared_galaxy[9][bs_y - 1 - b_y];
                float galaxy1b_y = shared_galaxy[10][bs_y - 1 - b_y];
                float galaxy1b_z = shared_galaxy[11][bs_y - 1 - b_y];

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

            __syncthreads();
        }
    }

    // printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, b_y, k_total_diff);
    k_total_diff /= nf;
    atomicAdd(&total_diff, k_total_diff);
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

    dim3 grid_size(grid_dim_x, grid_dim_y);
    dim3 block_size(block_dim_x, block_dim_y);

    float diff = 0.0f;
    cudaMemcpyToSymbol(total_diff, &diff, sizeof(float));

    kernel_main_simple<<<grid_size, block_size>>>(A, B, n);
    // kernel_main_simple<<<grid_size, block_size, 6 * (block_dim_x + block_dim_y)>>>(A, B, n);

    cudaMemcpyFromSymbol(&diff, total_diff, sizeof(float));

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
