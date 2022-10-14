//TODO kernel implementation

#include <iostream>



__device__ float total_diff = 0.0f;



// printf("[%d %d] [%d %d] k_total_diff = %f\n", g_x, g_y, b_x, b_y, k_total_diff);

__global__ void kernel_main_simple(sGalaxy galaxy_a, sGalaxy galaxy_b, int n) {
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

    int ks_x = (n - 1) / ts_x + 1; // [opt?] param
    int ks_y = (n - 1) / ts_y + 1;

    // float nf = n;

    float k_total_diff = 0.0f;

    for (int k_y = 0; k_y < ks_y; k_y++) {
        for (int k_x = 0; k_x < ks_x; k_x++) {
            int galaxy0_index = k_x * ts_x + t_x;
            int galaxy1_index = k_y * ts_y + t_y;

            if (galaxy0_index < n && galaxy1_index < n && galaxy0_index < galaxy1_index) {
                float galaxy0a_x = galaxy_a.x[galaxy0_index];
                float galaxy0a_y = galaxy_a.y[galaxy0_index];
                float galaxy0a_z = galaxy_a.z[galaxy0_index];

                float galaxy0b_x = galaxy_b.x[galaxy0_index];
                float galaxy0b_y = galaxy_b.y[galaxy0_index];
                float galaxy0b_z = galaxy_b.z[galaxy0_index];

                float galaxy1a_x = galaxy_a.x[galaxy1_index];
                float galaxy1a_y = galaxy_a.y[galaxy1_index];
                float galaxy1a_z = galaxy_a.z[galaxy1_index];

                float galaxy1b_x = galaxy_b.x[galaxy1_index];
                float galaxy1b_y = galaxy_b.y[galaxy1_index];
                float galaxy1b_z = galaxy_b.z[galaxy1_index];

                float dx = galaxy0a_x - galaxy1a_x;
                float dy = galaxy0a_y - galaxy1a_y;
                float dz = galaxy0a_z - galaxy1a_z;

                float diff_a2 = dx * dx + dy * dy + dz * dz;

                dx = galaxy0b_x - galaxy1b_x;
                dy = galaxy0b_y - galaxy1b_y;
                dz = galaxy0b_z - galaxy1b_z;

                float diff_b2 = dx * dx + dy * dy + dz * dz;

                float diff = sqrtf(diff_a2) - sqrtf(diff_b2);
                k_total_diff += diff * diff;
            }
        }
    }

    // k_total_diff /= nf * (nf - 1);
    atomicAdd(&total_diff, k_total_diff);
}

float solveGPU(sGalaxy A, sGalaxy B, int n) {
    //TODO kernel call and data manipulation

    size_t grid_dim_x = 256;
    size_t grid_dim_y = 256;

    size_t block_dim_x = 16;
    size_t block_dim_y = 16;

    size_t k_x = (n - 1) / (grid_dim_x * block_dim_x) + 1; // round up
    size_t k_y = (n - 1) / (grid_dim_y * block_dim_y) + 1;

    std::cout << "    [kernel params]\n";
    std::cout << "grid size : " << grid_dim_x << " x " << grid_dim_y << "\n";
    std::cout << "block size : " << block_dim_x << " x " << block_dim_y << "\n";
    std::cout << "k : " << k_x << " x " << k_y << "\n";
    std::cout << "\n";

    dim3 grid_size(grid_dim_x, grid_dim_y);
    dim3 block_size(block_dim_x, block_dim_y);

    float diff = 0.0f;
    cudaMemcpyToSymbol(total_diff, &diff, sizeof(float));

    kernel_main_simple<<<grid_size, block_size>>>(A, B, n);

    cudaMemcpyFromSymbol(&diff, total_diff, sizeof(float));

    std::cout << "teeeeeeeeeeeeeeeeeeeeest " << diff << "\n";

    float nf = n;
    diff = std::sqrt(diff / (nf * (nf - 1)));

    return diff;
}