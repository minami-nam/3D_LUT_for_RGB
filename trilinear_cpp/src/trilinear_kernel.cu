#include <math.h>
#include <float.h>
#include "trilinear_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)


// ==================================================
// Forward Kernel
// ==================================================
__global__ void TriLinearForward(
    const int nthreads,
    const float* lut,       // [3 * dim^3]
    const float* image,     // [B, 3, H, W]
    float* output,          // [B, 3, H, W]
    const int dim,
    const int shift,
    const float binsize,
    const int width,
    const int height,
    const int batch) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // index -> (b, y, x)
        int hw = index % (height * width);
        int b  = index / (height * width);
        int y  = hw / width;
        int x  = hw % width;

        // NCHW layout
        int base = b * 3 * height * width + y * width + x;

        float r = image[base + 0 * height * width];
        float g = image[base + 1 * height * width];
        float bch = image[base + 2 * height * width];

        // Clamp to [0,1]
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        bch = fminf(fmaxf(bch, 0.0f), 1.0f);

        int safe_max = dim - 1;

        int r_id = min(max((int)floorf(r / binsize), 0), safe_max - 1);
        int g_id = min(max((int)floorf(g / binsize), 0), safe_max - 1);
        int b_id = min(max((int)floorf(bch / binsize), 0), safe_max - 1);

        float r_d = (r - r_id * binsize) / binsize;
        float g_d = (g - g_id * binsize) / binsize;
        float b_d = (bch - b_id * binsize) / binsize;

        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = id000 + 1;
        int id010 = id000 + dim;
        int id110 = id010 + 1;
        int id001 = id000 + dim * dim;
        int id101 = id001 + 1;
        int id011 = id001 + dim;
        int id111 = id011 + 1;

        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

        // R channel
        output[base + 0 * height * width] =
            w000 * lut[id000] + w100 * lut[id100] +
            w010 * lut[id010] + w110 * lut[id110] +
            w001 * lut[id001] + w101 * lut[id101] +
            w011 * lut[id011] + w111 * lut[id111];

        // G channel
        output[base + 1 * height * width] =
            w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
            w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
            w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
            w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

        // B channel
        output[base + 2 * height * width] =
            w000 * lut[id000 + shift*2] + w100 * lut[id100 + shift*2] +
            w010 * lut[id010 + shift*2] + w110 * lut[id110 + shift*2] +
            w001 * lut[id001 + shift*2] + w101 * lut[id101 + shift*2] +
            w011 * lut[id011 + shift*2] + w111 * lut[id111 + shift*2];
    }
}

int TriLinearForwardLaucher(
    const float* lut, const float* image, float* output,
    const int lut_dim, const int shift, const float binsize,
    const int width, const int height, const int batch, cudaStream_t stream) 
{
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;

    TriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
        output_size, lut, image, output, lut_dim, shift, binsize, width, height, batch);

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf(stderr, "TriLinearForward kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}


// ==================================================
// Backward Kernel
// ==================================================
__global__ void TriLinearBackward(
    const int nthreads,
    const float* image,
    const float* image_grad,
    float* lut_grad,
    const int dim,
    const int shift,
    const float binsize,
    const int width,
    const int height,
    const int batch) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int hw = index % (height * width);
        int b  = index / (height * width);
        int y  = hw / width;
        int x  = hw % width;

        int base = b * 3 * height * width + y * width + x;

        float r = image[base + 0 * height * width];
        float g = image[base + 1 * height * width];
        float bch = image[base + 2 * height * width];

        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        bch = fminf(fmaxf(bch, 0.0f), 1.0f);

        int safe_max = dim - 1;
        int r_id = min(max((int)floorf(r / binsize), 0), safe_max - 1);
        int g_id = min(max((int)floorf(g / binsize), 0), safe_max - 1);
        int b_id = min(max((int)floorf(bch / binsize), 0), safe_max - 1);

        float r_d = (r - r_id * binsize) / binsize;
        float g_d = (g - g_id * binsize) / binsize;
        float b_d = (bch - b_id * binsize) / binsize;

        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = id000 + 1;
        int id010 = id000 + dim;
        int id110 = id010 + 1;
        int id001 = id000 + dim * dim;
        int id101 = id001 + 1;
        int id011 = id001 + dim;
        int id111 = id011 + 1;

        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

        float grad_r = image_grad[base + 0 * height * width];
        float grad_g = image_grad[base + 1 * height * width];
        float grad_b = image_grad[base + 2 * height * width];

        // R
        atomicAdd(lut_grad + id000, grad_r * w000);
        atomicAdd(lut_grad + id100, grad_r * w100);
        atomicAdd(lut_grad + id010, grad_r * w010);
        atomicAdd(lut_grad + id110, grad_r * w110);
        atomicAdd(lut_grad + id001, grad_r * w001);
        atomicAdd(lut_grad + id101, grad_r * w101);
        atomicAdd(lut_grad + id011, grad_r * w011);
        atomicAdd(lut_grad + id111, grad_r * w111);

        // G
        atomicAdd(lut_grad + id000 + shift, grad_g * w000);
        atomicAdd(lut_grad + id100 + shift, grad_g * w100);
        atomicAdd(lut_grad + id010 + shift, grad_g * w010);
        atomicAdd(lut_grad + id110 + shift, grad_g * w110);
        atomicAdd(lut_grad + id001 + shift, grad_g * w001);
        atomicAdd(lut_grad + id101 + shift, grad_g * w101);
        atomicAdd(lut_grad + id011 + shift, grad_g * w011);
        atomicAdd(lut_grad + id111 + shift, grad_g * w111);

        // B
        atomicAdd(lut_grad + id000 + shift*2, grad_b * w000);
        atomicAdd(lut_grad + id100 + shift*2, grad_b * w100);
        atomicAdd(lut_grad + id010 + shift*2, grad_b * w010);
        atomicAdd(lut_grad + id110 + shift*2, grad_b * w110);
        atomicAdd(lut_grad + id001 + shift*2, grad_b * w001);
        atomicAdd(lut_grad + id101 + shift*2, grad_b * w101);
        atomicAdd(lut_grad + id011 + shift*2, grad_b * w011);
        atomicAdd(lut_grad + id111 + shift*2, grad_b * w111);
    }
}

int TriLinearBackwardLaucher(
    const float* image, const float* image_grad, float* lut_grad,
    const int lut_dim, const int shift, const float binsize,
    const int width, const int height, const int batch, cudaStream_t stream) 
{
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;

    TriLinearBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        kThreadsPerBlock, 0, stream>>>(
        output_size, image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch);

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf(stderr, "TriLinearBackward kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}
