#include <torch/extension.h>
#include "trilinear_kernel.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>   // for at::cuda::getCurrentCUDAStream()

int TrilinearForwardCUDA(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                         int lut_dim, int shift, float binsize,
                         int width, int height, int batch) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return TriLinearForwardLaucher(
        lut.data_ptr<float>(),
        image.data_ptr<float>(),
        output.data_ptr<float>(),
        lut_dim, shift, binsize, width, height, batch, stream
    );
}

int TrilinearBackwardCUDA(torch::Tensor image, torch::Tensor grad_output, torch::Tensor grad_lut,
                          int lut_dim, int shift, float binsize,
                          int width, int height, int batch) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return TriLinearBackwardLaucher(
        image.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_lut.data_ptr<float>(),
        lut_dim, shift, binsize, width, height, batch, stream
    );
}
