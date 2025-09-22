// src/trilinear_cuda.cpp

#include "trilinear_kernel.h"
#include <torch/extension.h>

int TrilinearForwardCUDA(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                         int lut_dim, int shift, float binsize,
                         int width, int height, int batch);

int TrilinearBackwardCUDA(torch::Tensor image, torch::Tensor grad_output, torch::Tensor grad_lut,
                          int lut_dim, int shift, float binsize,
                          int width, int height, int batch);

torch::Tensor trilinear_forward(torch::Tensor lut, torch::Tensor image, int dim, int shift, float binsize, int W, int H, int B) {
    auto output = torch::zeros_like(image);
    TrilinearForwardCUDA(lut, image, output, dim, shift, binsize, W, H, B);
    return output;
}

torch::Tensor trilinear_backward(torch::Tensor image, torch::Tensor grad_output, int dim, int shift, float binsize, int W, int H, int B) {
    auto grad_lut = torch::zeros({3, dim, dim, dim}, image.options());
    TrilinearBackwardCUDA(image, grad_output, grad_lut, dim, shift, binsize, W, H, B);
    return grad_lut;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &trilinear_forward, "Trilinear forward (CUDA)");
    m.def("backward", &trilinear_backward, "Trilinear backward (CUDA)");
}