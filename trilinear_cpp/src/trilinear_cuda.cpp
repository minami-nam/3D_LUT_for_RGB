// src/trilinear_cuda.cpp

#include "trilinear_kernel.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>   // for c10::cuda::getCurrentCUDAStream()

// CUDA Forward wrapper
int trilinear_forward_cuda(torch::Tensor lut,
                           torch::Tensor image,
                           torch::Tensor output,
                           int lut_dim,
                           int shift,
                           float binsize,
                           int width,
                           int height,
                           int batch) {
    // 1) Ensure contiguous memory
    auto lut_c = lut.contiguous();
    auto img_c = image.contiguous();
    auto out_c = output.contiguous();

    // 2) Get raw pointers
    float *lut_flat    = lut_c.data_ptr<float>();
    float *image_flat  = img_c.data_ptr<float>();
    float *output_flat = out_c.data_ptr<float>();

    // 3) Get current CUDA stream from PyTorch
    auto stream_wrapper = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = stream_wrapper.stream();

    // 4) Launch the CUDA kernel
    TriLinearForwardLaucher(
        lut_flat,
        image_flat,
        output_flat,
        lut_dim,
        shift,
        binsize,
        width,
        height,
        batch,
        stream
    );

    return 1;
}

// CUDA Backward wrapper
int trilinear_backward_cuda(torch::Tensor image,
                            torch::Tensor image_grad,
                            torch::Tensor lut_grad,
                            int lut_dim,
                            int shift,
                            float binsize,
                            int width,
                            int height,
                            int batch) {
    // 1) Ensure contiguous memory
    auto img_c   = image.contiguous();
    auto img_g_c = image_grad.contiguous();
    auto lut_g_c = lut_grad.contiguous();

    // 2) Get raw pointers
    float *image_flat      = img_c.data_ptr<float>();
    float *image_grad_flat = img_g_c.data_ptr<float>();
    float *lut_grad_flat   = lut_g_c.data_ptr<float>();

    // 3) Get current CUDA stream
    auto stream_wrapper = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = stream_wrapper.stream();

    // 4) Launch the CUDA kernel
    TriLinearBackwardLaucher(
        image_flat,
        image_grad_flat,
        lut_grad_flat,
        lut_dim,
        shift,
        binsize,
        width,
        height,
        batch,
        stream
    );

    return 1;
}

// Python bindings via pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &trilinear_forward_cuda,  "Trilinear forward (CUDA)");
    m.def("backward", &trilinear_backward_cuda, "Trilinear backward (CUDA)");
}
