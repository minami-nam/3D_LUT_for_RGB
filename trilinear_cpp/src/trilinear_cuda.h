#ifndef TRILINEAR_CUDA_H_
#define TRILINEAR_CUDA_H_

#include <torch/extension.h>

int TriLinearForwardLaucher(const float* lut, const float* image, float* output,
                            const int lut_dim, const int shift, const float binsize,
                            const int width, const int height, const int batch,
                            cudaStream_t stream);

int TriLinearBackwardLaucher(const float* image, const float* image_grad, float* lut_grad,
                             const int lut_dim, const int shift, const float binsize,
                             const int width, const int height, const int batch,
                             cudaStream_t stream);

#endif