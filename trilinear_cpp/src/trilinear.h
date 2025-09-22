#ifndef TRILINEAR_H_
#define TRILINEAR_H_

#include <torch/extension.h>

torch::Tensor trilinear_forward(torch::Tensor lut, torch::Tensor image,
                                int dim, int shift, float binsize, int W, int H, int B);

torch::Tensor trilinear_backward(torch::Tensor image, torch::Tensor grad_output,
                                 int dim, int shift, float binsize, int W, int H, int B);

#endif
