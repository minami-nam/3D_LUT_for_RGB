#include "trilinear.h"


void TriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

void TriLinearBackwardCpu(const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    // 1) 연속 메모리 보장
    auto lut_c  = lut.contiguous();
    auto img_c  = image.contiguous();
    auto out_c  = output.contiguous();

    // 2) 안전한 포인터 획득
    float *lut_flat    = lut_c.data_ptr<float>();
    float *image_flat  = img_c.data_ptr<float>();
    float *output_flat = out_c.data_ptr<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriLinearForwardCpu(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;
}

int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // 1) 비연속 텐서도 안전하게 처리하려면 contiguous() 호출
    auto img_c     = image.contiguous();
    auto img_g_c   = image_grad.contiguous();
    auto lut_g_c   = lut_grad.contiguous();

    // 2) modern API 로 포인터 획득
    float *image_flat      = img_c.data_ptr<float>();
    float *image_grad_flat = img_g_c.data_ptr<float>();
    float *lut_grad_flat   = lut_g_c.data_ptr<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriLinearBackwardCpu(image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;
}

void TriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;;

    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = image[index];
	float g = image[index + width * height];
	float b = image[index + width * height * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	float r_d = fmod(r,binsize) / binsize;
	float g_d = fmod(g,binsize) / binsize;
	float b_d = fmod(b,binsize) / binsize;

	int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

	float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

	output[index] = w000 * lut[id000] + w100 * lut[id100] + 
                        w010 * lut[id010] + w110 * lut[id110] + 
                        w001 * lut[id001] + w101 * lut[id101] + 
                        w011 * lut[id011] + w111 * lut[id111];

	output[index + width * height] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] + 
                      		         w010 * lut[id010 + shift] + w110 * lut[id110 + shift] + 
                       		         w001 * lut[id001 + shift] + w101 * lut[id101 + shift] + 
                               	         w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

	output[index + width * height * 2] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] + 
                           		     w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] + 
                           		     w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] + 
                                	     w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    }
}

void TriLinearBackwardCpu(const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;

    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = image[index];
	float g = image[index + width * height];
	float b = image[index + width * height * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	float r_d = fmod(r,binsize) / binsize;
	float g_d = fmod(g,binsize) / binsize;
	float b_d = fmod(b,binsize) / binsize;

	int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

	float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

	lut_grad[id000] += w000 * image_grad[index];
        lut_grad[id100] += w100 * image_grad[index];
        lut_grad[id010] += w010 * image_grad[index];
        lut_grad[id110] += w110 * image_grad[index];
        lut_grad[id001] += w001 * image_grad[index];
        lut_grad[id101] += w101 * image_grad[index];
        lut_grad[id011] += w011 * image_grad[index];
        lut_grad[id111] += w111 * image_grad[index];

        lut_grad[id000 + shift] += w000 * image_grad[index + width * height];
        lut_grad[id100 + shift] += w100 * image_grad[index + width * height];
        lut_grad[id010 + shift] += w010 * image_grad[index + width * height];
        lut_grad[id110 + shift] += w110 * image_grad[index + width * height];
        lut_grad[id001 + shift] += w001 * image_grad[index + width * height];
        lut_grad[id101 + shift] += w101 * image_grad[index + width * height];
        lut_grad[id011 + shift] += w011 * image_grad[index + width * height];
        lut_grad[id111 + shift] += w111 * image_grad[index + width * height];

        lut_grad[id000 + shift* 2] += w000 * image_grad[index + width * height * 2];
        lut_grad[id100 + shift* 2] += w100 * image_grad[index + width * height * 2];
        lut_grad[id010 + shift* 2] += w010 * image_grad[index + width * height * 2];
        lut_grad[id110 + shift* 2] += w110 * image_grad[index + width * height * 2];
        lut_grad[id001 + shift* 2] += w001 * image_grad[index + width * height * 2];
        lut_grad[id101 + shift* 2] += w101 * image_grad[index + width * height * 2];
        lut_grad[id011 + shift* 2] += w011 * image_grad[index + width * height * 2];
        lut_grad[id111 + shift* 2] += w111 * image_grad[index + width * height * 2];
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_forward, "Trilinear forward");
  m.def("backward", &trilinear_backward, "Trilinear backward");
}
