#include <algorithm>
#include <cuda_fp16.h> // CUDA 半精度浮点型（FP16/half） 相关类型和操作
#include <cuda_runtime.h> // CUDA 运行时 API 的所有核心声明
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h> // 连接 Python 端 PyTorch 和 C++/CUDA 代码的桥梁
#include <torch/types.h> // 定义 PyTorch C++ 接口的基础类型
#include <vector>

// 每个 block 中的线程最多为 1024，由 GPU 硬件决定
#define MAX_BLOCK_THREAD 1024

// FP32
// x: N  y:N
// y = max(0, N)
// block(256)  grid(N/256)
__global__ void relu_f32_kernel(float *x, float *y, const int N) {
    // 一维线程模型：每个线程处理一个元素
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // fmaxf -> 单精度浮点最大值函数
        // 0.0后面的 f 必须加，避免发生隐式类型转换，因为不加则默认为 double 双精度
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

// 对 Kernel 封装成一个 Python 可以识别的启动函数
// Python Pytorch 传进来的参数肯定是个 tensor 张量
// 而且我们的输出肯定也必须是个 tensor 张量
void relu_f32(torch::Tensor x, torch::Tensor y) {
    // 检查入参是否为 fp32 单精度浮点数，对应 python 中的 torch.float32
    if(x.options().dtype() != torch::kFloat32) {
        std::cout << "Tensor info: " << x.options() << std::endl;
        throw std::runtime_error("value must be torch::kFloat32\n"); 
    }
    if(y.options().dtype() != torch::kFloat32) {
        std::cout << "Tensor info: " << y.options() << std::endl;
        throw std::runtime_error("value must be torch::kFloat32\n"); 
    }

    // 获得输入 x 的维度
    const int ndim = x.dim();
    if (ndim == 1 || ndim > 2) {
        // 如果 x 是一维向量或者大于 2 维的向量，则将其拉平成一维向量
        int N = 1;
        for (int i = 0; i < ndim; ++i) {
            // 遍历向量 x 的每个维度，计算共有多少个元素
            N *= x.size(i);
        }

        dim3 block(256);
        dim3 grid((N + 256 - 1) / 256); // 相当于对 N/256 向上取整，保证 Blcok 的数量够用
        relu_f32_kernel<<<grid, block>>> (
            reinterpret_cast<float *>(x.data_ptr()), // 将向量 x (不管是多少维)，强制类型转换成一维的 float *
            reinterpret_cast<float *>(y.data_ptr()), 
            N
        );
    } else {
        // 如果 x 是二维向量，可以做一些优化处理
        int x_dim = x.size(0);
        int y_dim = y.size(1);
        int N = x_dim * y_dim;
        if (x_dim <= MAX_BLOCK_THREAD) {
            dim3 block(x_dim);
            dim3 grid(y_dim);
            relu_f32_kernel<<<grid, block>>> (
                reinterpret_cast<float *>(x.data_ptr()),
                reinterpret_cast<float *>(y.data_ptr()), 
                N
            );
        } else {
            dim3 block(256);
            dim3 grid((N + 256 - 1) / 256);
            relu_f32_kernel<<<grid, block>>> (
                reinterpret_cast<float *>(x.data_ptr()),
                reinterpret_cast<float *>(y.data_ptr()), 
                N
            );
        }
    }
}

// 将 Pytorch C++ 接口暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("Python调用时的函数名", &C++函数名, "备注")
    m.def("relu_f32", &relu_f32, "This is a relu_f32_kernel interface ...");
}