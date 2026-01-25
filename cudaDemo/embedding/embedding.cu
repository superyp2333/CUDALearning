#include <algorithm>
#include <cuda_fp16.h> // CUDA 半精度浮点型（FP16/half） 相关类型和操作
#include <cuda_runtime.h> // CUDA 运行时 API 的所有核心声明
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h> // 连接 Python 端 PyTorch 和 C++/CUDA 代码的桥梁
#include <torch/types.h> // 定义 PyTorch C++ 接口的基础类型
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// input: seq_len  weight: (max_token_id, emb_dim)   emb_dim < 1024
// output: (seq_len, emb_dim)
// grid(seq_len) block(emb_dim)
// 前提条件：emb_dim < 1024
__global__ void embedding_f32_kernel(int *input, float *weight, float *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // CUDA 中不支持用[][]直接索引显存，因为 float * 指向的内存是一维排列的
    int offset = input[bx] * emb_dim; // 当前线程所读取的 weight 行的行首元素的位置
    output[tid] = weight[offset + tx];
}

// FP32 x 4  优化访存效率，每个线程每次处理 4 个元素
// grid(seq_len)  block(emb_dim / 4)
__global__ void embedding_f32x4_kernel(int *input, float *weight, float *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;

    int offset = input[bx] * emb_dim; // 当前线程所读取的 weight 行的行首元素的位置
    int offset_thd = threadIdx.x * 4;  // 当前行已经读了多少个位置
    output[tid] = weight[offset + offset_thd];
    output[tid + 1] = weight[offset + offset_thd + 1];
    output[tid + 2] = weight[offset + offset_thd + 2];
    output[tid + 3] = weight[offset + offset_thd + 3];
}

// FP32 x 4  进一步优化访存效率，单次指令读写 4 个 float 元素，比手动写 4 行赋值语句更高效
// grid(seq_len)  block(emb_dim / 4)
__global__ void embedding_f32x4_pack_kernel(int *input, float *weight, float *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;

    int offset = input[bx] * emb_dim; // 当前线程所读取的 weight 行的行首元素的位置
    int offset_thd = threadIdx.x * 4;  // 当前行已经读了多少个位置
    FLOAT4(output[bx * emb_dim + offset_thd]) = FLOAT4(weight[offset + offset_thd]);
}

// FP16
// grid(seq_len) block(emb_dim)
__global__ void embedding_f16_kernel(int *input, half *weight, half *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int offset = input[bx] * emb_dim;
    output[tid] = weight[offset + tx];
}

// FP16 x 8  
// grid(seq_len)  block(emb_dim / 8)
__global__ void embedding_f16x8_kernel(int *input, half *weight, half *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;

    int offset = input[bx] * emb_dim; // 当前线程所读取的 weight 行的行首元素的位置
    int offset_thd = threadIdx.x * 8;  // 当前行已经读了多少个位置
    output[tid] = weight[offset + offset_thd];
    output[tid + 1] = weight[offset + offset_thd + 1];
    output[tid + 2] = weight[offset + offset_thd + 2];
    output[tid + 3] = weight[offset + offset_thd + 3];
    output[tid + 4] = weight[offset + offset_thd + 4];
    output[tid + 5] = weight[offset + offset_thd + 5];
    output[tid + 6] = weight[offset + offset_thd + 6];
    output[tid + 7] = weight[offset + offset_thd + 7];
}

// FP16 x 8  进一步优化访存效率，单次指令读写 4 个 float 元素，相当于 8 个 haf 元素
// grid(seq_len)  block(emb_dim / 8)
__global__ void embedding_f16x8_pack_kernel(int *input, half *weight, half *output, int seq_len, int emb_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;

    int offset = input[bx] * emb_dim; // 当前线程所读取的 weight 行的行首元素的位置
    int offset_thd = threadIdx.x * 8;  // 当前行已经读了多少个位置
    FLOAT4(output[bx * emb_dim + offset_thd]) = FLOAT4(weight[offset + offset_thd]);
}

// m.def("Python调用时的函数名", &C++函数名, "备注")
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int seq_len = a.size(0);                                                   \
    const int emb_dim = weight.size(1);                                       \
    dim3 block(emb_dim / n_elements);                                         \
    dim3 grid(seq_len);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), seq_len, emb_dim);          \
  }

TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

// 将 Pytorch C++ 接口暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack);
}