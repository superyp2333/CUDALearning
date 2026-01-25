#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// 定义全局 alpha 值
#define ALPHA 1.0f

// 定义 CHECK_TORCH_TENSOR_DTYPE 宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("Tensor dtype must be " #th_type);                \
  }

// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// FP32
// __device__ 只能在 GPU 侧调用     __global__ CPU 和 GPU 都可以调用
// __forceinline__ 内联函数，即让编译器强制将该函数嵌入至调用处，消除函数调用的开销，提高性能
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// FP16
__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// FP32
// y = elu(x)
// x: N  y:N
// grid(N/256) block(256)
__global__ void elu_f32_kernel(float *x, float *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    y[tid] = elu(x[tid]);
}

// FP32 x 4
// y = elu(x)
// x: N  y:N
// grid(N/256) block(256/4)
// 这里 gird(N/256) 是没问题的！因为每个 block 还是处理了 256 个元素，只不过一个线程干了很多的活！
__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
  // 前面已经处理了多少个元素
  int offset_thd = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (offset_thd < N) {
    // 开一个寄存器，保存 4个连续的 float 元素，第一个元素的位置为 offset_thd
    float4 reg_x = FLOAT4(x[offset_thd]);
    float4 reg_y;

    // 对这 4 个连续的元素进行 elu 计算
    reg_y.x = elu(reg_x.x);
    reg_y.y = elu(reg_x.y);
    reg_y.z = elu(reg_x.z);
    reg_y.w = elu(reg_x.w);

    // 将计算结果写入输出
    FLOAT4(y[offset_thd]) = reg_y;
  }
}

// FP16
// y = elu_half(x)
// x: N  y:N
// grid(N/256) block(256)
__global__ void elu_f16_kernel(half *x, half *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    y[tid] = elu_half(x[tid]);
}

// FP16 x 2
// y = elu_half(x)
// x: N  y:N
// grid(N/256) block(256/2)
__global__ void elu_f16x2_kernel(half *x, half *y, int N) {
  // 前面已经处理了多少个元素
  int offset_thd = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (offset_thd < N) {
    // 开一个寄存器，保存 2 个连续的 half 元素，第一个元素的位置为 offset_thd
    half2 reg_x = HALF2(x[offset_thd]);
    half2 reg_y;
    // 对这 2 个连续的元素进行 elu 计算
    reg_y.x = elu_half(reg_x.x);
    reg_y.y = elu_half(reg_x.y);

    // 将计算结果写入输出
    HALF2(y[offset_thd]) = reg_y;
  }
}

// FP16 x 8
// y = elu_half(x)
// x: N  y:N
// grid(N/256) block(256/8)
__global__ void elu_f16x8_kernel(half *x, half *y, int N) {
  // 前面已经处理了多少个元素
  int offset_thd = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  // 开 4 个寄存器，保存 8 个连续的 half 元素，第一个元素的位置为 offset_thd
  half2 reg_x_0 = HALF2(x[offset_thd + 0]);
  half2 reg_x_1 = HALF2(x[offset_thd + 2]);
  half2 reg_x_2 = HALF2(x[offset_thd + 4]);
  half2 reg_x_3 = HALF2(x[offset_thd + 6]);

  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = elu_half(reg_x_0.x);
  reg_y_0.y = elu_half(reg_x_0.y);
  reg_y_1.x = elu_half(reg_x_1.x);
  reg_y_1.y = elu_half(reg_x_1.y);
  reg_y_2.x = elu_half(reg_x_2.x);
  reg_y_2.y = elu_half(reg_x_2.y);
  reg_y_3.x = elu_half(reg_x_3.x);
  reg_y_3.y = elu_half(reg_x_3.y);
  if ((offset_thd + 0) < N) {
    HALF2(y[offset_thd + 0]) = reg_y_0;
  }
  if ((offset_thd + 2) < N) {
    HALF2(y[offset_thd + 2]) = reg_y_1;
  }
  if ((offset_thd + 4) < N) {
    HALF2(y[offset_thd + 4]) = reg_y_2;
  }
  if ((offset_thd + 6) < N) {
    HALF2(y[offset_thd + 6]) = reg_y_3;
  }
}

// FP16 x 8 pack
// y = elu_half(x)
// x: N  y:N
// grid(N/256) block(256/8)
__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N) {
  // 前面已经处理了多少个元素
  int offset_thd = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  // 一次性加载缓存
  half2 reg_x[4], reg_y[4];
  FLOAT4(reg_x[0]) = FLOAT4(x[offset_thd]);

  reg_y[0].x = elu_half(reg_x[0].x);
  reg_y[0].y = elu_half(reg_x[0].y);
  reg_y[1].x = elu_half(reg_x[1].x);
  reg_y[1].y = elu_half(reg_x[1].y);
  reg_y[2].x = elu_half(reg_x[2].x);
  reg_y[2].y = elu_half(reg_x[2].y);
  reg_y[3].x = elu_half(reg_x[3].x);
  reg_y[3].y = elu_half(reg_x[3].y);

  // 一次性加载缓存
  if (offset_thd < N) {
    FLOAT4(y[offset_thd]) = FLOAT4(reg_y[0]);
  }
}

#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
  void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elu_##packed_type##_kernel<<<grid, block>>>(                             \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elu_f32)
  TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}