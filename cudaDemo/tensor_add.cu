#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

using namespace std;

// 检查 CPU 与 GPU 的运行结果是否一样
void check_result(float *host_res, float *dev_res, const int N) {
    double var = 1e-8; // 误差
    int flag = 1;
    for (int i = 0; i < N; ++i) {
        if (abs(host_res[i] - dev_res[i]) > var) {
            flag = 0;
            break;
        }
    }

    if (flag) {
        cout << "Result Match" << endl;
    } else {
        cout << "Result Not Match !!!" << endl;
    }
}

// compute on cpu
void tensor_add_cpu(float *a, float *b, float *res, const int N) {
    for (int i = 0; i < N; ++i) {
        res[i] = a[i] + b[i];
    }
}

// compute on gpu
__global__ void tensor_add_gpu(float *a, float *b, float *res, const int N) {
    // 1个线程完成一对数组元素的相加
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    res[idx] = a[idx] + b[idx];
}

void init(float *nums, const int N) {
    for (int i = 0; i < N; ++i) {
        nums[i] = i;
    }
}

int main() {
    // 当前 host 线程绑定到第 0 号 GPU
    cudaSetDevice(0);


    int N = 1024;
    int nBytes = N * sizeof(float);

    // 分配主机内存，初始化主机内存
    float *h_a, *h_b, *h_res, *h_truth;
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_res = (float *)malloc(nBytes);
    h_truth = (float *)malloc(nBytes);
    init(h_a, N);
    init(h_b, N);

    // 分配设备内存，初始化设备内存
    float *d_a, *d_b, *d_res;
    cudaMalloc((float**)(&d_a), nBytes);
    cudaMalloc((float**)(&d_b), nBytes);
    cudaMalloc((float**)(&d_res), nBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // gpu 执行计算：启动核函数
    dim3 block(32);
    dim3 grid(N / 32);
    tensor_add_gpu<<<grid, block>>>(d_a, d_b, d_res, N);

    // 将 gpu 上的计算结果拷贝回 cpu
    cudaMemcpy(h_res, d_res, nBytes, cudaMemcpyDeviceToHost);

    // cpu 执行计算：算个真值，用于对比
    tensor_add_cpu(h_a, h_b, h_truth, N);

    // 比较结果
    check_result(h_truth, h_res, N);

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);  
    
    free(h_a);
    free(h_b);
    free(h_res);
    free(h_truth);

    return 0;
}