import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

# 禁用反向传播
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="relu_lib", # 生成的扩展模块名（Python中import的名称）
    sources=["relu.cu"], # 待编译的CUDA源文件列表
    extra_cuda_cflags=[ # 传给nvcc（CUDA编译器）的编译参数
        "-O3",                           # 最高级别优化（速度优先，推理场景推荐）
        "-U__CUDA_NO_HALF_OPERATORS__",  # 取消禁用FP16算子（启用半精度运算）
        "-U__CUDA_NO_HALF_CONVERSIONS__",# 取消禁用FP16类型转换（支持__half类型）
        "-U__CUDA_NO_HALF2_OPERATORS__", # 取消禁用half2向量运算（SIMD加速）
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",  # 取消禁用BF16类型转换
        "--expt-relaxed-constexpr",      # 放宽constexpr限制（支持CUDA高级特性）
        "--expt-extended-lambda",        # 启用CUDA扩展lambda表达式（简化核函数）
        "--use_fast_math",               # 启用快速数学函数（牺牲少量精度换速度）
    ],
    extra_cflags=["-std=c++17"], # 传给C++编译器的参数（启用C++17标准）
)

# 性能测试函数
def run_benchmark(
    perf_func: callable,  # 要测试的函数（比如 lib.relu_f32）
    x: torch.Tensor,      # 输入张量
    tag: str,             # 测试标签（用于输出区分，比如 "f32" "f16_th"）
    out: Optional[torch.Tensor] = None,  # 输出张量（可选，避免重复创建）
    warmup: int = 10,     # 预热次数（消除GPU初始化开销）
    iters: int = 1000,    # 测试迭代次数（取平均更稳定）
    show_all: bool = False,  # 是否打印完整输出张量
):
    # 1. 初始化输出张量（如果传入）：清空为 0，保证每次测试初始状态一致
    if out is not None:
        out.fill_(0)
    
    # 2. 预热阶段：GPU 首次运行算子会有初始化开销（如核函数编译、显存分配），预热消除该影响
    if out is not None:
        # 有输出张量：直接写入out
        for i in range(warmup):
            perf_func(x, out)  
    else:
        # 无输出张量：丢弃返回值
        for i in range(warmup):
            _ = perf_func(x)   
    torch.cuda.synchronize()  # 关键动作：等待GPU预热完成（GPU异步执行，必须同步）

    # 3. 正式计时阶段
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()  # 关键动作：等待GPU所有迭代完成，确保计时准确
    end = time.time()

    # 4. 计算耗时并格式化输出
    total_time = (end - start) * 1000  # 总耗时转毫秒（ms）
    mean_time = total_time / iters     # 单次迭代平均耗时（核心性能指标）
    
    # 结果验证：取输出张量前2个元素（保留8位小数），确保算子计算结果正确
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]  # 展平取前2个值
    out_val = [round(v, 8) for v in out_val]                     # 保留8位小数
    out_val = [f"{v:<12}" for v in out_val]                      # 格式化输出（左对齐）
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")  # 右对齐标签，显示结果+耗时
    
    # 可选：打印完整输出张量（调试用）
    if show_all:
        print(out)
    return out, mean_time


# 尝试多种维度
x_dim = [1024, 2048, 4096]
y_dim = [1024, 2048, 4096]
xy_dim = [(x, y) for x in x_dim for y in y_dim] # x_dim 和 y_dim 中的元素两两配对，共 9 种组合

for x_dim, y_dim in xy_dim:
    print("-" * 85)
    print(" " * 40 + f"x_dim={x_dim}, y_dim={y_dim}")

    # 分配内存
        # .randn((x_dim, y_dim)) --> 创建 CPU 上的二维随机张量，尺寸为 x_dim * y_dim，数值满足标准正态分布，默认精度为 float32（FP32）
        # .cuda() --> 将张量从 CPU 转移到 GPU（显存），原 CPU 张量会被垃圾回收（无引用）
        # .float() --> 强制将张量精度转为 float32（FP32），如果已是 FP32，则无任何操作
        # .contiguous() --> 确保张量在 GPU 显存中是连续存储的
        # .zeros_like(input_x) --> 创建和 input_x 尺寸/精度/设备完全一致的全 0 张量（因为 input_x 已在 GPU，所以这一步直接创建在 GPU）
    input_x = torch.randn((x_dim, y_dim)).cuda().float().contiguous()
    output_y = torch.zeros_like(input_x).cuda().float().contiguous()

    # 启动 C++ 侧的核函数
    lib.relu_f32(input_x, output_y)

    # Pytorch 计算的真值
    th_out = torch.relu(input_x)

    # 查看前 5 个计算结果
    print(output_y[:5])
    print(th_out[:5])
    print('...')