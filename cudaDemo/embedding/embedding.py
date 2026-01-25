import time
from functools import partial
from typing import Optional

import torch
from torch.nn.functional import embedding
from torch.utils.cpp_extension import load

# 禁用反向传播
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="embedding", # 生成的扩展模块名（Python中import的名称）
    sources=["embedding.cu"], # 待编译的CUDA源文件列表
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
    perf_func: callable,  # 要测试的函数（比如 lib.embedding）
    a: torch.Tensor,      # 输入张量
    b: torch.Tensor,      # 输入张量
    tag: str,             # 测试标签（用于输出区分，比如 "f32" "f16_th"）
    out: Optional[torch.Tensor] = None,  # 输出张量（可选，避免重复创建）
    warmup: int = 2,     # 预热次数（消除GPU初始化开销）
    iters: int = 20,    # 测试迭代次数（取平均更稳定）
    show_all: bool = False,  # 是否打印完整输出张量
):
    # 1. 初始化输出张量（如果传入）：清空为 0，保证每次测试初始状态一致
    if out is not None:
        out.fill_(0)
    
    # 2. 预热阶段：GPU 首次运行算子会有初始化开销（如核函数编译、显存分配），预热消除该影响
    if out is not None:
        # 有输出张量：直接写入out
        for i in range(warmup):
            perf_func(a, b, out)  
    else:
        # 无输出张量：丢弃返回值
        for i in range(warmup):
            _ = perf_func(a, b)   
    torch.cuda.synchronize()  # 关键动作：等待GPU预热完成（GPU异步执行，必须同步）

    # 3. 正式计时阶段
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)  
    else:
        for i in range(iters):
            out = perf_func(a, b)   
    torch.cuda.synchronize()  # 关键动作：等待GPU所有迭代完成，确保计时准确
    end = time.time()

    # 4. 计算耗时并格式化输出
    total_time = (end - start) * 1000  # 总耗时转毫秒（ms）
    mean_time = total_time / iters     # 单次迭代平均耗时（核心性能指标）
    
    # 结果验证：取输出张量前2个元素（保留8位小数），确保算子计算结果正确
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]  # 展平取前3个值
    out_val = [round(v, 8) for v in out_val]                     # 保留8位小数
    out_val = [f"{v:<12}" for v in out_val]                      # 格式化输出（左对齐）
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")  # 右对齐标签，显示结果+耗时
    
    # 可选：打印完整输出张量（调试用）
    if show_all:
        print(out)
    return out, mean_time

Ms = [1024, 4096]  # max value of token_ids
Ns = [2048, 4096]  # seqlen
Ks = [512, 1024]  # embedding size
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for M, N, K in MNKs:
    print("-" * 110)
    print(" " * 45 + f"MaxV={M}, SeqLen={N}, EmbSize={K}")
    i = torch.randint(0, M, size=(N,)).cuda().int().contiguous()
    weight = torch.randn((M, K)).float().cuda().contiguous()
    o = torch.zeros((N, K)).float().cuda().contiguous()

    run_benchmark(lib.embedding_f32, i, weight, "f32", o)
    run_benchmark(lib.embedding_f32x4, i, weight, "f32x4", o)
    run_benchmark(lib.embedding_f32x4_pack, i, weight, "f32x4_pack", o)
    run_benchmark(partial(embedding), i, weight, "f32_th")

    print("-" * 110)
    weight_f16 = torch.randn((M, K)).half().cuda().contiguous()
    o_f16 = torch.zeros((N, K)).half().cuda().contiguous()
    run_benchmark(lib.embedding_f16, i, weight_f16, "f16", o_f16)
    run_benchmark(lib.embedding_f16x8, i, weight_f16, "f16x8", o_f16)
    run_benchmark(lib.embedding_f16x8_pack, i, weight_f16, "f16x8_pack", o_f16)
    run_benchmark(partial(embedding), i, weight_f16, "f16_th")
    print("-" * 110)