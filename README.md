# CUDA 学习

## 一、仓库管理

```bash
# 初始化本地库
git init
# 本地库与远程库建立关联：仓库别名默认为origin
git remote add origin git@github.com:superyp2333/CUDALearning.git
git remote -v

# 将修改添加到暂存区跟踪
git add .
# 提交暂存区，形成提交记录
git commit -m "第一次提交"

# 修改上次提交
git add .
# 只合并代码修改，「保留原来的提交备注」（推荐）
git commit --amend --no-edit
# 合并代码修改 + 「同时修改提交备注」（按需使用）
git commit --amend -m "balabala"

# 拉取最新代码：单人用pull「拉取+合并，会直接报冲突」，多人用fetch「只拉取，不合并，需要手动合并」
git pull
git fetch

# 首次推送需要加 -u，将「本地的main分支」与「远程origin仓库的main分支」建立关联
git push -u origin master
# 建立起关联后，后续推送可简化
git push
```



## 二、Kaggle

### 2.1 GPU环境准备

🚀 获取免费的GPU环境：[Kaggle](https://www.kaggle.com/)

<img src="./assets/image-20260110171544563.png" alt="image-20260110171544563" />

⚠️ 注意：手动关闭后台任务，否则GPU一直在占用，很快就把每周30小时的GPU额度用完了

<img src="./assets/image-20260110172228083.png" alt="image-20260110172228083.png" width="60%" />

✅ 查看每周剩余的GPU额度：头像 -> settings -> Quotas

<img src="./assets/image-20260111144127561.png" alt="image-20260111144127561" width="60%" />

### 2.2 CUDA C/Cpp

**在 Kaggle Notebook 中运行 CUDA C/Cpp 代码，需要在环境中安装一个CUDA运行插件，否则无法编译运行**

✅ 1. 验证 GPU 是否配置正确、验证 nvcc 编译器是否安装

```bash
# 查询当前机器的 GPU 硬件信息 + 显卡驱动版本 + CUDA 驱动 API 版本 + 显存占用 / 算力状态
!nvidia-smi

# 查看 CUDA C/Cpp 编译器版本
!nvcc -v
```

✅ 2. 安装 `nvcc4jupyter` 插件

```bash
# 安装插件
!pip install nvcc4jupyter

# 加载插件，让该环境支持直接编译、运行 CUDA C/Cpp 代码；在运行 CUDA 代码时，必须在代码前加上魔法命令：%%cuda
%load_ext nvcc4jupyter
```

✅ 3. 运行 CUDA C/Cpp 代码

```cpp
%%cuda
#include <iostream>
using namespace std;

// GPU端 CUDA核函数：最简单的加法运算
__global__ void simple_add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    // CPU 端 变量 (主机端 Host)
    int a = 10, b = 20, res;
    // GPU 端 指针 (设备端 Device)
    int *d_a, *d_b, *d_c;

    // 1. 给GPU分配显存
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    // 2. CPU数据 拷贝到 GPU显存
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // 3. 启动GPU核函数：<<<线程块数, 每个线程块的线程数>>>
    simple_add<<<1, 1>>>(d_a, d_b, d_c);

    // 4. GPU计算结果 拷贝回 CPU内存
    cudaMemcpy(&res, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    cout << "CUDA验证成功！计算结果：" << a << " + " << b << " = " << res << endl;

    // 释放GPU显存（好习惯）
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

<img src="./assets/image-20260111160415399.png" alt="image-20260111160415399" width="60%"/>
