# CUDA 学习

## 1、仓库管理

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

# 首次推送需要加 -u，将「本地的main分支」与「远程origin仓库的main分支」建立关联
git push -u origin master
# 建立起关联后，后续推送可简化
git push

# 拉取最新代码：单人用pull「拉取+合并，会直接报冲突」，多人用fetch「只拉取，不合并，需要手动合并」
git pull
git fetch
```



## 2、Kaggle

获取免费的GPU环境：[Kaggle](https://www.kaggle.com/)

![image-20260110171544563](./assets/image-20260110171544563.png)

每周30小时的GPU

<img src="./assets/image-20260110172228083.png" alt="image-20260110172228083" style="zoom: 50%;" />

