# CS149 并行计算 · 实验记录

Stanford CS149（Parallel Computing）课程 Lab 的学习与实验记录，包含实现思路、性能测试脚本和实测数据。

## 实验环境

- CPU：x86_64 虚拟机，**2 核可用**（注意：课程参考机器为 4 核 8 超线程，本文档中的加速比数据受 2 核硬件上限约束）
- 编译器：g++ 11；ISPC v1.28.1（`--target=avx2-i32x8`，8 路 SIMD）
- 构建：各程序目录下 `make` 即可

## Lab 1：从多线程到 SIMD

| 程序 | 主题 | 状态 | 实测结果 |
|------|------|------|----------|
| prog1_mandelbrot_threads | 线程级并行、负载均衡 | ✅ 完成 | [results_block.md](lab1/prog1_mandelbrot_threads/results_block.md) / [results_interleaved.md](lab1/prog1_mandelbrot_threads/results_interleaved.md) |
| prog2_vecintrin | SIMD intrinsics、掩码执行 | ✅ 完成 | [results_prog2.md](lab1/prog2_vecintrin/results_prog2.md) |
| prog3_mandelbrot_ispc | ISPC、foreach 与 task | ✅ 完成 | [results_prog3.md](lab1/prog3_mandelbrot_ispc/results_prog3.md) |
| prog4_sqrt | 牛顿迭代、算术强度 | ✅ 完成 | 见下方小结 |
| prog5_saxpy | 内存带宽瓶颈 | ⬜ 待完成 | — |
| prog6_kmeans | 性能热点定位 | ⬜ 待完成 | — |

### prog1：线程间的工作分配

- 连续块分解：加速比卡死在 ~1.8x（2~8 线程全程不变）——拿到昂贵中间行的线程是瓶颈。
- **行交错映射**（线程 i 负责第 i, i+T, i+2T… 行）：静态分配、零同步、对任意线程数通用，各线程耗时趋于一致。
- 附带修复：整除分块在 7 线程时会漏算最后 3 行，交错循环天然覆盖所有行。

### prog2：软件模拟的向量机

- 用掩码把"每元素分支"翻译成"分支两边都算、掩码选结果"（predication）。
- `clampedExpVector`：公式改写为 `1·x^y` 消掉 `y==0` 特判；共享 while 循环 + 每轮收缩掩码。
- `arraySumVector`：主循环向量累加，`hadd + interleave` 做 log2(W) 轮横向归约。
- 实测：VECTOR_WIDTH 2→16，利用率 87.6%→78.3%（分歧 lane 空转），指令数 ~减半但不严格（收益递减 = 利用率损失）。

### prog3：ISPC 的 gang 与 task

- `foreach` 声明独立迭代空间，编译器自动映射到 8 条 SIMD lane；分歧导致加速比只有 5.0x（view1）/ 4.2x（view2）。
- `launch[N]` 把任务扔进运行时队列，worker 动态领取 → 切得碎可以摊平负载不均。
- 实测任务数扫描：view2 从 2 任务的 6.44x 升到 40 任务的 7.14x，16 左右进入平台期；最终选择 **16 任务**（需整除 height=800）。
- 排坑记录：N 不整除 800 时整除截断导致末行漏算，验证失败——任务数必须整除，或自行处理余数。

### prog4：牛顿法求 sqrt

- 迭代式 `g ← (3g − x·g³)/2` 是方程 `1/g² − x = 0` 的牛顿迭代，全程无除法；`√x = x·g`。
- 全填 1.0：0 次迭代，内核塌缩为内存带宽密集，加速比只剩 1.84x —— **没有计算就没有并行收益**。
- 全填 2.999（最大迭代数且完全均匀）：SIMD 5.92x，加任务 9.81x。

## 自动化测试脚本

prog1/2/3 各目录下有 `run_sweep.sh`，自动完成"改参数 → 重编 → 跑全档位 → 解析输出 → 生成 Markdown 结果表"：

```bash
cd lab1/prog1_mandelbrot_threads && ./run_sweep.sh   # 线程数 2~8 × 两个视图
cd lab1/prog2_vecintrin && ./run_sweep.sh            # VECTOR_WIDTH 2/4/8/16
cd lab1/prog3_mandelbrot_ispc && ./run_sweep.sh      # 任务数 2~40 × 两个视图
```

## 其他笔记

- [git.md](git.md)：git 使用笔记
- [learning-roadmap.md](learning-roadmap.md)：学习路线
- 课程原始说明：[lab1/README_zh.md](lab1/README_zh.md)（中文）/ [lab1/README.md](lab1/README.md)（英文）
