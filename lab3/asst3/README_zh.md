# 作业 3：一个简单的 CUDA 渲染器

**截止日期：10 月 30 日（周四）23:59 PST**

**总分 100 分**

![My Image](handout/teaser.jpg?raw=true)

## 概述

在本作业中，你将用 CUDA 编写一个绘制彩色圆形的并行渲染器。
虽然这个渲染器很简单，但并行化它会要求你设计并实现
能够高效地并行构建和操作的数据结构。这是一个有难度的作业，
建议你尽早开始。**说真的，尽早开始。** 祝好运！

## 环境配置

1. 你需要在 AWS 带 GPU 的虚拟机上采集结果（即跑性能测试）。请按照 [cloud_readme.md](cloud_readme.md) 的说明搭建运行环境。

2. 从课程 Github 下载起步代码：

`git clone https://github.com/stanford-cs149/asst3`

CUDA C 编程指南的 [PDF 版](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) 或 [网页版](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 是学习 CUDA 编程的绝佳参考。网上和 [NVIDIA 开发者站点](http://docs.nvidia.com/cuda/) 上有大量 CUDA 教程和 SDK 示例（直接 Google 即可）。特别推荐免费的 Udacity 课程 [Introduction to Parallel Programming in CUDA](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)。

[CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 中的表 21 是一个方便的参考，列出了本作业所用 NVIDIA T4 GPU 的每个线程块最大 CUDA 线程数、线程块尺寸、共享内存大小等限制。T4 GPU 的 CUDA compute capability 为 7.5。

C++ 问题（比如 _virtual_ 关键字是什么含义）推荐 [C++ Super-FAQ](https://isocpp.org/faq)，讲解详尽易懂（不像很多 C++ 资料），合著者之一就是 C++ 之父 Bjarne Stroustrup！

## Part 1：CUDA 热身 1：SAXPY（5 分）

作为 CUDA 编程练习，热身任务是用 CUDA 重新实现作业 1 中的 SAXPY 函数。
本部分起步代码在仓库的 `/saxpy` 目录。在 `/saxpy` 下执行 `make` 和 `./cudaSaxpy` 即可构建运行。

请完成 `saxpy.cu` 中 `saxpyCuda` 函数的实现。你需要分配设备全局内存数组，并在计算前把主机输入数组 `X`、`Y`、`result` 的内容拷贝到 CUDA 设备内存。CUDA 计算完成后，结果必须拷回主机内存。参见编程指南（网页版）3.2.2 节中 `cudaMemcpy` 的定义，或参考起步代码中指出的教程。

实现时，请在 `saxpyCuda` 中围绕 CUDA kernel 调用添加计时器。完成后程序应能测出两种时间：

- 起步代码自带的计时器测量**整个过程**：拷贝数据到 GPU、运行 kernel、把结果拷回 CPU。

- 你还应插入**只测 kernel 运行时间**的计时器（不含 CPU→GPU 数据传输和 GPU→CPU 结果回传的时间）。

**后一种计时要特别小心：** CUDA kernel 在 GPU 上的执行默认与 CPU 上的主应用线程是**异步**的。比如这样写：

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

你会测出一个快得离谱的 kernel 时间！（因为你只测到了 API 调用本身的开销，而不是 GPU 上真正执行计算的时间。）

因此，你需要在 kernel 调用之后加一句 `cudaDeviceSynchronize()`，等待 GPU 上所有 CUDA 工作完成。该调用在 GPU 完成此前所有 CUDA 工作后才返回。注意：`cudaMemcpy()` 之后**不需要** `cudaDeviceSynchronize()` 来确保传输完成，因为在我们使用的条件下 `cudaMemcpy()` 是同步的。（想深入了解可看[这份文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)。）

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

注意：在包含数据往返传输的测量中，最后的计时器之前（即在拷回 CPU 的 `cudaMemcpy()` 之后）**不需要** `cudaDeviceSynchronize()`，因为 `cudaMemcpy()` 直到拷贝完成才返回。

**问题 1.** 与 SAXPY 的 CPU 串行实现相比，你观察到的性能如何？（回顾作业 1 Program 5 中 saxpy 的结果）

**问题 2.** 对比并解释两组计时结果的差异（只测 kernel vs 测含数据往返的整个过程）。观察到的带宽值是否与机器各部件的标称带宽**大致**吻合？（请上网查 NVIDIA T4 GPU 的显存带宽。提示：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>。AWS 内存总线的预期带宽为 5.3 GB/s，与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的标称并不吻合。有多个因素导致达不到峰值带宽，包括 CPU 主板芯片组性能，以及作为传输源的主机内存是否为“锁页（pinned）”内存——锁页内存允许 GPU 不经虚拟地址翻译直接访问。感兴趣可看：<https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>）

## Part 2：CUDA 热身 2：并行前缀和（10 分）

熟悉了 CUDA 程序的基本结构后，第二个练习是并行实现 `find_repeats` 函数：给定整数数组 `A`，返回所有满足 `A[i] == A[i+1]` 的下标 `i` 组成的列表。

例如，给定数组 `{1,2,2,1,1,1,3,5,3,3}`，程序应输出 `{1,3,4,8}`。

#### 排他前缀和（Exclusive Prefix Sum）

我们要求你先实现并行排他前缀和，再用它实现 `find_repeats`。

排他前缀和：输入数组 `A`，输出数组 `output` 的每个下标 `i` 处是 `A[i]` 之前（不含 `A[i]`）所有元素的和。例如 `A={1,4,6,8,2}`，排他前缀和输出 `output={0,1,5,11,19}`。

下面的“类 C”代码是 scan 的迭代版本。伪代码中用 `parallel_for` 表示可并行的循环。这就是课上讲过的算法：<https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17>

```
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // 上扫（upsweep）阶段
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // 下扫（downsweep）阶段
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

请用这个算法实现 CUDA 版并行前缀和。你需要实现 `scan/scan.cu` 中的 `exclusive_scan` 函数，实现包含 host 端和 device 端代码。实现需要多次 CUDA kernel 启动（对应伪代码中每个 parallel_for 循环各一次）。

**注意：** 起步代码中，上述参考 scan 实现假设输入数组长度 `N` 是 2 的幂。`cudaScan` 函数里的解决办法是：分配 GPU 缓冲区时把长度向上取整到 2 的幂；但拷回时只从 GPU 缓冲区拷回 `N` 个元素到 CPU。这会让你的 CUDA 实现简单一些。

编译产出二进制 `cudaScan`。命令行用法：

```
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      运行指定功能。可选：scan, find_repeats（默认 scan）
  -i  --input <NAME>     输入类型。可选：ones, random（默认 random）
  -n  --arraysize <INT>  数组元素个数
  -t  --thrust           使用 Thrust 库实现
  -?  --help             显示帮助
```

#### 用前缀和实现 "Find Repeats"

写完 `exclusive_scan` 后，实现 `scan/scan.cu` 中的 `find_repeats` 函数。这需要再写一些 device 代码，并调用一次或多次 `exclusive_scan()`。你的代码应把重复元素下标列表写入给定的输出指针（device 内存），并返回输出列表的长度。

调用你的 `exclusive_scan` 时注意：`start` 数组的内容会被拷贝到 `output` 数组。另外，传给 `exclusive_scan` 的数组都假设已在 device 内存中。

**评分：** 我们会用随机输入数组测试正确性和性能。

下面给出一个 scan 分数表作参考，是一个简单 CUDA 实现在 K80 GPU 上的性能。检查你的 `scan` 和 `find_repeats` 的正确性与性能得分，分别运行 **`./checker.py scan`** 和 **`./checker.py find_repeats`**，会产出如下参考表；得分只看你代码的性能。要拿满分，你的代码性能必须在参考解的 20% 以内。

```
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

本部分重点是练习 CUDA 编程和数据并行思维，不是性能调优。拿满性能分不需要太多（其实不需要任何）调优，把算法伪代码直接移植到 CUDA 即可。但有一个坑：朴素的 scan 实现可能对伪代码中并行循环的每次迭代都启动 N 个 CUDA 线程，再在 kernel 里用条件判断哪些线程真正干活。这种解法性能不行！（想想 upsweep 最后一轮外层循环：只有两个线程有活干！）满分做法是：**只为最内层并行循环的每次有效迭代启动一个 CUDA 线程**。

**测试框架：** 默认情况下，测试框架使用伪随机生成、每次运行都相同的数组，方便调试。传 `-i random` 可使用真随机数组——评分时我们会用它。鼓励你自己构造输入来测试。也可以用 `-n <size>` 改变输入数组长度。

`--thrust` 参数会使用 [Thrust 库](http://thrust.github.io/) 的 [exclusive scan](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums) 实现。**谁能做出与 Thrust 有竞争力的实现，最多加 2 分额外分。**

## Part 3：一个简单的圆形渲染器（85 分）

正戏开始！

起步代码的 `/render` 目录包含一个绘制彩色圆形的渲染器实现。构建后运行 `./render -r cpuref rgb`，程序会输出 `output_0000.ppm`，内含三个圆。再运行 `./render -r cpuref snow`，输出图像是飘落的雪。PPM 图像在 OSX 上可用“预览”直接查看；Windows 可能需要下载查看器。

注意：也可以用 `-i` 选项把渲染输出送到显示器而不是文件（snow 场景能看到下雪动画）。但交互模式需要配置到本机的 X-windows 转发。（[参考一](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) 或 [参考二](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) 可能有帮助。）

起步代码包含两个版本的渲染器：`refRenderer.cpp` 中的串行单线程 C++ 参考实现，和 `cudaRenderer.cu` 中一个**不正确**的 CUDA 并行实现。

### 渲染器概述

建议先读 `refRenderer.cpp` 熟悉代码结构。`setup` 方法在渲染第一帧之前调用——在你的 CUDA 加速渲染器中，这个方法一般会包含所有初始化代码（分配缓冲区等）。`render` 每帧调用，负责把所有圆画进输出图像。另一个主要函数 `advanceAnimation` 也是每帧调用一次，更新圆的位置和速度——本作业**不需要**修改它。

渲染器的输入是圆的数组（3D 位置、速度、半径、颜色）。渲染每帧的基本串行算法是：

    清空图像
    for each circle
        更新位置和速度
    for each circle
        计算屏幕包围盒
        for 包围盒内所有像素
            计算像素中心点
            if 中心点在圆内
                计算圆在该点的颜色
                把该圆的贡献混合（blend）进该像素的图像

下图展示了用点在圆内测试计算圆-像素覆盖的基本算法。注意：只有当像素中心落在圆内时，圆才对该像素有颜色贡献。

![Point in circle test](handout/point_in_circle.jpg?raw=true "A simple algorithm for computing the contribution of a circle to the output image: All pixels within the circle's bounding box are tested for coverage. For each pixel in the bounding box, the pixel is considered to be covered by the circle if its center point (black dots) is contained within the circle. Pixel centers that are inside the circle are colored red. The circle's contribution to the image will be computed only for covered pixels.")

渲染器的一个重要细节是它绘制**半透明**圆。因此，一个像素的颜色不是某一个圆的颜色，而是所有覆盖该像素的半透明圆贡献混合的结果（注意上面伪代码里的“blend contribution”步骤）。渲染器用四元组（红 R、绿 G、蓝 B、不透明度 alpha，即 RGBA）表示圆的颜色。Alpha = 1 表示全不透明，Alpha = 0 表示全透明。要把颜色为 `(C_r, C_g, C_b, C_alpha)` 的半透明圆画到颜色为 `(P_r, P_g, P_b)` 的像素上，渲染器使用如下公式：

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

注意混合**不可交换**（X 叠在 Y 上和 Y 叠在 X 上看起来不同），所以渲染器必须按照应用提供圆的顺序来画（可以假设应用按深度顺序提供圆）。例如下面两张图：蓝圆叠在绿圆上、绿圆叠在红圆上。左图按正确顺序绘制，右图顺序错误，图像看起来就不对。

![Ordering](handout/order.jpg?raw=true "The renderer must be careful to generate output that is the same as what is generated when sequentially drawing all circles in the order provided by the application.")

### CUDA 渲染器

熟悉了参考代码中的圆形渲染算法后，现在研究 `cudaRenderer.cu` 中提供的 CUDA 实现。可以用 `--renderer cuda`（或 `-r cuda`）选项运行 CUDA 版渲染器。

提供的 CUDA 实现把计算按圆并行化：每个 CUDA 线程负责一个圆。虽然这个实现完整实现了圆形渲染的数学，但它包含几个将由你来修复的重大错误。具体来说：当前实现**没有保证图像更新的原子性**，也**没有保持图像更新所需的顺序**（顺序要求见下文）。

### 渲染器要求

你的并行 CUDA 渲染器必须维持两个不变量——串行实现天然满足它们：

1. **原子性（Atomicity）：** 所有图像更新操作必须是原子的。临界区包括：读取 4 个 32 位浮点值（像素的 rgba 颜色）、把当前圆的贡献与当前像素值混合、把像素颜色写回内存。
2. **顺序（Order）：** 渲染器对任一像素的更新必须按**圆的输入顺序**进行。即：若圆 1 和圆 2 都覆盖像素 P，则圆 1 对 P 的更新必须先于圆 2 对 P 的更新。如上所述，保持顺序才能正确渲染半透明圆（这对图形系统还有其他好处，好奇可以问 Kayvon）。**一个关键观察是：顺序的定义只约束对同一像素的更新顺序。** 如下图所示，不覆盖同一像素的圆之间没有顺序要求，可以独立处理。

![Dependencies](handout/dependencies.jpg?raw=true "The contributions of circles 1, 2, and 3 must be applied to overlapped pixels in the order the circles are provided to the renderer.")

由于提供的 CUDA 实现两条都不满足，在 rgb 和 circles 场景上运行 CUDA 渲染器就能看见后果：图像中出现水平条纹，如下图所示，且每帧条纹都会变化。

![Order_errors](handout/bug_example.jpg?raw=true "Errors in the output due to lack of atomicity in frame-buffer update (notice streaks in bottom of image).")

### 你要做什么

**你的任务是写一个尽可能快且正确的 CUDA 渲染器**。方法不限，但必须满足上面的原子性和顺序要求。两条都不满足的解在 Part 3 最多得 12 分——我们已经给了你一个这样的解！

建议的入手点：通读 `cudaRenderer.cu`，说服自己它**确实不**满足正确性要求。重点看 `CudaRenderer:render` 如何启动 CUDA kernel `kernelRenderCircles`（所有工作都在这个 kernel 里）。想直观感受违反两条要求的效果：`make` 编译后运行 `./render -r cuda rand10k`（1 万个圆，对应上图底行），与串行版 `./render -r cpuref rand10k` 的正确图像对比。

我们建议的步骤：

1. 先重写 CUDA 起步代码，使其并行运行时逻辑正确（推荐**不使用锁或同步**的方案）；
2. 然后找出你的方案有什么性能问题；
3. 这时真正的思考才开始……（提示：`circleBoxTest.cu_inl` 里给你的“圆与矩形相交测试”是你的好朋友，鼓励使用。）

`./render` 的命令行选项：

```
Usage: ./render [options] scenename
可选场景: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
          bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  选择渲染器：ref 或 cuda（默认 cuda）
  -s  --size  <INT>             图像尺寸：<INT>x<INT> 像素（默认 1024）
  -b  --bench <START:END>       渲染帧区间 [START,END)（默认 [0,1)）
  -c  --check                   用 CPU 参考输出校验 CUDA 输出的正确性
  -i  --interactive             渲染到交互式显示器
  -f  --file  <FILENAME>        输出文件名（FILENAME_xxxx.ppm）（默认 output）
  -?  --help                    显示帮助
```

**检查器：** `render` 提供了方便的 `--check` 选项检测正确性。它会同时跑串行 CPU 参考渲染器和你的 CUDA 渲染器，比较两张图像。你的 CUDA 实现的耗时也会打印出来。

我们提供共八个圆形数据集用于评分。要拿满分，你的代码必须通过所有正确性测试。检查正确性和性能得分：在 `/render` 目录运行 **`./checker.py`**（注意 .py 后缀）。对起步代码运行它会打印类似下表的结果：

```
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：某些运行中你**可能**在个别场景拿到分——因为起步渲染器的运行有不确定性，偶尔可能碰巧正确。但这不改变当前 CUDA 渲染器总体上不正确的事实。

“Ref time” 是我们参考解在你当前机器上的性能（由提供的 `render_ref` 可执行文件测得）。“Your time” 是你当前 CUDA 渲染器的性能，`(F)` 表示结果不正确。你的成绩取决于你的实现与参考实现的性能对比（见评分细则）。

除代码外，还需提交一份清晰、高层次的实现说明，并简述你是如何得到这个方案的。请具体说明过程中尝试过的方法，以及你是如何决定优化方向的（例如：你做了哪些测量来指导优化？）。

报告中应包含以下内容：

1. 报告顶部写上两位搭档的姓名和 SUNet id。
2. 附上你的解的分数表，并注明运行所用的机器。
3. 描述你如何分解问题、如何把工作分配给 CUDA 线程块和线程（甚至 warp）。
4. 描述你的解中同步发生在哪里。
5. 你为降低通信开销（同步或主存带宽需求）采取了哪些措施（如果有）？
6. 简述你如何得到最终方案：过程中试过哪些其他方法？它们有什么问题？

### 评分细则

- 作业报告 18 分。
- 并行前缀和实现 10 分。
- 渲染器实现 72 分，每个场景 9 分，细分如下：
  - 每个场景 2 分正确性分。我们只用 256 的倍数的图像尺寸测试。
  - 每个场景 7 分性能分（正确才可得）。性能相对提供的参考渲染器 T<sub>ref</sub> 评分：
    - 时间 T 达到 10 倍 T<sub>ref</sub> 的解不得性能分；
    - T <= 1.20 * T<sub>ref</sub> 得满性能分；
    - 介于两者之间（1.20 T<sub>ref</sub> < T < 10 T<sub>ref</sub>）时，性能分按 `7 * T_ref / T` 在 1~7 分区间计算。

- 显著超过要求性能的解最多加 5 分额外分（由教师酌情），报告必须清楚完整地解释你的方法。
- 最多 5 分额外分给高质量的纯 CPU 并行渲染器：充分利用所有核及各核的 SIMD 向量单元。工具不限（SIMD intrinsics、ISPC、pthreads 皆可）。要拿这部分分数，你需要分析 GPU 与 CPU 方案的性能，并讨论两者实现选择差异的原因。

本项目总分构成：

- part 1（5 分）
- part 2（10 分）
- part 3 报告（13 分）
- part 3 实现（72 分）
- 可能的**额外**分（最多 10 分）

## 作业提示与建议

以下是往年积累的一组提示。渲染器有多种实现方式，并非所有提示都适用于你的方案。

- 本作业有两个潜在的并行维度：一是**按像素并行**，二是**按圆并行**（前提是重叠的圆之间保持顺序要求）。解通常需要两者结合，在计算的不同阶段各取所需。
- `circleBoxTest.cu_inl` 中提供的“圆与矩形相交测试”是你的好朋友，鼓励使用。
- `exclusiveScan.cu_inl` 中提供的共享内存前缀和在本作业中可能有用（不是所有方案都会用到）。前缀和的简介见[这里](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums)。我们提供的是**2 的幂长度**数组上的共享内存排他前缀和实现。**该代码不支持非 2 的幂输入，且要求线程块中的线程数等于数组长度。请阅读代码中的注释。**
- 看一下被调用的 `shadePixel` 方法：注意它做了很多次全局内存操作来更新一个像素的颜色。明智的做法是在 `kernelRenderCircles` 里改用**局部累加器**——在寄存器里累加像素值，得到最终值后只对全局内存做**一次**写入。
- 实现中允许使用 [Thrust 库](http://thrust.github.io/)，但达到优化版参考实现的性能并不需要它。有一种流行解法使用我们给的共享内存前缀和实现，另一种流行解法使用 Thrust 库的前缀和例程。两种都是可行的策略。
- 渲染器里存在数据复用吗？可以怎样利用这种复用？
- CUDA 语言没有原语能原子地完成整个图像更新逻辑，你打算如何保证原子性？用全局内存原子操作构造锁是一种方案，但记住：即使更新是原子的，更新也必须按规定顺序进行。**建议你先思考如何在并行方案中保证顺序，然后再考虑原子性问题（到那时它可能已经不存在了）。**
- 对于圆数量特别大的测试（`rand1M` 和 `micro2M`），在全局内存中分配临时结构要小心：分配过多会耗尽设备内存。如果你没有检查 `cudaMalloc` 返回的 `cudaError_t`，程序仍会继续运行，但你不知道显存已耗尽——然后你会因为临时结构没建成而挂在正确性检查上。所以我们建议用下面的 CUDA API 包装宏包住 `cudaMalloc` 调用，显存耗尽时直接报错。
- 如果有空闲时间，可以好玩地做些自己的场景！

### 捕获 CUDA 错误

默认情况下，数组越界、显存分配过多等错误 CUDA **不会**主动告知你——它只会静默失败并返回错误码。可以用下面的宏（可自行修改）包装 CUDA 调用：

```
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

注意：代码正确后可以 undefine DEBUG 关掉错误检查以提升性能。

包装 CUDA API 调用的写法：

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

注意 kernel 启动不能直接包，它们的错误会在你包装的下一个 CUDA 调用处被捕获：

```
kernel<<<1,1>>>(a); // 假设 kernel 出错了！
cudaCheckError( cudaDeviceSynchronize() ); // 错误在这一行打印出来
```

所有 CUDA API 函数都可以包：`cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset` 等等。

**重要：** 如果之前某个 CUDA 函数出错但没被捕获，该错误会出现在下一次错误检查里——即使那次包的是另一个函数。例如：

```
...
line 742: cudaMalloc(&a, -1); // 执行，然后继续
line 743: cudaCheckError(cudaMemcpy(a,b)); // 打印 "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此调试时建议包装**所有** CUDA API 调用（至少你自己写的代码里）。

（致谢：改编自 [Stack Overflow 帖子](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)）

## 提交说明

请通过 Gradescope 提交。如果与搭档合作，记得在 Gradescope 上标记搭档。

1. **报告请提交为 `writeup.pdf` 文件。**
2. **请运行 `sh create_submission.sh` 生成提交用的 zip。** 注意该脚本会在代码目录执行 make clean，所以之后要重新 make 才能运行代码。如果脚本报 'Permission denied'，先运行 `chmod +x create_submission.sh` 再重试。

我们的评分脚本会重新运行 checker 代码，以核实你的分数与 `writeup.pdf` 中提交的一致。我们也可能在其他数据集上运行你的代码，进一步检查正确性。
