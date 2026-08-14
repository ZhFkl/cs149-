# 作业 4：为机器学习加速器编程（Programming a Machine Learning Accelerator）

**截止日期：11 月 13 日（周四）23:59**

**总分 100 分**

## 概述（Overview）

在本作业中，你将学习如何为 [AWS Trainium2](https://aws.amazon.com/ai/machine-learning/trainium/) 架构实现并优化 kernel。该架构包含多个面向张量的加速处理引擎，以及由软件管理的片上存储，后者为这些引擎提供对数据的高带宽访问。

本作业分为两部分。在 Part 1 中，你将通过研究几个简单的向量加法 kernel、并亲自编写一个矩阵转置 kernel，来熟悉 Trainium 架构和数据搬运模式。在 Part 2 中，你将在 Trainium2 上实现一个融合的卷积 + 最大池化层。

总体而言，本作业将：

1) 让你获得在加速器上进行张量处理底层细节和管理片上 SRAM 的实践经验。

2) 向你展示循环分块、循环融合等保持局部性的关键优化的价值。

## 环境配置（Environment Setup）

你将在配备 Trainium 加速器的 AWS 虚拟机上编写并测试代码。请按照 [cloud_readme.md](cloud_readme.md) 的说明搭建用于运行本作业的机器。

登录 AWS 机器后，使用以下命令从课程 Github 下载作业起步代码：

`git clone https://github.com/stanford-cs149/asst4-trainium2`

下载作业 4 的仓库后，进入 `asst4-trainium2` 目录并**运行我们提供的安装脚本**：
```
cd asst4-trainium2
source install.sh
```
安装脚本会激活一个包含本作业全部所需依赖的 Python [虚拟环境](https://builtin.com/data-science/python-virtual-environment)。它还会修改你的 `~/.bashrc` 文件，使虚拟环境在你以后登录机器时自动激活。最后，脚本会配置好你的 InfluxDB 凭证，以便你使用 `neuron-profile`。

## Part 0：熟悉 Trainium 与 Neuron Core 架构

### Trainium 架构概述（Trainium Architecture Overview）

首先，让我们来认识一下 Trainium。

本作业使用的 `Trn2.3xlarge` 实例配备单个 Trainium device，它由八个 NeuronCore 组成。如下图所示，每个核心都配备自己专用的 HBM（高带宽内存）。每个 NeuronCore 可以看作一个独立的处理单元，包含自己的片上存储以及一组专用计算引擎，用于执行 128x128 矩阵运算（张量引擎，Tensor Engine）、128 宽向量运算（向量引擎，Vector Engine）等。虽然每个 Trainium device 有八个 NeuronCore，但在本作业中，我们编写的 kernel 只在单个 NeuronCore 上执行。

<p align="center">
  <img src="handout/trainium_chip.png" width=45% height=45%>
  <img src="handout/neuroncore_v3.png" width=30% height=30%>
</p>

关于 NeuronCore 中存在的四种不同计算引擎的更多细节，请参见[这里](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html)。

### Trainium 内存层次（Trainium Memory Hierarchy）

在作业 3 中，关键概念之一是学习 CUDA 呈现的 GPU 内存层次：host 主内存、GPU device 全局内存、每个线程块的共享内存、以及每个 CUDA 线程私有的内存。在 Trainium 上，内存层次由四层组成：**host 内存（DRAM）**、**device 内存（HBM）**，以及两种快速的片上内存：**SBUF（State Buffer，状态缓冲区）**和 **PSUM（Partial Sum Buffer，部分和缓冲区）**。在本作业中，我们编写的 kernel 只面向 device/片上内存，因此可以忽略 DRAM（它位于 Trainium device 之外），专注于 HBM、SBUF 和 PSUM。

<p align="center">
  <img src="handout/memory_hierarchy.png" width=80% height=80%>
</p>

* __HBM__ 是位于 Trainium device 上的高带宽内存。HBM 是 device 的主内存，提供大容量存储（96 GiB）。在 kernel 之外创建的大多数数据类型（例如 NumPy 数组）默认分配在 HBM 中。
* __SBUF__ 是 NeuronCore 上的片上存储。相比之下，SBUF 比 HBM 小得多（28 MiB），但带宽高得多（约为 HBM 的 ~20 倍）。程序员必须显式地把数据移入、移出 SBUF，才能用 NeuronCore 执行计算。
* __PSUM__ 是一个小而专用的内存块（2 MiB），专门用于存放张量引擎产生的矩阵乘法结果。

<p align="center">
  <img src="handout/neuron_core.png" width=40% height=40%>
</p>

回想一下，在配备传统数据缓存（cache）的系统中，哪些来自片外内存的数据会被复制并存入片上存储，是由缓存决定的（基于缓存的组织结构和替换策略）。软件加载某个内存地址上的数据，硬件负责从内存取回该数据，并管理缓存中存放哪些数据以便将来高效访问。换句话说，从软件正确性的角度看，缓存并不存在——它只是硬件实现细节。

与此相反，NeuronCore 可用的内存是*由软件管理*的。这意味着软件必须使用数据搬运命令显式地把数据移入、移出这些内存。要么由程序员在程序中显式描述数据搬运，要么由 NKI 编译器分析应用程序并生成相应的数据搬运操作。高效使用 NeuronCore 架构所面临的最大挑战之一，就涉及高效地编排数据在机器中的流动。

## Part 1：通过向量加法和矩阵转置学习 Neuron Kernel Interface（30 分）

在本节中，我们通过一个把两个向量的元素相加的应用程序的几种不同实现，来介绍 Trainium 编程模型的基础知识。随后我们将编写一个简单的 kernel 来转置一个二维矩阵。

相应代码组织在 `/part1` 目录中。具体来说，这里讨论的向量加法 kernel 可以在 `kernels.py` 中找到。此外，我们还提供了一个脚本 `run_benchmark.py`，它提供了便捷的命令行接口，可以用不同的向量大小执行这些 kernel。该脚本还包含一个用于采集 profiling 指标的可选标志。

```
usage: run_benchmark.py [-h] --kernel {naive,tiled,stream,transpose} -n N [-m M] [--profile_name PROFILE_NAME]

options:
  -h, --help            show this help message and exit
  --kernel {naive,tiled,stream,transpose}
  -n N
  -m M
  --profile_name PROFILE_NAME
                        Name used to save .NEFF and .NTFF files
```

### NKI 编程模型：

Neuron Kernel Interface（NKI）是用于开发在 Trainium device 上运行的 kernel 的语言和编译器。NKI kernel 用 Python 编写，并使用三类 NKI 操作：
1. **加载数据**：从 HBM 到片上 SBUF。
2. **计算**：在 NeuronCore 计算引擎上执行。
3. **存储输出**：从 SBUF 写回 HBM。

例如，下面的 kernel 定义了如何用 NKI 执行向量加法。注意，`@nki.jit` 是一个 Python 装饰器，表示函数应被编译为在 NeuronDevice 上运行——很像 CUDA C++ 中的 `__global__` 函数修饰符指定函数被编译为 device 端函数并在 GPU 上运行。

与 CUDA kernel 的参数是 CUDA device 全局内存中的数组类似，用 `@nki.jit` 装饰的 Python 函数的参数是驻留在 NeuronCore 可访问的 HBM 中的张量。`@nki.compiler.skip_middle_end_transformations` 装饰器会禁用一些可能以意想不到的方式变换 kernel 的编译器优化，这会让调试更容易。

在下面的代码中，假设 `a_vec` 和 `b_vec` 是 HBM 中长度为 128 的向量。（该代码对大于 128 的向量不适用，我们很快会解释原因。）
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Allocate space for the input vectors in SBUF and copy them from HBM
    a_sbuf = nl.ndarray(shape=(a_vec.shape[0], 1), dtype=a_vec.dtype, buffer=nl.sbuf)
    b_sbuf = nl.ndarray(shape=(b_vec.shape[0], 1), dtype=b_vec.dtype, buffer=nl.sbuf)
    
    nisa.dma_copy(src=a_vec, dst=a_sbuf)
    nisa.dma_copy(src=b_vec, dst=b_sbuf)

    # Add the input vectors
    res = nisa.tensor_scalar(a_sbuf, nl.add, b_sbuf)

    # Store the result into HBM
    nisa.dma_copy(src=res, dst=out)

    return out
```

在上面的代码中……

- `a_vec` 和 `b_vec` 是在 kernel 之外创建、驻留在 HBM 中的 NumPy 数组。
- `a_sbuf` 和 `b_sbuf` 是在 SBUF 中显式分配的数组，形状和 dtype 与 `a_vec`、`b_vec` 相同。
- `nisa.tensor_scalar(..., nl.add, ...)` 使用向量引擎执行向量加法。`tensor_scalar` 这个签名意味着第二个操作数应为一个向量（即形状为 (N, 1)）或一个常量标量，这使它比通用的 `tensor_tensor` 操作略快一些。
- `nisa.dma_copy` 在 HBM 和 SBUF 之间搬运相关数据（概念上类似于 NVIDIA GPU 上的 `cudaMemcpyAsync`）。

<p align="center">
  <img src="handout/sbuf_layout.png" width=60% height=60%>
</p>

**阅读上面的代码时，请注意 NKI 操作作用于张量，而不是标量值。** 具体来说，片上内存 SBUF 和 PSUM 存储的数据按二维内存数组组织。二维数组的第一维称为“分区维度”（partition dimension）`P`，第二维称为“自由维度”（free dimension）`F`。NeuronCore 能够沿分区维度并行地加载和处理数据，*但该架构还有一个限制：分区维度的大小不能超过 128。*
换句话说，把张量从 HBM 加载到 SBUF 时，张量的分区维度最大为 128。自由维度的限制我们稍后再讨论。

因此，在上面的代码中，由于 `a_vec` 和 `b_vec` 是一维向量，它们唯一的维度就是分区维度，因此其大小被限制为 128 个元素。换句话说，这段代码只适用于大小不超过 128 的向量。

### Step 1：将向量分块以在 128 条计算通道上并行（6 分）

要让代码适用于大小超过 128 的向量，我们需要分块（原张量的子集）加载向量。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    # TODO: You should modify this variable for Step 1
    ROW_CHUNK = 1

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range(M // ROW_CHUNK):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        nisa.dma_copy(src=a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=a_tile)
        nisa.dma_copy(src=b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=b_tile)

        # Add the row chunks together
        res = nisa.tensor_scalar(a_tile, nl.add, b_tile)

        # Store the result chunk into HBM
        nisa.dma_copy(src=res, dst=out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
    
    return out
```

上面的例子把向量的行拆成单元素块（块大小为向量的 1 个元素——是的，这很低效，我们马上会回到这个问题）。这是通过标准的 Python 切片语法 `Tensor[Index:Index:...]` 对向量进行索引实现的。关于 NKI 中张量索引的更多细节，请参见[这里](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing)。

上面代码中使用的 [affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) 为循环迭代器生成一个数字序列，类似于 Python 的 `range` 函数，但它要求各次迭代之间没有循环携带依赖（loop-carried dependency）。对于存在循环携带依赖的情形，NKI 还提供了 [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html)。

通常，`affine_range` 让 NKI 编译器能更激进地优化循环迭代，从而增强各计算引擎之间的流水线化。不过，由于我们为了透明性/可复现性禁用了编译器优化，这两种构造实际上效果相同。

**你需要做的：**
1. 在向量大小为 25600 时运行上述 *row_chunk = 1* 的 `vector_add_tiled` 实现（*这可能需要几分钟*）。可以使用以下命令：

   ```
   python run_benchmark.py --kernel tiled -n 25600
   ```

   执行时间是多少微秒（μs）？

2. 回想一下，NeuronDevice 上一次能加载的最大分区大小（行数）是 128。在 `kernels.py` 中修改 `vector_add_tiled`，使其使用 *row_chunk = 128*。记录 *row_chunk = 128* 的 `vector_add_tiled` 在向量大小为 25600 时的执行时间（微秒 μs）。在向量大小为 25600 时，*row_chunk = 128* 比 *row_chunk = 1* 快多少？你认为为什么更快？（*提示：* 你应该把执行过程想象成从 HBM 并行加载 `ROW_CHUNK` 个元素，然后在 SBUF 中的向量上执行一次 `ROW_CHUNK` 宽的向量加法。）

3. 尝试在向量大小为 25600、*row_chunk = 256* 时运行 `vector_add_tiled`。你应该会看到报错。用一句话解释为什么尝试运行 *row_chunk = 256* 时会报错。

### Step 2a：改进数据流式传输（4 分）

到目前为止，我们已经利用了向量引擎能够用全部 128 条向量通道并行执行计算这一事实，每条通道从单个 SBUF/PSUM 分区流入/流出单个元素。

然而，我们可以通过沿自由维度传输更多元素来进一步提升性能。为此，让我们更深入地思考直接内存访问（DMA）传输。你应该把一次 DMA 传输（即一次 `nisa.dma_copy` 调用）看作一个异步操作，它把一块数据从 HBM 搬到 SBUF，或反向搬运。

每个 NeuronCore 有 16 个 DMA 引擎，它们都可以并行处理不同的数据传输操作。但要注意的是，建立一次 DMA 传输并给 DMA 引擎分配任务是有开销的。为了降低这种建立开销，高效的实现应力求在每次传输中搬运大量数据，以摊销 DMA 传输开销。

虽然 SBUF 张量的第一维（分区维度）不能大于 128，但单条 SBUF 向量指令的第二维最大可达 64K 个元素。这意味着可以用一条指令把 128 * 64k = 8192k 个元素从 HBM 加载到 SBUF。此外，我们可以在一条 `nisa.tensor_tensor` 指令中对两个 8192k 元素的 SBUF tile 执行向量加法。因此，与其对向量的每个 128 元素块都执行一次 `nisa.dma_copy`，我们不如让每次 DMA 传输请求搬运多个 128 行的块。这种精简的做法让我们能够摊销传输数据所需的建立时间。

为了降低 DMA 传输开销，我们需要把向量 reshape 成二维 tile，而不是线性化的数组。在作业 3 中，我们把 CUDA 线程块划分到整张图像上，为了把 CUDA 线程映射到图像像素，我们通过计算线程的全局线性索引把网格展平。你可以把 NeuronCore 上的 reshape 过程看作它的逆过程：目标是把一维向量变成稠密的二维矩阵。NumPy 自带一个 [reshape 函数](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)，可以把数组 reshape 成你选择的形状。

<p align="center">
  <img src="handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="handout/reshaped_DMA.png" width=48% height=48%>
</p>


看一下 `vector_add_stream`，它扩展了 `vector_add_tiled`，使 DMA 传输次数更少：
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # TODO: You should modify this variable for Step 2a
    FREE_DIM = 2

    # The maximum size of our Partition Dimension
    PARTITION_DIM = 128

    a_vec_re = a_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    b_vec_re = b_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # Loop over the total number of tiles
    for m in nl.affine_range(M // (PARTITION_DIM * FREE_DIM)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=b_vec.dtype, buffer=nl.sbuf)

        # Load the input tiles
        nisa.dma_copy(src=a_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=a_tile)
        nisa.dma_copy(src=b_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=b_tile)

        # Add the tiles together. Note that we must switch to tensor_tensor instead of tensor_scalar
        res = nisa.tensor_tensor(a_tile, b_tile, op=nl.add)

        # Store the result tile into HBM
        nisa.dma_copy(src=res, dst=out[:, m * FREE_DIM : (m + 1) * FREE_DIM])

    # Reshape the output vector into its original shape
    out = out.reshape(a_vec.shape)

    return out
```

**你需要做的：**
1. 运行上述 *FREE_DIM = 2* 的 `vector_add_stream` 实现。在向量大小为 25600 时运行用了多少微秒（μs）？与 Step 1 中 *row_chunk = 128* 的 `vector_add_tiled` 相比快了多少？
2. 当前的 `vector_add_stream` 实现略微减少了 DMA 传输次数，但 DMA 传输次数还可以进一步减少。在 `kernels.py` 中修改 `vector_add_stream` 的 *FREE_DIM* 值，使向量大小为 25600 时的 DMA 传输次数尽可能少。

   你选择的 *FREE_DIM* 值是多少？在这个 *FREE_DIM* 值下，向量大小为 25600 时的执行时间是多少微秒（μs）？
   
   用你选择的 *FREE_DIM* 值的 `vector_add_stream` 比 *FREE_DIM = 2* 时快多少？比 *row_chunk = 128* 的 `vector_add_tiled` 快多少？

### Step 2b：学习使用 Neuron-Profile（5 分）

选择 tile 的自由维度大小时存在一个权衡：
1. tile 太小会暴露显著的指令开销，导致引擎执行低效。
2. tile 太大往往会导致引擎之间的流水线低效，并且在存在数据复用的情况下给 SBUF 造成很高的内存压力（“内存压力”指 SBUF 可能会被填满）。

目前，我们已经探索了把 tile 大小增加到最大值的好处，以便摊销指令开销以及 DMA 传输的建立/拆除开销。现在，我们将探索为什么把自由维度设得尽可能大并不总是最佳方案。

本任务需要使用 NeuronDevice 的 profiling 工具：`neuron-profile`，它可以对在 NeuronCore 上运行的应用程序的性能提供详细分析。要运行 profiling 工具，你必须确保已经按照[环境配置](#environment-setup)中的说明运行过安装脚本，并且在 ssh 登录机器时转发了 3001 和 8086 端口。重申一下后者，你应当运行的命令是：

 `ssh -i path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086`
 
 关于为什么需要这样做的更多细节，请参见 [cloud_readme.md](/cloud_readme.md)。

**你需要做的：**
1.  这次，我们把向量大小扩大 10 倍，即不再加 25600 个元素，而是加 256000 个元素。这能让我们看到 tile 过大带来的权衡。  

    首先，在 `vector_add_stream` 中设置 *FREE_DIM = 2000*。现在，像之前的步骤一样执行我们的 kernel，但这次我们要把编译后的 kernel 
    保存到一个 **.neff** 文件，把 kernel 执行轨迹保存到一个 **.ntff** trace 文件。让我们在向量大小为 256000 时运行 `vector_add_stream`，并用以下命令把编译后的 kernel 和 
    轨迹保存为前缀为 `stream_256k_fd2k` 的文件：

    ```
    python run_benchmark.py --kernel stream -n 256000 --profile_name stream_256k_fd2k
    ```

    你应该生成了两个文件：***stream_256k_fd2k.neff*** 和 ***stream_256k_fd2k.ntff***。（你可能会在 stdout 中看到一条写着 "hw profiler overview not found" 的报错——这可以安全忽略，不用担心。）
    
    现在，用类似的工作流程，在向量大小为 256000、*FREE_DIM = 1000* 时运行 `vector_add_stream`，并把编译后的 kernel 和轨迹保存为前缀为 
    `stream_256k_fd1k` 的文件。
2.  这些生成的文件让我们可以用 `neuron-profile` 工具收集 kernel 执行指标。这些 profiling 指标对分析你的 
    kernel 的性能非常有用。运行以下命令，查看 *FREE_DIM = 2000* 时 `vector_add_stream` 的执行指标简要摘要：

    ```
    neuron-profile view --output-format summary-text -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
    ```

    你会看到一个由按字母顺序排列的各项执行指标组成的摘要输出。我们来看两个具体的指标： 
    
     * **dma_transfer_count**：DMA 传输次数
     * **total_time**：kernel 执行时间（秒）

    *FREE_DIM = 2000* 时 kernel 执行时间是多少秒？进行了多少次 DMA 传输？
    
    用与之前相同的工作流程，查看 *FREE_DIM = 1000* 时的执行指标摘要。
    
    *FREE_DIM = 1000* 时 kernel 执行时间是多少秒？进行了多少次 DMA 传输？

3. 尽管 *FREE_DIM = 1000* 的 kernel 的 DMA 传输次数更多，但它更快！我们来分析一下原因。
  
   我们可以使用 `neuron-profile` 的 GUI 功能更深入地研究 kernel 执行指标。让我们通过运行 
   以下命令，启动 *FREE_DIM = 2000* 的 `vector_add_stream` 的 GUI：

   ```
   neuron-profile view -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   运行命令后，你会看到类似下面的输出：

   `View profile at http://localhost:3001/profile/...`

   把这个 *http* 链接粘贴到你选择的浏览器中，即可查看更深入的 profiler 分析内容。（页面顶部出现的任何警告都可以忽略。）
   
> [!NOTE]
> 只有在 ssh 登录机器时正确转发了 3001 和 8086 端口，你才能查看该页面。

   你应该能看到一张由 profiler 生成的图，描绘了随时间推移发射到不同引擎的指令。
   
   为了便于我们的观察，请到底部的 `View Settings` 并进行以下操作：
   * 把 `Instructions color group` 改为 `Instruction Type`
   * 在 `Timeline display options` 下关闭 `Show individual NeuronCore layout`
   * 在 `DMA display options` 下关闭 `Show expanded DMA`
   * 点击最底部的 `Save`。
   
   完成这些步骤后，profiler 图应如下所示：

   ![Profiler GUI Example](handout/profiler_gui.png)
   
   你还可以把鼠标悬停在图中的各种事件上查看更多信息。试着悬停在以下几类事件上：
   
   * **DMA-E79**：显示 DMA 引擎在相应缓冲区之间搬运输入和输出数据（数一数指令数量——它与 `nisa.dma_copy` 的预期调用次数一致吗？）
   * **VectorE**：显示向量引擎通过 `nisa.tensor_tensor` 把两个输入向量相加（这部分应以绿色高亮显示）
   * **Pending DMA Count**：显示随时间变化的待处理 DMA 传输数量
   * **DMA Throughput**：显示随时间变化的 device 带宽利用率

   现在，在你的终端中按 `ctrl-c` 退出当前的 `neuron-profile view`。注意，你仍然可以在浏览器中查看 *FREE_DIM = 2000* 的 `vector_add_stream` 的 GUI 分析结果，因为它们已被临时存储在数据库中。按照相同的工作流程，启动 *FREE_DIM = 1000* 的 `vector_add_stream` 的 GUI 分析。

4. 在分析了 *FREE_DIM = 2000* 和 *FREE_DIM = 1000* 两种情况下 `vector_add_stream` 的 GUI 分析图之后，简要解释为什么 FREE_DIM = 1000 尽管需要更多次 DMA 传输，执行时间却比 
   FREE_DIM = 2000 更快（*提示：* 流水线）。

   也欢迎你随意尝试 `neuron-profile` GUI 中的各种功能。你可能还想看看位于底部工具栏的 `Summary` 标签页。该标签页显示的内容与我们在问题 2 中运行  
   `neuron-profile view --output-format summary-text ...` 时看到的执行指标简要摘要相同。想进一步了解 `neuron-profile` 的功能可查阅[用户指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)，想了解 NKI kernel 中值得关注的性能指标可查阅 [NKI 性能指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html)。

### Step 3：矩阵转置（15 分 = 10 分编程 + 5 分报告（+ 1 分附加分））
### NeuronCore 上的矩阵运算（Matrix Operations on a NeuronCore）
开始之前，我们先演示如何在 NeuronCore 上执行矩阵运算。如前所述，NeuronCore 配备了多种计算引擎，各自针对特定类型的算术运算进行了优化。Trainium 上的张量引擎专为加速矩阵运算而设计，例如矩阵乘法和矩阵转置。 

<p align="center">
  <img src="handout/tensor_engine.png" width=60% height=60%>
</p>

上图描绘了张量引擎的架构。张量引擎围绕一个 128x128 的[脉动处理阵列](https://gfxcourses.stanford.edu/cs149/fall25/lecture/proghardware/slide_10)构建，它从 SBUF（片上存储）流入矩阵输入数据，并把输出写入 PSUM（也是片上存储）。与 SBUF 一样，PSUM 是快速的片上内存，但它比 SBUF 小得多（2MiB vs 28 MiB），且专门用于存储张量引擎计算出的矩阵乘法结果。张量引擎能够对 PSUM 中的每个地址执行读-加-写。因此，当以分块方式执行大型矩阵乘法、并把每次矩阵乘法的结果累加到同一个输出 tile 中时，PSUM 就很有用。

### 编写 kernel（Writing the kernel）
在这里，你将先尝试用张量引擎编写自己的转置矩阵小 kernel，然后再进入 Part 2 中涉及真正矩阵乘法的更复杂 kernel。看一下 `kernels.py` 中的起步代码。你的 kernel 应接受一个形状为 (M, N) 的二维张量作为输入，并返回一个形状为 (N, M) 的二维张量。对 M 和 N 唯一的限制是两者都能被 128（最大分区维度）整除。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    tile_dim = 128

    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"

    # TODO: Your implementation here. The only compute instruction you should use is `nisa.nc_transpose`.

    return out
```

要真正执行转置，你必须调用 [nisa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html#nki.isa.nc_transpose)，它是一条内置指令，使用张量引擎转置最大 128x128 的 tile，并把结果存入 PSUM。你**不允许**使用其他计算指令，包括 `nisa.dma_tranpose` 或 `nl.transpose`。（内存指令，包括 `nisa.dma_copy` 和 `nl.ndarray`，当然是允许的。）

由于你要转置的矩阵远大于 128x128，你的 kernel 需要管理数据 tile 在 HBM/SBUF 之间的来回搬运。回顾一下之前的向量加法 kernel 如何分配和搬运数据可能会有帮助。

> [!TIP]
> `nisa.dma_copy` 只适用于 SBUF/HBM 中的张量。由于 `nisa.nc_transpose` 的输出是 PSUM tile，你需要先把它拷贝到 SBUF。你可能会发现 [`nisa.tensor_copy`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html#nki.isa.tensor_copy) 对此很有用。

**你需要做的：**
1.  用你的实现补全 kernel。然后运行以下命令，在 1024x1024 矩阵上测试它：
    ```
    python run_benchmark.py --kernel transpose -n 1024
    ```
    并记录执行时间（微秒 μs）。
2. 在不使用 profiler 的情况下，你认为你的 kernel 是内存受限（memory-bound）还是计算受限（compute-bound）？解释你的答案。然后，用与 `vector_add_stream` 相同的方式 profile 你的代码来确认。（你可以附上截图，但请提供书面描述，说明它如何验证你的答案。）
3.  **（附加分，1 分）** 优化你的实现以最小化延迟。要获得该分数，你应能在 4096x4096 转置上达到 <700 μs。请确保在*不*传 `--profile_name` 的情况下测量延迟（profiler 会改变执行时间）。

    在这一部分，可以随意尝试 `nisa.nc_transpose` 之外的其他 API。同时请提交一份简短报告，说明你是如何识别性能瓶颈并解决它们的。

## Part 2：实现融合的卷积 - 最大池化层（70 分）

现在你已经学会了如何在 NeuronCore 上高效搬运数据，是时候自己编写一个真正的 Trainium kernel 了。在本节中，你的任务是实现一个同时执行卷积和一种称为“最大池化”（max pooling）的操作的 kernel。正如我们在课堂上讨论的，这两个操作是现代卷积神经网络（CNN）的基本组成部分，CNN 被广泛用于计算机视觉任务。一个重要的细节是：你对这两个操作的实现将是“融合”的，也就是说你将在 Trainium 上完成计算，而不把中间值转储到片外 HBM。 

### 一个 NKI 矩阵乘法 kernel（An NKI Matrix Multiplication Kernel）

回想一下，向量引擎能够处理大小为 (128, 64k) 的 SBUF tile。然而，张量引擎有其独特的 SBUF tile 大小约束，与向量引擎的不同。假设我们想让张量引擎执行矩阵乘法 C = A x B，其中 A 和 B 位于 SBUF 中，结果 C 存入 PSUM。Trainium 施加了以下约束。 
  - 矩阵 A——左侧 tile——不能大于 (128, 128)
  - 矩阵 B——右侧 tile——不能大于 (128, 512)。
  - PSUM 中的输出 tile C 被限制为 (128, 512)。

鉴于张量引擎的这些约束，要在 Trainium 上实现任意维度的矩阵乘法，需要把计算分块，使其以固定大小 tile 上的一系列矩阵乘法的形式执行。（这与 Part 1 中向量加法为处理大输入向量而进行分块的方式类似。）下面的例子改自 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials)，演示了如何用分块方式实现矩阵乘法，其中 tile 的大小满足 Trainium 张量引擎的 tile 大小约束。注意：代码清单之后附有对代码的说明。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
```

我们来拆解这个计算矩阵乘法 `result = lhsT x rhs` 的 kernel 的各个组成部分。

  - 输入张量：
      - `lhsT` 是左侧矩阵。但该矩阵以__转置格式__提供，形状为 `[K,M]`，其中 `K` 和 `M` 都是 128 的倍数。
      - `rhs` 是右侧矩阵，形状为 `[K,N]`，其中 `K` 是 128 的倍数，`N` 是 512 的倍数。
      - `result` 是形状为 `[M,N]` 的输出矩阵。
      - 在矩阵乘法中，**收缩维度**（contraction dimension）指左侧矩阵的列维度和右侧矩阵的行维度。例如，设 
        我们有如下矩阵乘法：`A x B = C`。矩阵 `A` 的形状为  
        `[M, N]`，矩阵 `B` 的形状为 `[N, M]`。那么 `C` 的形状就是 `[M, M]`。 
        也就是说，被消去的维度是 `A` 的列维度和 `B` 的行维度。
      - 请注意，在上面的 `nki_matmul_tiled_` 示例中，矩阵是转置形式，即 `lhsT=A^T`。`nisa.nc_matmul` 接受 `lhsT=A^T` 和 `rhs=B` 作为参数，返回 `A x B`。
  - Tile 维度：
      - tile 大小根据张量引擎矩阵乘法操作的约束设置，如[这里](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html)所述。
        - `TILE_M`：128 — `M` 维度的 tile 大小。
        - `TILE_K`：128 — `K` 维度的 tile 大小。
        - `TILE_N`：512 — `N` 维度的 tile 大小。
  - 遍历 tile：
      - kernel 使用 `affine_range` 循环沿 `result` 矩阵的 `M` 和 `N` 维度遍历 tile。
      - 对于每个形状为 `(TILE_M, TILE_N)` 的输出 tile，它在 PSUM 内存中分配一个临时部分和张量 `res_psum`。
  - 加载 tile：
      - 对于每个输出 tile，把 `lhsT` 和 `rhs` 的 tile 加载到片上 SBUF 内存中以便高效访问。
      - `lhsT_tile` 加载形状为 `[TILE_K, TILE_M]` 的切片，`rhs_tile` 加载形状为 `[TILE_K, TILE_N]` 的切片。
  - 矩阵乘法：
      - 使用加载的 tile 执行部分矩阵乘法，并把部分结果累加到 `res_psum` 中。
  - 存储结果：
      - 一旦某个结果块的所有 tile 全部计算完成，`res_psum` 中的部分和就被拷贝到 SBUF 并转换为所需的数据类型。
      - 最终结果存回 `result` 张量中的对应位置。

> 注意，相对在线教程，我们把 `nl.matmul()` 和 `nl.load()/nl.store()` 换成了 `nisa.nc_matmul()` 和 `nisa.dma_copy()`。这把 nki.lang API 下降到了 nki.nisa 层面。我们建议对任何计算指令都使用 nki.isa API。其 lowering 行为更确定，也更少出现可能导致莫名其妙编译错误的意外行为。 

总而言之，这种分块实现通过把大矩阵维度拆解成硬件兼容的 tile 大小来处理它们。它利用专用的内存缓冲区（即 PSUM）来最小化内存延迟并优化矩阵乘法性能。你可以在[这里](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials/matrix_multiplication.html)阅读更多关于 NKI 矩阵乘法的内容。

### 卷积层概述（Convolution Layer Overview）

现在让我们把注意力转向卷积层。回想一下课堂上讨论的[卷积操作](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_26)。它涉及在__输入特征图__上滑动一个滤波器，在每个位置上，滤波器与重叠的输入区域相互作用。在每个重叠区域中，滤波器权重与输入区域的值之间执行逐元素乘法。这些逐元素乘法的结果随后相加，为输出特征图中的对应位置产生一个值。这个过程捕捉局部空间模式以及相邻特征之间的关系。

<p align="center">
  <img src="handout/convolution.png" width=55% height=55%>
</p>

输入特征图通常包含多个通道。例如，一张图像通常包含三个 RGB 通道（红、绿、蓝）。在这种情况下，卷积不再只对二维空间区域计算加权和，而是同时对二维空间区域和通道深度计算加权和。下图展示了一个在带有三个 RGB 通道的 32x32 输入图像上执行卷积层的例子。图中，一个 5x5x3 的滤波器作用于 32x32x3 的图像，产生一个 28x28x1 的输出特征图。

<p align="center">
  <img src="handout/cs231n_convolution.png" width=55% height=55%>
  <br>
  <em>Source: CS231N https://cs231n.stanford.edu/slides/2025/lecture_5.pdf </em>
</p>

__如图所示，每个滤波器产生单通道的输出。__ 要生成多个输出通道，需要对输入特征图应用多个滤波器。除此之外，每个卷积滤波器还包含一个标量偏置值，要加到每个加权和上。 

卷积算子的输入和输出可以总结如下（暂时忽略偏置）：

<p align="left">
  <img src="handout/conv2d_summary.png" width=50% height=50%>
</p>

此外，[卷积层](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)除了输入特征图、滤波器权重和标量偏置之外，还可以接受额外的超参数，如 padding 和 stride。不过，我们*简化了你的卷积的约束*，让你更容易实现。你**只需要支持 stride 为 1**，并且**不必担心 padding**，因为在输入特征图传入你的 kernel 之前，我们会为你做好 padding。

### 把卷积映射到矩阵乘法（Mapping Convolution to Matrix Multiplication）

现在，我们的目标是把卷积算子映射到 Trainium 张量引擎支持的高性能矩阵运算上。为此，我们可以比较卷积与矩阵乘法的数学形式。

**Conv2D：**

<p align="center">
  <img src="handout/conv2d_formula.png" width=65% height=65%>
</p>

**矩阵乘法：**

<p align="center">
  <img src="handout/matmul_formula.png" width=25% height=25%>
</p>

课堂上我们讨论过一种把多滤波器卷积转化为单个大矩阵乘法的方法。这里我们做同样的事情，但采用一种不同的方法，它能在 Trainium 上得到高效的实现。在这种方法中，卷积操作被表述为一系列独立的矩阵乘法。下面是这种表述方式的可视化示意。

> [!NOTE]
> **这种 conv -> matmul 的归约与课上讲的那种为每个空间 patch 创建单独一行的归约不同。**

<p align="center">
  <img src="handout/conv2d_matmul_diagram.png" width=100% height=100%>
</p>

在这种方法中，输入特征图的高度和宽度维度被展平成单个维度，把输入 reshape 为 `(Height × Width) × Input Channels`。reshape 后的输入随后与滤波器的每个位置相乘，其中 `i` 和 `j` 分别从 `0` 取到 `Filter Height - 1`、从 `0` 取到 `Filter Width - 1`。每个滤波器切片的形状为 `Input Channels × Output Channels`，所得的矩阵乘法沿 `Input Channels` 维度收缩。为了让输入与每个滤波器切片对齐，输入必须按与滤波器当前位置 `(i, j)` 对应的偏移量进行平移。这些矩阵乘法的结果累加起来，产生输出张量。

下面是所述算法的伪代码：
```
- Have the input image with shape (Input Channels, Image Height * Image Width)
- Have the filter weights with shape (Filter Height, Filter Weight, Input Channels, Output Channels)
- Initialize the output to appropriate shape of (Output Channels, Output Height * Output Width)

# Iterate over the filter height
for i in range(Filter_Height):
    # Iterate over the filter width
    for j in range(Filter_Width):

        # Shift the Input tensor by (i, j) to align with the filter's current position
        input_shifted = shift(input, (i, j))

        # Perform matrix multiplication between the input and the filter slice
        # Note that this is a full matmul, without limit on input sizes
        output += matmul(transpose(weight[i,j,:,:]), input_shifted)
```

> [!NOTE]
> **这只是一个算法层面的描述，而本作业的目的就是让你弄清楚如何把这个算法描述映射为该硬件上的高效实现！**

### 最大池化层概述（Max Pool Layer Overview）
最大池化层通常用于 CNN 中相邻卷积层之间，以减小特征图的大小。这不仅能防止特征图过大——过大可能给计算资源带来问题——还能减少 CNN 中的参数量，从而有效减少模型过拟合。

最大池化层的工作方式与卷积层类似，都是在输入特征图上空间地滑动一个滤波器。不过，最大池化层不是对每个重叠区域计算加权和，而是从每个区域中选取最大值并存入输出特征图。该操作独立地应用于特征图的每个通道，因此通道数保持不变。例如，考虑一张带三个 RGB 通道的 4x4 输入图像经过一个 2x2 滤波器的最大池化层。所得输出是一张带三个 RGB 通道的 2x2 图像，表明空间维度缩小为原来的一半，而通道数保持不变。

<p align="center">
  <img src="handout/maxpool.png" width=37% height=37%>
</p>

如上所示，[最大池化层](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d)通常有独立的 stride 和滤波器大小超参数。与卷积层类似，我们简化了你需要实现的最大池化层的约束。你的 kernel 不用分别定义这两个参数，而是使用单个参数 `pool_size`，它同时对应滤波器大小和 stride。`pool_size` 只能设置为 1 或 2。当 `pool_size` 为 2 时，最大池化操作的行为如上图所示。当 `pool_size` 为 1 时，最大池化层相当于无操作（no-op），产生与输入完全相同的输出。`pool_size` 为 1 看似没有意义，但它实际上为你的融合层提供了额外的灵活性，你马上就会看到。 

### 融合卷积与最大池化（Fusing Convolution and Max Pool）
你将实现一个 NKI kernel，把卷积层和最大池化层组合成单个融合操作。下面我们将概述你的融合层的详细规格和要求。

<p align="center">
  <img src="handout/fused_kernel.png" width=95% height=95%>
</p>

上图展示了你的融合 kernel 在单输入通道的 6x6 输入上要执行的计算。融合 kernel 先用一个滤波器、stride 为 1 执行标准卷积。然后，融合 kernel 对卷积结果用 2x2 池化滤波器执行最大池化。

你的融合 kernel 接受以下参数：
  - `X` — 一批输入图像。`X` 的形状为 `(Batch Size, Input Channels, Input Height, Input Width)`。保证 `Input Channels` 是 128 的倍数。
  - `W` — 卷积滤波器权重。`W` 的形状为 `(Output Channels, Input Channels, Filter Height, Filter Width)`。保证 `Filter Height == Filter Width`。还保证 `Output Channels` 是 128 的倍数。此外，你可以假设权重的大小总是能完全放进 SBUF。
  - `bias` — 卷积滤波器偏置。`bias` 的形状为 `(Output Channels)`
  - `pool_size` — 最大池化滤波器的大小和池化 stride。保证输入大小、滤波器大小和 `pool_size` 之间都能整齐地整除。更具体地说，`(Input Height - Filter Height + 1) % Pool Size == 0`。注意，如果 `pool_size` 的值为 `1`，那么融合 kernel 就相当于一个普通的卷积 kernel。这让我们能灵活选择是否要最大池化。

可以随意使用[课程幻灯片](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_57)中的卷积层实现作为起点。如果你参考课程幻灯片，在我们的命名方案中，`INPUT_DEPTH` 与 `Input Channels` 同义，`LAYER_NUM_FILTERS` 与 `Output Channels` 同义。注意，你的融合 kernel 的输入参数形状与卷积课程幻灯片中描绘的不同。你可以像 Part 1 的 `vector_add_stream kernel` 中那样，使用 [NumPy reshape 方法](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)把输入 reshape 成你想要的任何形状。我们还在 `part2/conv2d_numpy.py` 中给出了卷积层和 maxpool 层的 NumPy 实现。NumPy 实现应能让你大致了解每层的编程逻辑。思考一下如何把 NumPy 实现融合成单个层会是个不错的练习——这正是你要在 kernel 中做的事情。可以翻阅 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html)，了解更多优化或其他 API 函数。你也可以查看 [NKI API 参考手册](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html)，了解所有可用的 API 函数及其用法。你可能会发现其中一些很有用。*提示：*[nisa.tensor_reduce(nl.max, ...)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_reduce.html) 对最大池化应该有帮助。[nisa.tensor_tensor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html) 对加偏置应该有帮助。

### 你需要做的（What You Need To Do）
对于本部分作业，只需关注 `part2/conv2d.py` 文件。我们提供了基本的起步代码；你的任务是在 `fused_conv2d_maxpool` 函数中完成（融合的）Conv2D kernel 的实现。
#### 通用建议（General Tips）
* **正确性优先。** 我们建议从最简单的情况开始：小图像、无偏置、无 maxpool。一旦你的 kernel 对小图像工作正常，就扩展其功能以处理大到无法完全放进 SBUF 缓冲区的图像。接下来加入偏置加法，然后把最大池化操作融合进你的 kernel。得到一个完全正确的解之后，再开始为性能/附加分做优化。
  * 测试框架会按由易到难的顺序在测试用例上运行你的 kernel。此外，如果你想先做出高性能的 conv2d kernel 之后再处理融合最大池化，你也可以选择在运行测试框架时省略 maxpool 测试用例。
* **彻底理解你的算法。** 在考虑任何分块策略之前，确保你对上述算法所需的矩阵运算（乘法、平移、加法）有扎实的理解。然后画出矩阵及其维度，思考如何把它们映射到硬件上，尤其是内存层次方面。
  * 你可能还需要对输入数组做预处理（例如 reshape 或转置它们）以便更高效地访问。提示：如果你在疑惑为什么可能需要转置，想想 NKI 矩阵乘法接口的独特之处——第一个输入矩阵是转置的。
* **记录 tile 维度。** 由于你无法一次计算整个输出，你需要思考把哪个输出维度拆成 tile。回想 SBUF tile 的约束——分区维度最多为 128，并且必须是张量的第一维。一旦你决定了输出形状，这对输入意味着什么形状？换句话说，计算单个输出 tile 需要 X 和 W 的哪个子集？
* **安排循环顺序时牢记数据局部性。**  你需要的 `for` 循环来自多个来源：算法定义的滤波器高度和宽度、分块矩阵乘法、以及批处理。 
  * 识别出这些循环之后，一个推荐的目标是把它们排序，使中间结果保留在 `PSUM` 中，直到每个 tile 的计算完全完成。这能确保 `SBUF` 中结果数组的每个部分只被写入一次，改善输出数据局部性——不过其他方法也可能达到相当的性能。
  * 在此之后，安排其余循环的顺序以优化输入数据局部性。如果不确定，就尝试不同的数据访问模式，找出效果最好的，并思考原因！
* **用 profiler 指导性能调优。** 一旦你有了可工作的 kernel，很可能需要进一步调优性能才能拿到满分/附加分。这时 profiler 是你的好帮手：寻找张量引擎空闲、利用率低的大段空隙/阶段，并尝试重构代码以最小化花在这些部分的时间。
  * 回顾 Part 1 也会有所帮助，在那里我们优化了一个简单的向量加法 kernel（如果你做了附加分，还有一个转置 kernel）。

#### 测试（Testing）
使用提供的测试框架脚本验证你的实现。要运行测试，进入 `part2/` 目录并执行：
```
python3 test_harness.py
```

要检查你实现的带融合 maxpool 的 Conv2D kernel 的正确性和性能，带 `--test_maxpool` 标志调用测试框架。 

测试框架会先运行正确性测试，然后运行性能检查。满分方案必须在保持正确性的同时，达到参考 kernel 性能 120% 以内的性能。它会以 float32 和 float16 数据类型的输入张量调用你的 kernel，其中 float16 的性能要求更严格。编写 kernel 时请务必牢记这一点！

注意，你的 kernel 的性能测试将在*不带* `--profile` 的情况下进行（它会轻微改变执行时间），以与性能阈值的设定方式保持一致。

#### 报告与 Profiling（Writeup and Profiling）
学生需要提交一份简要描述实现的书面报告。还要描述你是如何着手优化实现的。请务必 profile 你的实现，并报告在 `float16` 和 `float32` 两种数据类型下达到的 MFU（模型 FLOPs 利用率，Model FLOPs Utilization）。你可以带 `--profile <profile_name>` 标志运行测试框架来捕获 trace，然后运行：
```
neuron-profile view -n [profile_name].neff -s [profile_name].ntff
```

> [!TIP]
> 打开 profiler 时，你可能会看到一些关于缺少 benchmark 参数的警告。这里你唯一需要提交的参数是 MFU 值，把鼠标悬停在 GUI 的 Estimated MFU 部分中的 "Cumulative Utilization" 线上即可查看，如下图所示。（请务必取最末尾处的 MFU。）

<p align="center">
  <img src="handout/mfu.png" alt="Profiler warning" width="90%">
</p>

### NKI 使用建议（Tips on Using NKI）
* 在以下场景优先使用 nki.isa API：
    * 所有计算操作
      * 用 nisa.nc_matmul 而不是 nl.matmul
      * 用 nisa.tensor_scalar(op=nl.add, <>) 而不是 nl.add
    * 优先使用 nisa.dma_copy() 而不是 nl.load()/nl.store()。
    * 调用 nisa 计算操作时，确保只把 op=nl.* 代码作为参数传入。例如，不要传 op=math.sin。
* 避免使用嵌套函数。所有函数都定义在模块级。 
* 调试实现时，可以带 `--simulate` 标志运行测试框架。它会用 `nki.simulate_kernel()` 调用包装你的实现：更多信息见[这里](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html#nki.simulate_kernel)。在模拟模式下运行时，你可以在 kernel 中插入 `nl.device_print(str, tensor)` 来打印 device 张量的中间值。不过，CPU 模拟和 device 上执行之间__可能__存在一些分歧。如果你对结果不确定，建议通过返回中间张量来调试。
* 修改张量赋值时要小心。一些 nisa API 把 dst 张量作为参数，例如 nisa.dma_copy(src=<>, dst=<>)。另一些 API 通过函数自身产生 dst 张量，可能需要用来修改已存在的张量。在未来的 NKI 版本中，所有 ISA API 都将把 dst 作为参数。例如：
  * x_sbuf = nl.zeros(shape=hbm_tensor.shape, buffer=nl.sbuf)（创建数组）
  * nisa.dma_copy(src=hbm_tensor, dst=x_sbuf)（拷贝进数组）
  * 具体来说，如果你选择使用 `nl.load(...)`，`x = nl.load(...)`（创建新数组）与 `x[...] = nl.load(...)`（修改已存在的数组）是不同的。
* 避免使用 block dimension，它是纯软件构造，对硬件没有影响。（如果你不知道它是什么，不用担心。）要么把它放进自由维度，要么使用张量列表。参见公开[文档](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_block_dimension_migration_guide.html#nki-block-dimension-migration-guide)。
* 张量索引优先使用整数切片。需要更高级的索引时，使用 [`nl.mgrid`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.mgrid.html)。不要使用嵌套切片/mgrid（例如 t[0:128, 128:256][0:64, 0:64]）。不要使用 nl.arrange()。


## 附加分（Extra Credit）
在更小的图像上再次运行 `neuron-profile`。小图像和大图像之间的 MFU 有差别吗？如果有，你会如何针对小图像优化你的融合卷积层？（了解以下事实可能有帮助：`nisa.nc_matmul` 可以接受 >2D 的张量作为 `moving` 参数，只要满足 PSUM 的硬件约束。）

满足小图像性能目标（更严格的目标）的方案将获得最多 5 分的附加分。你的报告必须清楚地解释你的方法以及为优化方案所采取的步骤。

## 评分标准（Grading Guidelines）

正确性测试使用两类图像。第一类是尺寸为 32×16 的小图像。第二类是尺寸为 224×224 的大图像，它超出 SBUF 的容量，无法一次全部放入。你的代码必须通过所有正确性测试才能获得性能分数。

性能测试在不同配置下评估你的 kernel 相对于参考 kernel 的性能：有/无 maxpool、float16 和 float32 精度。

作为中间目标，我们纳入了参考 kernel 未优化版本的放宽延迟。如果你的 p99 延迟在放宽延迟的 120% 以内，你将获得 95% 的性能分数。如果在优化后的参考延迟的 120% 以内，你将获得全部性能分数。

附加分（EC）部分只设一个性能阈值，即参考延迟的 120%。

**报告：30 分**
  - Part 1 问题：20 分
  - Part 2 问题：10 分

**矩阵转置 kernel 正确性：10 分（+1 分性能附加分）**

**融合卷积 - 最大池化 kernel 正确性：10 分**
  - 小图像：2.5 分
  - 大图像：2.5 分
  - 偏置加法：2.5 分
  - 最大池化：2.5 分

**融合卷积 - 最大池化 kernel 性能：50 分（+5 分附加分）**
  - 无最大池化（float16）：17.5 分
  - 无最大池化（float32）：17.5 分
  - 有最大池化（float16）：7.5 分
  - 有最大池化（float32）：7.5 分
  - 小图像上无最大池化（float16）：1.25 分附加分
  - 小图像上无最大池化（float32）：1.25 分附加分
  - 小图像上有最大池化（float16）：1.25 分附加分
  - 小图像上有最大池化（float32）：1.25 分附加分

## 提交说明（Hand-in Instructions）

请使用 Gradescope 提交作业。如果你与搭档合作，请记得在 Gradescope 上标记你的搭档。

1. **请将你的报告提交为 `writeup.pdf` 文件。**
2. **请运行 `sh create_submission.sh` 生成要提交到 gradescope 的 `asst4.tar.gz`。** 如果脚本报错 'Permission denied'，你应运行 `chmod +x create\_submission.sh`，然后重新运行脚本。还请仔细检查生成的 `tar.gz` 是否包含：
  * 包含 Part 1 转置 kernel 的 `kernels.py` 文件。
  * 包含 Part 2 融合 Conv2D kernel 的 `conv2d.py` 文件。
