# AI 基础设施方向学习路线(大模型训练 / 部署 / ML Systems)

> 目标:具备大模型训练、部署、推理优化的系统能力(GPU 编程、分布式训练、训练/推理框架)。
> 制定时间:2026-08,基于已完成 xv6 全部核心 lab(含 mmap)及网络、数据库、编译器、体系结构基础。

## 现状盘点

**已有的系统内功:**

- 操作系统:xv6 10 个 lab(进程、虚拟内存、文件系统、锁、驱动全做过)
- 计算机体系结构 / 组成原理基础
- 计算机网络、数据库、编译器
- C/C++ 工程能力、调试能力(GDB)

**目标方向:** AI 基础设施(AI Infra)——大模型训练框架、分布式训练、推理引擎。
这条线的核心不是"设计模型",而是"让模型在成百上千张 GPU 上算得快、算得起、算得稳"。

**尚缺的三块拼图:**

1. GPU / 并行计算内功(单机算力榨干)
2. 大模型本身的完整实现经验(从分词到训练到部署)
3. 多机分布式容错与共识(训练平台层)

---

## 路线总览

```
阶段一  CS149 并行计算(GPU 内功)            [~2 个月]
阶段二  CS336 从零造语言模型(主线核心课)     [~2-3 个月]
        动手学ai，主要是进行手写的搭建模型并且训练
阶段三  MIT 6.824 Lab1+2 + DDIA(多机容错)   [穿插进行,~1.5 个月]
阶段四  工业界源码实战(Megatron/vLLM)       [长期]
```

ML 基础(反向传播、Transformer、Adam)在 CS336 里顺带补齐,不单独开课;
CS224n 不需要专门学(其 Transformer 部分被 CS336 覆盖)。

---

## 阶段一:Stanford CS149 — Parallel Computing(并行计算)

**为什么先学它:** 大模型训练的性能问题本质是并行计算问题。单卡 kernel、内存带宽、
并行划分,这些是大模型一切优化(数据并行、算子融合、FlashAttention)的概念原型。

**学什么:**

- GPU 体系结构:SIMT、warp、线程块、内存合并(coalescing)、shared memory
- 并行思维:Amdahl 定律、负载均衡、通信/计算比
- 内存层次结构与带宽瓶颈(推理优化的根基)
- CUDA 编程实战

**资源:**

- 视频:CMU 15-418 录像(同一位老师 Kayvon,B 站有搬运)
- 作业:GitHub 组织 [stanford-cs149](https://github.com/orgs/stanford-cs149/repositories),全公开,带本地 checker
- 参考:[CS 自学指南 CS149 词条](https://csdiy.wiki/en/%E5%B9%B6%E8%A1%8C%E4%B8%8E%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F/CS149/)

**作业清单(按顺序):**

| 作业 | 内容 | 备注 |
|---|---|---|
| asst1 | CPU 多核性能分析 | 纯 CPU,本地可跑 |
| asst2 | 并行任务调度器 | 纯 CPU |
| asst3 | **CUDA 渲染器** | 核心作业,需 NVIDIA GPU(无卡用 Colab / AutoDL) |
| asst4 | OpenMP 大图计算(PageRank) | 纯 CPU |
| asst5 | LLM 推理相关 | 直接对口目标方向 |

**完成标志:** 能解释"为什么这个 kernel 是带宽瓶颈而不是算力瓶颈";asst3 渲染器达到参考实现的主要加速比。

**止损线:** 若 1 个月还没做完 asst3,先跳过 asst2/4,保住 CUDA 主线。

---

## 阶段二:Stanford CS336 — Language Modeling from Scratch(主线核心课)

**为什么是它:** 和 xv6 同一个哲学——"像从零写操作系统一样从零造一个语言模型"。
一半内容(Triton kernel、并行训练、推理优化)与 CS149 咬合;另一半(手写 Transformer、
训练流程)补齐 ML 基础。CS 系统背景出身学这门课有天然优势。

**学什么:**

- BPE 分词器(手写)
- Transformer 每一层(手写,不调用现成模块)
- 训练循环、优化器(AdamW)、混合精度
- GPU/Triton kernel 优化
- 并行训练(数据并行、张量并行、流水线并行)
- 推理、评测、数据清洗、对齐(SFT/RLHF)

**资源:**

- 作业:[github.com/stanford-cs336](https://github.com/stanford-cs336)(学生版仓库,自包含)
- 课件:[stanford-cs336/spring2025-lectures](https://github.com/stanford-cs336/spring2025-lectures)
- 课程网站:[2025 春存档](https://cs336.stanford.edu/spring2025/),视频 YouTube/B 站
- 笔记参考:[QihongRuan/stanford-cs336-notes](https://github.com/QihongRuan/stanford-cs336-notes)

**算力说明:** 作业按单卡可完成设计(大部分 Colab 可跑),完整预训练不做规模要求。

**完成标志:** 不看参考代码能从零写出可训练的小 Transformer(分词→模型→训练→推理全通);
能说清数据并行/张量并行/流水线并行各自的通信开销来源。

---

## 阶段三:MIT 6.824(最小可行路径)+ DDIA(分布式容错)

**定位:** 多机那半边的内功——几千张卡训几周,每天都有机器挂,checkpoint、容错、
共识是训练平台的日常。作为"思维基本功"必修,但只取核心,不求全做完。

**6.824 最小可行路径:**

1. Lab 1 MapReduce(≈"把计算摊到多机"的原型,顺便把 Go 练了)——2~3 周
2. Lab 2 Raft(全课灵魂,做好磨一个月的心理准备)
3. Lab 3/4(KV、分片 KV)——**可跳过**,Raft 做完已拿到 80% 价值

**Go 语言:** 半天过 [A Tour of Go](https://go.dev/tour/)(重点:goroutine、channel、sync.Mutex),
然后边做 lab 边学。Go 是云基础设施通用语言(K8s/etcd/Docker),迟早要会。

**配套阅读:《Designing Data-Intensive Applications》(DDIA)**

- 中文在线版(冯若航译):[ddia.vonng.com](http://ddia.vonng.com/) / [GitHub Vonng/ddia](https://github.com/Vonng/ddia)
- 本地已生成:`~/cs/ddia-book/output/ddia.pdf`(第二版中译,352 页,平板阅读)
- **重点读第 8、9 章**(部分失败、一致性与共识)——Raft lab 前的最好热身
- 时间紧就精读第二部分(5~9 章:复制、分区、事务、分布式的麻烦、共识)

**完成标志:** Raft 三个子测试(选举/日志复制/持久化)全过;能讲清"为什么分布式里
'对方没回'无法区分是死了、慢了还是网断了"。

---

## 阶段四:工业界源码实战(长期,无固定终点)

课程做完后,直接读真实系统,从"会学"变成"会用":

- **训练侧**:Megatron-LM(并行策略)、DeepSpeed(ZeRO 显存优化)、PyTorch DDP/FSDP
- **推理侧**:vLLM(PagedAttention、连续批处理)、Triton 算子库、FlashAttention 论文与源码
- **方法:** 用之前练的"代码库导航术"——先看骨架(目录/入口),再追一条主干
  (一次训练 step / 一次推理请求的完整链路),落到自己的笔记里

**选方向参考:**

- 偏训练性能 → 深挖 Megatron + Triton/CUDA kernel
- 偏推理部署 → 深挖 vLLM + 量化/KV cache
- 偏平台调度 → 6.824 路线延伸(K8s、Ray、弹性训练)

---

## 学习方法备忘(从 xv6 阶段验证过的经验)

1. **定好止损线再开工**:默认"先做完当前最小单元再决定要不要继续",不要默认"学完这门课"
2. **视频和作业穿插**,绝不先刷完全部视频再动手
3. **读代码用导航术不用记忆力**:先骨架(目录/头文件)→ 追一条主干链路 → 动态跑起来校准 → 写进笔记
4. **每门课记"契约/不变量"而非细节**(参照 xv6-notes.md 的格式,每门课建一份笔记)
5. 踩坑即收获:把每个 panic/bug 的根因记下来,这比看十遍讲义管用

## 预计总时长

每周投入 10~15 小时计,阶段一~三约 **5~6 个月**;阶段四为长期持续。
节奏上允许拉长,不允许跳步——CS149 的 CUDA 内功是后面一切的地基。
