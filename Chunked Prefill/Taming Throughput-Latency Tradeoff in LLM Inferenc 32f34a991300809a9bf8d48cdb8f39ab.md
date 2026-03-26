# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

就是：Chunked Prefill

很好文章使我大脑旋转。

学会分析性能问题，非常关键，在发现问题，研究问题时都能得到很insightful的结果。很多问题本质是tradeoff，该怎么tradeoff怎么思考，这篇文章给了一个很好的示范。

看完这篇对continuous batching有了更深理解，但是看完之后对于一些细节还有点模糊。

# Abstract

Challenge：将多个requests batch到一起，特别是不同阶段（prefill和decode），会限制 high throughout和 low latency。

Technical Contribution：

1. Chunked-prefills：将prefill request 分到近乎相当的chunks中，并且创造 stal-free 调度，在batch中添加新的requests。
2. 

# Introduction

Prefill 是 compute-bound，因为输入prompt之后会并行计算所有token。

 Decode是memeory-bound，因为一次只产生一个token。【已经默认使用kv cache了】→ batching是非常有利于decode的，因为更大的batch就能更有效的利用GPU（reuse parameters），但是对于prefill的意义有限，算力已经满了。

当前schedulers分为两类：

1. 在request精度上的decode-prioritizing：在throughput上妥协，当batch中的request完成，不会返回，会继续执行，此时实际的batchsize是减少了的，直到batch中最后一个request完成，才会一起返回。
2. prefill-prioritizing：在Orca这些system中，只要GPU memory还有位置，就会将处于prefill阶段的requests早早放入。→ 导致的问题：虽然这样会提升吞吐量，但是由于prefill阶段计算量大，这样decode必须等很久才能继续到下一个iteration。
    
    > However, **prioritizing prefills leads to high latency because it interferes with ongoing decodes**. Since prefills **can take arbitrarily long time depending on the lengths of the given prompts**, prefill-prioritizing schedulers lead to an undesirable phenomenon that we refer to as generation stalls in this paper.
    > 
    
    prefill优先会导致高的latency，因为会干扰正在decode的request。
    

challenge2: pipeline stalls/bubbles

[**大模型推理并行策略(DP/TP/PP/SP/EP)原理简介**](https://zhuanlan.zhihu.com/p/2003423046342554380)

<aside>
💡

推理并行策略

1. **DP 数据并行：**不同的GPU运行LLM模型的多个副本，并在每个模型副本上独立处理用户请求组。多个模型副本共用一个**推理实例**，由**推理实例中的调度器负责分配请求**给不同DP的模型副本。
2. **TP 张量并行：** 将模型的每一层分割到不同GPU上执行，用户请求（输入数据）会在GPU间流转，**每个GPU计算的部分结果最终重新组合为完整输出。**
    
    ![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image.png)
    
3. **SP 序列并行：** 将**长序列切分成多个片段**，分配到不同GPU设备上并行处理的模型并行策略。
4. **PP**：模型按层拆分到不同设备，数据以流水线方式在不同设备间顺序流动处理。【PP前向与后向计算中会出现空泡，训练中需要考虑空泡的消除。】
    
    **Pipeline bubbles (流水线气泡):** 指流水线中因为前后处理速度不一致，导致某些 GPU 在“干等”的空闲时间。
    
</aside>

Technical Contribution: Sarathi-Serve

Chunked-Prefills & Stal-free

> Chunked-prefills splits a prefill request into equal computesized chunks and computes a prompt’s prefill phase over multiple iterations
> 

# Background

主要是介绍Transformer架构+LLM Inference Process

Paged-Attention →解决碎片问题，空出更多空间，容纳更多batch，允许更大的batch size

## Multi-GPU LLM Inference

**micro-batching**

PP 与 TP 在通信成本上的**差距**：

1. PP只需要point-to-point：模型被按层切分。比如第 1-10 层在 GPU 0，11-20 层在 GPU 1。GPU 0 算完后，只需要把结果（中间特征向量）发给“下游”的 GPU 1。
2. TP需要all-reduce：把每一层内部的矩阵乘法拆开。每一张 GPU 只算矩阵的一部分，算完后必须把**所有人的结果汇总、求和，再分发回给所有人。**

# Motivation

1. batching对decode phase非常有用，但是对prefill阶段没什么用。【增加了实际的batchsize，每次参与计算的batch多了，对于decode来说reuse parameters】
    
    其实linear operators 对于runtime cost的占比是最大的。
    
    ![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image%201.png)
    
2. Decode阶段计算利用率低；这也意味着有更多的token可以和decode batch一起处理，同时不显著地增加latency。
    1. Memory-bound $T_{math}<T_{mem}$.  low Model FLOPs Utilization 
    2. 预填充批次（Prefill batches）将从 HBM（显存）向 GPU 缓存提取线性算子权重的开销，分摊到了大量的 Token 上，从而使其具备极高的计算强度（Arithmetic Intensity）
3. Orca 支持prefill和decode的混合，但是vLLM只支持batch中要么全部是prefill or decode requests
    1. vLLM中的generation stall：scheduler会优先prefill的batch抢占decode的batch，会导致decode的request stall
    2. Orca中的stall：由于batch是可以混合的，所以不存在prefill抢占。但正是因为混合，所以在同一个iteration中，prefill计算所需时间一般是比decode要长的多的，也会导致decode request的stall。【One way to reduce latency spikes in iteration-level batching systems is to use smaller batch sizes as recommended in Orca  减少batch-size从而避免decode和大prefill】
    
    **二者都是generation stall，但是具体原因是不同的。**
    
    ![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image%202.png)
    
4. pipeline bubbles
    
    首先要清楚的是：在continuous batching中，不同phase以及处理不同token index的request拼在一起，在大部分部分计算是一样的，但是比如在计算attention这部分不一样
    
    所以prefill和decode混合，二者比例不定，可能造成巨大的气泡。
    
    > There can be a large variance in compute time of LLM iterations depending on composition of prefill- and decode-tokens in the batch. This can lead to significant bubbles when using pipeline-parallelism.
    > 

# Sarathi-Serve: Design and Implementation

## Chunked-prefills

motivation：decode计算利用率低，可以有其他计算一起。→ 那就和prefill一起！但是直接和很长的prompt的prefill一起又会导致很高的TBT latency → 所以分块，把这些prefill 分成小的chunk 几个iteration完成。

<aside>
💡

为什么prefill需要的计算量远大于decode呢？

prefill前面token还需要计算的原因，就是如果有多个decoder 层的时候，下一个层输入的就是上一层计算得到的，这样所有token都会带有前文信息，否则多层和一层没有区别。必须计算才能获得最后正确生成的token。

所以在prefill阶段，prompt的所有token都是需要计算的。

但是后续decode，每次的输入还是原始的词embedding后的，（并不是经过多层处理后的token），所以对于每一次来说经过 $W_q W_k W_v$ 变换后得到token的QKV都是一样的（推理中权重固定）。那么输入到下一层decoder的也是一样的，只是会新增一个token。所以缓存KV是可行的，后续也不需要Q，因为前面的KV是固定的了。【换句话说：前面的token每次计算结果是一致的】

所以这里是我之前没理解清楚的：并不是因为不需要前面这些token了，所以不需要缓存Q。而是因为Query不会再更新了【所以不需要继续往下输出】，对应的KV也不会更新了，所以新的token的query在计算的时候要使用的不会变。

</aside>

## Stall-free batching

怎么做到的？Based Chunked prefill

> Sarathi-Serve first calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO.
> 

先计算出一个iteration中一个batch中最多能计算的token数（基于SLO）就是Budget 这是需要测试的

> In every scheduling iteration, we first pack all the running decodes in the next batch (lines 6-8 in Algorithm 3). After that, we include any partially completed prefill (lines 9-12). Only after all the running requests have been accommodated, we admit new requests
> 

在每一轮iteration中，先pack所有decode在下一个batch中，再看有没有部分完成的prefill。同时，只有当所有request妥当处理，才去接新的request（就是new prefill）[没有固定batch的概念，只有一个一个iteration 一个iter结束后都会重新pack batch]

![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image%203.png)

算法逻辑

![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image%204.png)

## Determining Token Budget

> From a TBT minimization point of view, a smaller token budget is preferable because iterations with fewer prefill tokens have lower latency.
> 

一个TBT=相邻两个token生成时间=一个iteration时间

token budget小 意味着处理的prefill少（都是得到一个token）

<aside>
💡

这里需要提到：所以可以进行pack组合token

至于细节的，decode和prefill的token大小还是有差距，工程中如何解决呢？

![image.png](Taming%20Throughput-Latency%20Tradeoff%20in%20LLM%20Inferenc/image%205.png)

</aside>

但导致问题：1. budget小，GPU效率低。整体吞吐量下降。 2. 重复的KV访问。切得越碎，重复读取旧 KV Cache 的次数就越多。这造成了巨大的**显存带宽浪费**，原本只需要读一次的数据，现在因为分段，被反复搬运了好多次。

当然，还考虑到底层矩阵计算-tile，多一个token都可能导致性能的巨大变化。如果你的 Budget刚好是 Tile 大小的倍数（比如 **256**），GPU 的所有核心都能整齐划一地干活，效率最高。

> Larger chunks lead to higher inter-batch runtime variations that result in pipeline bubbles which results in lower overall system throughput. On the other hand, picking a very small token budget can lead to higher overhead due to lower arithmetic intensity and other fixed overheads.
> 

我们要选择一个合适的token budget，这是需要在特定环境下测试得到的。

# 参考

1. https://www.usenix.org/conference/osdi24/presentation/agrawal
2.  [大佬解读很有启发](https://zhuanlan.zhihu.com/p/718715866)
3. [大模型推理并行策略](https://zhuanlan.zhihu.com/p/2003423046342554380)