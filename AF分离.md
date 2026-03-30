# AF分离

MegaScale-Infer technical的创新很多。

# MoE 基础知识

## Route structure

![image-20260330150005101](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330150005101.png)

## ALL-to-ALL 通信

**Tutorial Link:** https://zhuanlan.zhihu.com/p/1976060556906624692

看完两篇AFD的文章，心中很大的一个困惑是，standard MoE为什么使用的是all2all通信。按道理来说，AF分离后，不同的attention node只要向route选择的expert node去发送data就可以了。

以下均为NCCL实现的ALL2ALL。

### All2ALL 是什么？

每个GPU都需要向所有GPU发送不同数据，同时从所有GPU接收不同的数据。与AllGather不同，**每个节点发送给不同目标的数据是不同的。**

![image-20260330155124813](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330155124813.png)

### NCCL VS RDMA

**Tutorial Link**: https://zhuanlan.zhihu.com/p/720502061(NCCL)

NCCL中：API为 ncclCommInitAll 支持P2P操作（`ncclSend`/ `ncclRecv`）

在通信前会用ncclCommInitRank()或ncclCommInitAll()初始化rank进程的communicator，初始化之后才会开始通信。意思就是说，初始化时就会规定所有参与通信的rank进程，通信需要在静态进程组内进行同步。因此，当组成communicator后，其中一个GPU发生问题就会导致集体通信（ALL2ALL）发生问题。

但问题是：这是双边的，发送方调用send时，接收方必须有对应的Recv在运行。

---

### RDMA

**Tutorial Link**: https://zhuanlan.zhihu.com/p/55142557

EaaS使用的是**RDMA** ，所有读写操作都由client发起，直接读写对方内存，服务器不主动发起。[这也是为什么EaaS要说expert service是**stateless**的]

传统的TCP/IP技术在数据包处理过程中，要经过操作系统及其他软件层，需要占用大量的服务器资源和内存总线带宽，数据在系统内存、处理器缓存和网络控制器缓存之间来回进行复制移动，给服务器的CPU和内存造成了沉重负担。

RMDA是一种新的直接内存访问技术。RDMA让计算机可以直接存取其他计算机的内存，而不需要经过处理器的处理。RDMA将数据从一个系统快速移动到远程系统的内存中，而不对操作系统造成任何影响。

其优势：

1. 零拷贝(Zero-copy) - 应用程序能够直接执行数据传输，在不涉及到网络软件栈的情况下。数据能够被直接发送到缓冲区或者能够直接从缓冲区里接收，而不需要被复制到网络层。
2. 内核旁路(Kernel bypass) - 应用程序可以直接在用户态执行数据传输，不需要在内核态与用户态之间做上下文切换。
3. 不需要CPU干预(No CPU involvement) - 应用程序可以访问远程主机内存而不消耗远程主机中的任何CPU。远程主机内存能够被读取而不需要远程主机上的进程（或CPU)参与。远程主机的CPU的缓存(cache)不会被访问的内存内容所填充。
4. 消息基于事务(Message based transactions) - 数据被处理为离散消息而不是流，消除了应用程序将流切割为不同消息/事务的需求。

RDMA的硬件实现 









# MegaScale-Infer

为解决计算资源浪费的问题，MegaScale-Infer将Attention层与FFN层部署在不同服务器上。

如果简单的让Attention跑完再发给FFN，这样会导致空转。

引入ping-pong pipeline，将一个大的request batch分为很多个micro-batchs 同时让他们在attention和FFN中穿梭。

## Intro

MoE 动态地将input tokens route到FFN的subset（子集）中（也叫experts），而不是使用所有FFNs。

但是减少的计算复杂度并没有转换成更低的计算开销，这种不一致来自LLM 推理特性和计算性能的mismatch。【传统方式的部署导致GPU在跑MoE的时候算力浪费，于是文章提出新的方式】

### Benefits

1. **Tailored parallelism strategies**：允许module使用特定的并行策略，特别地，attention module **被复制到多个不同的GPU上使用DP**（数据并行），FFN module 使用专家并行。这样每个专家的GPU利用率大幅提升，因为每个attention replica的batch size增大，汇总后再发给专家，专家处理的request会更多，**充分使用性能**。【解释了为什么MoE能减少计算量，但是实际上使用却没有发挥性能，因为一个attention做完还是只有一个专家在FFN算，处理的token很少，这样处理之后专家最后处理的batch size很多】
2. **Independent hardware selection**：允许Attention和FFN modules部署到异构的GPU，充分利用不同的capacity，**达到更小的cost**。

### Technical Contribution 

1. Challenge: 分离架构会导致attention和FFN modules的**闲置**（另一个工作另一个等数据）**Contribution: ping-pong pipeline parallelism strategy**
2. Challenge: 传统的架构中token routing是all2all，现在调整为M2N[M个attention给N个expert]。传统的通信库NCCL在面对这种M2N表现不好。**Contribution: 文章提出一种专注减少操作开销和改善通信稳定性的M2N 通信库。**

## Background and Motivation

### Mixture of experts

### Model Parallelism

将model parameters分散到多个device。

TP张量并行会带来大量的通信开销，于是TP被局限在单机上的多个GPU（NVlink带宽比network带宽高的多）

PP将model layers分为几个stage，在不同的device上运行从而形成pipeline。但由于各个node间要传数据，单次的推理所花的时间可能会多一点，但是总体吞吐量却是上涨。

为MoE特殊制定的strategy叫做：expert parallelism(EP) ，比起TP好在，计算是使用完整矩阵有益于利用GPU。

有两次all2all通信：1. attention计算完后的GPU要把token发送给gating选择好的experts 2. 要把各个experts处理好的token返回。

但也存在问题：1. 专家间负载不均衡，有的专家一直用，有的一直不用。 2.  随着toop-k专家增加，通信量激增。

![image-20260329133814671](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329133814671.png)

### Problems in Large-scale MoE Serving

![image-20260329140158719](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329140158719.png)

以上证明了要使计算性能充分发挥，要使得计算时间多于参数搬运时间，所需要的B、F关系，以及得到的batchsize大小。

理想上，增大batch size就能提高推理效率，但是实际上有很多因素限制。1. Latency 2. memory constraint 3。更多GPU可以有更大batch size但花费也大。

## MegaScale-Infer Overview

![image-20260329140628210](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329140628210.png)

### Disaggregated expert parallelism.

组成：1. 一个**expert node**由1-8个GPU组成（这是一个physcial server 存一个expert参数）2. 所有expert nodes组合在一起形成一个expert parallelism group 3.  attention的parameters复制到每个attention node里面

**然后在每个Node内部使用TP**

### High-performance M2N communication.

设计高效M2N 通信库。

## Disaggregated Expert Parallelism

使用的是GQA(grouped-query attention)

- 将查询头进行“分组”（Grouped）。比如 32 个查询头分成 8 组，每组 4 个查询头。
- **分配：** **每一组共享一对键（Key）和值（Value）**。
- **效果：** 它的 KV Cache 大小适中，既比 MHA 省显存，又比 MQA 聪明。

### Ping-Pong Pipeline Parallelism

单纯的“拆分（解耦）”只会让系统变得更慢，因为**同步等待和网络延迟**会让昂贵的 GPU 处于闲置状态。我们必须通过“**流水线化**”来填补这些空隙。

![image-20260329142906429](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329142906429.png)

目的是为了使**通信**与计算的时间**重叠overlap**

$T_a$和$T_e$ 分别是micro-batch 在attention和expert node的计算时间。  

![image-20260329144353737](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329144353737.png)

Constraint 1 是为了最小化GPU的空闲时间。

Constraint 2&3 是为了隐藏通信开销。[这样通信能在**计算的阴影中**]。处理batch的时间必须要长到足够最后一个micro-batch处理完之后，第一个micro-batch正好回来，继续下一层的计算。

![image-20260329145526325](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329145526325.png)

经过计算，通信好的至少需要3个micro，不好的4个。

![image-20260330092124613](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330092124613.png)



## High-Performance M2N Communication

NCCL的问题：1. 高额外开销 2. 传统的 NCCL 通信库在处理大规模 MoE 节点跳转时，容易在极端情况下出现“网络大塞车”，导致推理延迟变得不可控。

具体问题：

1. 中间副本开销 Intermediate Copies
   1. NCCL 网络需要将数据从 **GPU 显存** 传输到 **CPU 代理 (CPU proxy)**，由 CPU 代理执行实际的网络操作。尽管有“用户缓冲区注册（User buffer registration）”等优化技术，但仍无法完全消除这些中间副本的拷贝过程，从而增加了延迟。
2. P2P 操作的批处理限制 (Batch Size Limitations)
   1. 点对点（Peer-to-Peer）组操作在处理时受到限制，**每次最多只能处理 8 个操作**。随着接收端（Receivers）数量的增加，这种较小的批处理规模会产生负面影响，限制了系统的扩展性。
3. 通用框架的设置开销 (General-Purpose Setup Overhead)
   1. 作为一个通用集体通信库，NCCL 为了保证广泛的适用性，在执行操作前需要进行复杂的设置工作。

### M2N Sender

![image-20260329150823728](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329150823728.png)

这里涉及到一些CUDA内核的东西。流程如下：

1. M2N sender用CUDA event等待内核计算完成
2. 确保要传输的**pre-registered** tensor已经正确写入了，然后**在确认数据已经安全地离开本地显存、到达目标节点之前，禁止 GPU 的下一个任务启动。** 
   1. 为什么需要这个**预注册**？这里涉及到网络传输的内容，锚定要用来网络传输的内存（不会交换到磁盘中）。
3. 阻塞GPU流，在数据传输完成之前，防止流中的下一个内核启动。（否则可能会覆盖掉当前正在传输的东西）
4. CPU communication library（Core Sender）接管任务，用RDMA write with immediate 来传输tensor
5. 轮询完成队列。确定数据已经写入remote buffer
6. 解锁GPU流。一旦确认传输完成，发送端通过更新 **共享内存标志位 (Shared memory flag)** 来解除对 GPU 流的阻塞。允许其他kernel重新使用这块注册过的内存区域。[也可以放在这个pre- regist的地方来发送]

### M2N Receiver

![image-20260329154329436](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329154329436.png)

流程如下：

1. 等待CUDA时间 
2. pre-registered tensor区域没有在使用，保证数据传过来有东西放
3. 阻塞GPU数据流，保证接下来的kernel不会运行，直到完成
4. receiver通过轮询完成队列，确认数据成功发送过来[由于数据是通过 RDMA 直接写入 GPU 显存的（由发送端触发），接收端需要检查本地的**完成队列 (CQ)**。]
5. 保证数据一致性。CPU Core Receiver 利用GDRCopy和flush 操作，即便数据到了显存，GPU 的二级缓存（L2 Cache）里可能还残留着旧数据的备份。通过 **GDRCopy** 库执行一个 **Flush（刷新）** 操作。它强制让 GPU 丢弃旧缓存，直接从显存读取刚刚收到的新鲜数据。
6. 解锁GPU数据流。一旦 Flush 完成，意味着数据不仅到了显存，而且对 GPU 的计算内核也是“可见”且“一致”的。

### Traffic-oriented Optimization

1. ACKs高优先队列
2. 拥塞控制微调：标准拥塞控制算法会激烈地限制传输速率，微调参数、加速收敛。

与DeepEP对比

1. DeepEP直接使用 GPU-to-GPU 通信
2. 如果专家数量减少，每个通信开销变小，那么GPU的并行处理能力能发挥到极致。

## Implement

1. Load balance. 基于expert的popularity部署副本；用贪心算法来算哪些专家值得多开几个副本。一个专家可以部署到多个node上，一个node上也可以有多个专家。**一个专家拥有多个副本，分布在不同的机器上。**

# Expert-as-a- service(EaaS)

这个还是使用GPU-to-GPU通信

>  ''NCCL [19] lacks suitable single-sided P2P operations, while RDMA-based libraries like NVSHEMEM [23] impose restrictive symmetric buffer requirements.''

在NCCL通信库中，缺少稳定的单边P2P操作。传统的 NCCL 必须先建一个“Group（组）”，所有人都在组里才能说话。**而文章这个库是 点对点、随用随建 的。**

在RDMA-based 的通信库中，要在多个GPU间通信，需要：symmetric buffer requirements。如果 GPU 0 申请了一块 1GB 的缓冲区，那么集群中**所有其他 GPU**（GPU 1, 2, ... N）也必须在它们各自的显存中，在**完全相同的位置**（或者相对于基地址相同的偏移量）申请一块同样大小为 1GB 的缓冲区。这样操作，多个GPU间通信是很快的（如果内存是对称的，GPU 0 想要往 GPU 5 的缓冲区写数据时，它根本不需要问 GPU 5“你的地址在哪里”。它只需要计算 `自己的地址 + 偏移量` 就能直接算出对方的地址。）

但是在MoE中，热门expert要求留出2GB内存，但是其他expert压根用不到。存在内存浪费、动态缩放困难、初始化开销大。

问题是 在初始时期就要初始化组

## previous work limitations

1. Coarse-Grained Elasticity
2. Brittleness and Low Fault Tolerance
3. Static and Inefficient Load Balancing：专家在GPU上的映射在服务初始化启动的时候就固定了。这样无法适应动态变化的workload

Stateful Attention Layers & Stateless MoE Layers

1. Independent, Fine-Grained Scaling 允许调整expert数量
2. Robustness through Replication. 如果某个expert出了问题，直接将requests送到它的replica上
3. Dynamic Load Balancing via Service Instantiation.  当system检测到某个expert成为hot spot 就动态实例化这些特定experts的copy到没用的GPU上。这些新实例可以立马处理requests。**这将负载管理从静态的部署前猜测转变为动态的自适应系统优化。**

## System Design

Standard Expert Parallelism

![image-20260330112854833](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330112854833.png)

Decoupled EAAS Expert Service

![image-20260330112920031](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330112920031.png)

1. **Attention Clients client's router** 将activation通过P2P通信分发给合适的expert
2. **Scalable Expert Servers** 每个server都有一些experts，并作为一个独立无状态个体仅专注于执行expert computation

### Shared Communication Buffer Design

![image-20260330120452694](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330120452694.png)

### Asynchronous Server Operation

server不停地轮询所有client buffer的state flags，直到确认所有buffer在client write down(flags=1)时，server把这些准备好要处理的requests合并一起，组成一个batch，这些token会被重组进行高效处理。完成后，server会写回加权求和后的结果到data payload中。最后，server会更新flag state到Server Computation Done，来告诉各自的clients可以取回。

### Fault Tolerance Mechanisms

![image-20260330121601146](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330121601146.png)

#### Client Failure

要把离线的client的buffer release掉。

#### Server Failure

通过对expert service的复制。monitor监测到server离线通知给所有client，或者client也可以独立推断server的failure-通过轮询时间超时。得到消息的client立即更新它的本地mapping，移除failed server，然后自动重传原来的requests去替代的server（有原要求expert的副本）

**这个机制对final inference outcome和持续鲁棒服务影响重大。**        why？

## Optimization

建立IBGDA connection（只是建立连接）-就是握手

![image-20260330133001700](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330133001700.png)

![image-20260330134111005](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260330134111005.png)

比起NCCL优势：1. 按需建立连接，点对点动态建立 2. 传统的静态通信组要求所有 GPU 在启动时必须全部到齐并同时初始化 。

### Dynamic Load Balancing for Experts







# 参考

1. MegaScale-Infer：https://dl.acm.org/doi/epdf/10.1145/3718958.3750506
2. EaaS：https://arxiv.org/pdf/2509.17863
3. EP：https://proceedings.mlr.press/v162/du22c.html
4. AFD参考：https://zhuanlan.zhihu.com/p/1952393747112367273
5. EaaS论文overview：https://zhuanlan.zhihu.com/p/1961085349108364663
6. MHA、GQA：
7. 补MoE知识：https://apxml.com/zh/courses/mixture-of-experts/chapter-1-moe-foundations-sparse-models/sparse-moe-paradigm
8. All2ALL: https://zhuanlan.zhihu.com/p/1976060556906624692
9. RDMA: https://zhuanlan.zhihu.com/p/55142557
10. 

