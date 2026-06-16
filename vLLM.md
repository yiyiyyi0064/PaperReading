# Arch

## LLM Engine

![LLMEngine Diagram](https://docs.vllm.ai/en/stable/assets/design/arch_overview/llm_engine.excalidraw.png)

## AsyncLLMEngine

提供的LLMEngine是无法做在线服务的，调用一次 `LLMEngine.step()` 是同步且阻塞(Synchronous & Blocking) 的操作。

而使用库`asyncio`的AsyncLLMEngine就是一个异步调度层，它的核心任务是将网络通信（I/O 密集型）和 GPU 模型推理（计算密集型）剥离开来。

## Worker/ModelRunner/Model

![Class Hierarchy](https://docs.vllm.ai/en/stable/assets/design/hierarchy.png)

Worker直接受到上层LLM Engine的管理，而Model Runner封装在Worker内部，负责加载和运行模型（数据转换Tensor、CUDA Graph），Model Runner对象内部都会有Model对象，就是标准的模型层。

# Design

## Attention Backend Feature Support

可以支持手动选择attention backend

## CUDA Graph

Tutorial: https://zhuanlan.zhihu.com/p/699754357. &   https://zhuanlan.zhihu.com/p/700224642

## CustomOp

可以支持调用forward底层计算时，根据当前运行环境自动路由到正确的底层实现。

同时，也可以决定是否使用自定义算子，也支持对部分自定义算子的精准管理。这些算子都是直接操作底层显存的cpp算子。

### Guidelines for Implementing a New CustomOp

可以自己手写新算子，就是二次开发！

具体步骤：

1. 编写底层CUDA。实现一个新的算子类，该类继承自 CustomOp 基类。
2. 在该算子类上添加 @CustomOp.register ("op_name") 装饰器，以将其注册到自定义算子系统中。
3. 根据您的需求实现不同的 forward_xxx () 方法。

实际中使用流程应该是这样：

1. **写 C++ / CUDA：** 硬件工程师用 C++ 写了一个厉害的 `flash_attn` 算子。
2. **TORCH_LIBRARY 绑定：** 用 C++ 宏把这个算子注册成了 `torch.ops.vllm.flash_attn_varlen_func`。
3. **写 CustomOp 包装（你贴出的代码）：** Python 工程师写了这个 `MMEncoderAttention` 类。在它的 `forward_cuda` 方法里，调用了上面那个 `torch.ops`。
4. **OOT 机制：** 如果你是第三方厂商，你的硬件既不是 CUDA 也不支持 FA，那你就可以继承这个类，重写一个 `forward_oot`，在里面调用 `torch.ops.your_company.your_hardware_kernel`。

## Dual Batch Overlap

这个是针对MoE在分布式推理中提出的并发优化技术，目的是将ALL2ALL通信隐藏在GPU计算中。

具体实现：在`modelrunner`中将当前的batch对半切开，同时开启两个CPU worker线程，然后在这两个worker线程上跑。这两个线程各自带着一半的数据，同时开始执行模型的前向传播。[p.s UBatch就是微批次]

启用 DBO 时，`FusedMoEModularKernel` 内的让步点可让两个 CPU 工作线程（亦称 UBatch 线程）交替执行，从而实现一个线程进行计算时，另一个线程等待通信完成。[否则两个worker会一直争抢GPU资源]

PS DBO强绑定DeepEP 用DeepSeek没有写4090硬件支持

使用指令：

```bash
vllm serve  /root/share/models/DeepSeek-V2-Lite \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --port 20101
```

## DBO Component



# How to debug the vLLM-torch.compile integration

如何进行一步一步排查？

**降级级别 1（彻底关闭加速）**：

- 命令：`--enforce-eager`
- 效果：关闭 `torch.compile` 和 `CUDAGraphs`，退回到最基础的 PyTorch Python 模式。如果此时代码能跑通，说明模型逻辑没问题，绝对是底层编译系统背锅。

**降级级别 2（仅关闭图编译）**：

- 命令：`-cc.mode=0` (或 `--compilation-config.mode=0`)
- 效果：仅关闭 `torch.compile`，保留 CUDAGraphs。

**降级级别 3（仅关闭 CUDAGraphs）**：

- 命令：`-cc.cudagraph_mode=NONE`
- 效果：保留编译，但关闭底层的显存流打包录制。

**降级级别 4（仅关闭 Inductor 后端优化）**：

- 命令：`-cc.backend=eager`
- 效果：保留计算图的捕获，但不去生成底层的 C++ Triton 代码。

## 图编译加速流程

![vLLM-compile diagram](https://docs.vllm.ai/en/stable/assets/design/debug_vllm_compile/design_diagram.png)

使用诊断工具：tlparse 设置环境变量 `TORCH_TRACE=~/trace_dir` 运行 vLLM 后，系统会把编译器的每一个思考步骤都记录下来。用 `tlparse` 可以直观地看到编译出的底层长什么样，这对于写 CustomOp 却发现无法正常编译的人来说是救命神器。

## Fused MoE Modular Kernel

传统的 MoE CUDA 核函数（Kernel）通常非常臃肿且难以维护，因为它们需要同时处理多种量化格式（如 FP8, INT8, BF16）、不同的专家数量（Expert Count）以及不同的 Top-K 选择。vLLM 引入 `ModularKernel` 的核心思想是：**将复杂的 GPU 计算逻辑拆解为相互解耦、可替换的组件。**

### TopKWeightAndReduce

先由Gate计算得出logits选择K个专家，然后根据在route时算出的权重进行加权后，将加权后的向量相加(即Reduction)

### FusedMoEPrepareAndFinalizeModular

![FusedMoEPrepareAndFinalizeModular Blocks](https://docs.vllm.ai/en/stable/assets/design/fused_moe_modular_kernel/prepare_and_finalize_blocks.png)

这里涉及到Quantization？什么是量化？其实就是会有精度的调整变化，在进入每个module前都会有量化的调整。**为了对齐参数权重。**

`prepare_no_receive`是一种异步操作，不会强制停下等待workers的结果。好处是：可以将work与all2all通信交错；比如将shared experts与fused experts交错。

**`finalize` (收尾)**：拿到所有数据后，`finalize` 负责执行 `All2All Combine`（把各个 GPU 算好的碎片拼成完整的输出），并决定是否在这里应用 Top-K 的路由权重将它们相加。

## FusedMoEExpertsModular

专门负责执行矩阵运算的“GPU 计算引擎”

![FusedMoEExpertsModular Blocks](https://docs.vllm.ai/en/stable/assets/design/fused_moe_modular_kernel/fused_experts_blocks.png)

#### apply() 

核心执行流水线 MoE Forward Pass 底层流程

1. Permute 内存重排/置换 
2. Matmul with weight W1 MLP第一层projection
3. Act + Mul
4. Quantization 转换精度以进行下一步计算
5. Matmul with weight W2    MLP的第二层 down 降回原来dim
6. Unpermute
7. Maybe TopK Weight Application + Reduction

#### workspace()

FusedMoE会执行一系列操作，如果分别为这些输出创建空间会低效，为了高效，要声明2个workspace shapes，该信息用于在 FusedMoEModularKernel::forward () 中分配工作区张量与输出张量，并传递至 FusedMoEExpertsModular::apply () 方法。

**这些工作区随后可在融合混合专家模型（FusedMoE）的实现中用作中间缓冲区。**

### 二次开发方法

没仔细看。

## Hybrid KV Cache Manager

BackGround: attention types不同，system是如何分配内存的。

kv_hidden_size: The number of bytes to store one token's KV cache for a single layer. 在一个layer中，保存1个token的（Key向量和Value）所需要消耗的绝对物理显存大小。

#### Sliding Window Attention(SWA)

无论总文本多长，在计算当前token时，最多只回看最近的4096个Token，更早的token被无视。

![image-20260423145240218](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423145240218.png)

在SWA中怎么实现prefix Caching？

不使用`round-robin`这种轮询的方式，因为这样的话，一开始在显存中只被分配了固定大小的block，后面生成的token会占掉前面的block，这样无法进行reuse。所以，使用的是：将不同的tokens分配到不同的blocks中，并且将sw之外的blocks去free掉。

会从右至左去check cache，因为越早的match那么得到的缓存碎片就会越新（靠右），那么hit length也就越大。

![image-20260423153508657](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423153508657.png)

这与我们的full attention是不同的：从左至右去遍历，知道fail。

### sliding window attention & full attention models 混合

必须使用intersect ：

1. 在full attention中先找到longest cache hit
2. 在swa中从full attention得到的cache hit length开始从右往左遍历cache hits。

exit early这样就可以保证两种attention都找到对应cache。

这种方法还可以应用到所有 full attention + x



## Logits Processors

vLLM中logits processors是按照batch粒度来操作的。某些是有状态的stateful，比如“重复惩罚处理器（Repetition Penalty）”，它必须记住每一个请求在过去的轮次中**已经生成过哪些词**，才能在当前轮次对这些词进行打压。

两类：

1. **argmax-invariant logits processor**：（Min-P Top-P），不会调整argmax。如果是这种不会修改最高最高分数的，且正好使用的是greedy decoding，那么就可以直接跳过这种processor，因为不会对最后结果产生影响，节约算力。
2. **non-argmax-invariant logits processor**：（强制结束符、重复惩罚）这种会对最高分数产生影响，那么肯定不能跳过。

同时注意：因为粒度是batch，那么要跳过也对整个batch中的request都是greedy 才行。

Scheduler重新分配batch之后，logits processor要处理batch中的request前得先对batch中的metadata进行更新。

Example 多种情况

#### 如何在vLLM中引入新的Logits Processors？

apply & update_state

内置的Logits Processors(基于以上的programming model实现的几种方法）：Min-P Logit bias Min-tokens（需要新增时参考）

以下是还未按照programming model实现的几种支持的sampling 方式

![image-20260423190003167](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423190003167.png)

## Multi-Modal Data Processing

处理多模态输入，兼容HuggingFace。

### Prompt Update Detection

HF会把原本的 1 个 `<image>` 替换/膨胀成 256 个连续的 `<image>` 占位符，这个动作就叫 **Prompt Update**。

vLLM做法：

为了支持高级优化（如 Chunked Prefill 和 Prefix Caching），vLLM 必须**精准地知道图片特征到底对应在句子中的哪个位置**。因此，vLLM 引入了 `_get_prompt_updates` 机制，自动检测并记录 HF 处理器到底把 Prompt 改成了什么样，从而建立起“占位符”与“真实多模态矩阵”之间的完美映射。

### Tokenized Prompt Inputs

HF processors的处理步骤：

1. 先tokenize text
2. 再处理multi-modal inputs
3. 最后prompt updates

但是vLLM要求：

- For text + multi-modal inputs, apply all steps 1--3.
- For tokenized + multi-modal inputs, apply only steps 2--3.

HF不支持tokenized + multi-modal。所以只能自己完成step3.

可以使用Dummy text（根据image来生成文本）来避免输入占位符令牌数量与多模态输入数量不匹配，系统抛出错误问题。

**Automatic prompt updating（自动更新）**：等图片特征拿到手后，vLLM 自己接管第三步，通过 `_apply_prompt_updates` 函数，把那些膨胀后的占位符，手动强行插入到之前已经准备好的 Token IDs 序列中，完美拼装出最终的输入！

### Processor Output Caching

有的model的mm数据hf处理很慢，所以vLLM会将mm数据cache起来，避免重复加载同样的mm input。[如何判断是否是同一张图片？pixel-level hashing]

## Python Multiprocessing

处理Python多进程时面临问题。正在考虑做一个独立的进程管理器。

## P2P NCCL Connector

An implementation of xPyD with dynamic scaling based on point-to-point communication, partly inspired by Dynamo.

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

### KV Cache Transfer Methods

PUT_ASYNC → GET → PUT

### P2P Communication via ZMQ & NCCL

只要获知对方设备的地址，即可完成点对点键值缓存传输（采用 NCCL），不受进程编号与全局进程数的限制，以此支持具备解耦部署架构的推理 / 解码实例动态扩缩容（扩容与缩容）。这意味着新增或移除推理、解码实例时，无需对整个系统进行重启。

每个P/D实例都需要创建一个单独的`P2pNcclEngine` instance来维持ZMQ Server（用来监听`zmq_addr` address和接收其他instaces的control flow requests）

PD间要传输KV Cache要先建立ZMQ connection和NCCL Group [持续的传输就重用]

### NCCL Group Topology

每个NCCL group都会拥有一个确定大小的GPU memory buffer用来通信，大小主要由by the `NCCL_MAX_NCHANNELS`决定。

Future work：RDMA and UCCL

memory buffer存不下，就存在Tensor memory中，但读写要通过PCIe。





## Metrics

主要讲的是：统计了哪些指标，这些指标是怎么统计的。

### Intervals 计算

基于engine core来记录这些event的时间戳

![Interval calculations - common case](https://docs.vllm.ai/en/stable/assets/design/metrics/intervals-1.png)

### Fronted Stats Collection

### KV Cache Residency Metrics

### Autoscaling and Load-balancing





## Paged Attention

底层GPU显存读取：向量化访存

有一些新概念：

1. Warp：一个warp里有32个threads，同步在一个SM上处理，在这个kernel上，每个warp处理query token与一个block中的整个key tokens计算。
2. Thread block：是a group of threads，可以共享同样的shared memory，每一个thread block有过个warps，而且在这个kernel中，每一个tread block会计算一个query token与整个上下文的key token(就是当前全部block的k)
3. Grid：是a collection of thread blocks，在这个kernel中， `(num_heads, num_seqs, max_num_partitions)`。因此，每个线程块仅负责处理一个注意力头、一个序列以及一个分区的计算。
4. Vec：thread每次fetch data的单位，query/key的`VEC_SIZE `为4，value的`V_VEC_SIZE`为8。
5. **Thread group**：是一个小的线程group。

### Query

底层的Query存储在memoy中是线性存储的，要找到对应request的对应head的query。

如何找到q存储的位置：

```c
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

如何将这些数据搬运到寄存器中：

```c
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

有thread-group-size个thread来一起fetch query，如果一个thread的vec-size为4，那么在head-size为128的情况下，就需要32个vecs。所以当thread-group-size为2的情况下，分为2个row，每个thread处理16个vecs。

![q_vecs](https://docs.vllm.ai/en/stable/assets/design/paged_attention/q_vecs.png)

### Key

```c
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

在不同的iteration中，k_ptr会指向不同的key token。

在query的取址中，不同thread是同一个ptr地址。 

![key](https://docs.vllm.ai/en/stable/assets/design/paged_attention/key.png)

### QK

具体的workflow

```c
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

1. 准备Query：将其存入Shared Memory
2. 外层循环：遍历历史数据块，每一次进入这个外层循环，就是找到数据块的具体位置。
3. 内层循环：找到数据块之后从中取出Key数据，由thread-group-size个thread来拉取到register中。
4. 点积与归约：Thread 0把对应head的Query与对应head的K与

### Softmax

就是flashattention

## Automatic Prefix Caching

vllm method: **hash-based** 

使用sha256 可以避免collision

处理多模态inputs

### Cache isolation for security

隔离prefix cache的重用，使用optional per-request salting，保证有同样salt的才能重用cached KV blocks。（来防范timing-based attacks 根据latency来推断cached content）

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

这样设置的话，cache sharing就被限制在同一个salt中。

### Data Structure 

```python
class KVCacheBlock:
    # The block ID (immutable)
    block_id: int
    # The block hash (will be assigned when the block is full,
    # and will be reset when the block is evicted).
    block_hash: BlockHash
    # The number of requests using this block now.
    ref_cnt: int

    # The pointers to form a doubly linked list for the free queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

基本就是building block。

![image-20260423133923165](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423133923165.png)

注意，cache只会cache full blocks！！！

v1中，会有重复块的存在，不会再将重复的block给free掉。

free中，首先会将request的最后的block放到free queue的tail队尾，因为它被reuse的可能是最低的。这时block中还是有历史数据的。

Eviction（RLU）

当free queue队列满了or request需要新的block，就会从队列的头部（最近最少使用）淘汰掉block，给其他cached block。淘汰要把block ID（物理id）以及block hash（逻辑id）全都清空。

文档中有例子，看完更好地理解了。

特别是这里：block2、3是还被cached，但是block4由于没被加入cache blocks dict所以没有cached。

![image-20260423140331004](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423140331004.png)

最后结果：

![image-20260423140553875](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260423140553875.png)

从queue中取出缓存的部分，再evicate后重制的block，这两个部分。

## 大规模vLLM部署中的重难点

Kubernetes + 容器化部署

### vLLM状态监测

1. `/health`以及`metrics`端口
2. 核心`metrics`监测
   1. Num_waiting_requests > 0 处理不过来就会排队
   2. Num_running_requests 观察节点压力
   3. SLO related 服务用户 TTFT decoding_throuhout
3. 实战：Prometheus+Grafana UI配置

### 负载均衡和路由算法

1. 负载均衡：如何判断overload？-> 
2. 路由算法：session-based / 最大前缀匹配（分成block去看）

### 自动扩容/缩容

### 容灾处理

vllm挂了怎么办？

业务层异常捕获+动态更新路由+重新发送请求

如何把已经跑了一半的request迁移到另一台上？











