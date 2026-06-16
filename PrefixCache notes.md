接下来的几篇文章都是与KV Cache相关的内容。

# Prefix Cache

SGLang (Structured Generation Language)

## RadixAttention

使用LRU替换策略

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202506051025880.png)

少样本学习few -shot learning

各node颜色表示状态：绿色-新加入的点、蓝色-当前时间点访问到的缓存节点、红色-已经被淘汰的点。

具体流程：

1. 一开始radix tree 是空的
2. 服务器接收到用户消息 `"Hello"`，并生成 LLM 回复 `"Hi"`。system prompt `"You are a helpful assistant"`、用户消息 `"Hello!"` 和模型回复 `"Hi!"` 被整合为一条边，并连接到一个新节点。
3. 一条新的prompt到达，server找到prompt的prefix，并且复用它的KV cache
4. 新的chat session开始，node b （3）中被分为两个nodes，允许两个chat sessions共享一个一个system prompt。
5. 新的chat session继续，由于memory limit，node c必须被淘汰。新的turn将会在node d后添加。
6. server接收到few-shot learning queries，插入树中。root node被split，因为在先存的node中没有任何prefix
7. 此时，server又收到其他的few-shot learning queries，他们共享同一些few-shot examples，所以我们分出node e去共享他们。
8. 此时server又收到来自第一个chat的一条新消息，淘汰第二个chat中最近没被用的node。
9. server又收到request去为node j中的问题采样更多答案，像self-consistency prompting。为了腾出空间，我们淘汰node i k l。

### cache-aware scheduling algorithm

> In the batch-processing setting we sort the requests by matched prefix length and prioritize requests with longer matched prefixes instead of using a first-come, first-served schedule. Alg. 1 (Appendix) shows the pseudo-code for cache-aware scheduling with contiguous batching. The algorithm uses longest-shared-prefix-first order. In more latency-sensitive settings we may still be able to tolerate limited batch re-ordering to improve cache reuse. Additionally, we prove the following theorem for optimal scheduling in the offline case.

不再是FCFS而是谁能重用更多缓存，谁就插队先排。



## compressed finite state machine

![image-20260331141539473](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260331141539473.png)

压缩有限状态机（FSM），大幅度提升大模型在生成结构化数据时的速度。

为什么要把regex argument转换成FSM呢-**Guided Decoding**





### 传统做法

通常会将正则表达式转换为一个有限状态机，每生成一个token，system都会检查FSM的当前状态，看看下一个允许出现的单词是什么，然后把不允许出现的单词概率设置为0。

e.g. 假如要输出 `{"summary": "` 这个在逻辑上是固定的，但是传统做法仍然需要跑几次次完整forward才会输出得到最后结果。

### SGLang做法

既然这些路径是唯一的，为什么不让model直接写完呢？

SGLang会分析FSM，找到那些只有唯一后续路径的相邻边，并把**它们压缩为一根长边**。在decoding中，如果遇到了这种压缩边，SGLang可以直接在一个forward中生成这一串token，不需要反复确认。

## API speculative execution

对于只能调用API的黑箱，可以加速执行并且减少重复调用的API cost。

> In SGLang, we can enable speculative execution on the first call and let it continue the generation of a few more tokens by ignoring the stop condition. The interpreter keeps the additional generation outputs and matches and reuses them with later primitives.

这里有点promt engineering了。

SGLang 在发第一个请求时，虽然程序里写了 `stop="\n"`，但它会告诉 API：**“先别管那个换行符，多写一点再停。”** 

由于大模型非常擅长模仿格式，它写完“张三”后，很可能会自动猜到你后面要干什么，直接写出：张三\n职业：工程师

# KV主机卸载

CachedAttention 重用KV cache

## Challenge

1. High KV cache access overheads.
2. High storage capacity requirement of KV caches.
3. Suitable placement of KV caches in different hierarchies.
4. Unexpected invalidation of the saved KV caches.

## The CachedAttention Design

### Overlapped KV Cache Access

Layer-wise Pre-loading from Memory to HBMs

这是load读取阶段。主要思路是：overlap the loading of the KV cache with the prefilling computation of new input tokens。LLM 是由多个layers连接成的，每个layer都有自己的KV cache。这一层在进行计算的时候，下一层可以把需要的KV cache加载进去。

为了进一步消除最后一个工作和当前工作第一层之间的gap

，Cached Attention保留了HBM read buffer。

### Layer-wise Pre-loading from Memory to HBMs

![image-20260401085301792](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401085301792.png)

通过增大buffer，可以让pre-loading提早开始。

### Asynchronous Saving from HBMs to Memory

![image-20260401090351100](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401090351100.png)

baseline method：在conversation结束后将所有KV cach一起存进去 -> 导致gap

使用overlap去异步的存KV cache，同时prefill和decode阶段有不同特性采用不同的mechanism。

1. Prefill阶段：
2. Decoding阶段：如果decoding结束了但是KV cache还没完全写回，我们保留HBM write buffer（read buffer）。这个没完成的KV cache将暂时移去write buffer去避免阻碍下个job的执行。

## Hierarchical KV Cache Placement

分层KV cache的放置（**这不就类似OS里的存储结构嘛**）

### Scheduler-aware Fetching from Disks to Memory

CachedAttention 利用了host的memory和disks去拓展KV cache的可用空间。

DRAM： 对host memory访问的速度 （比对disks的快很多）

为了取KV cache的尽量从memory中取，CachedAttention采用了**scheduler-aware fetching scheme**，提前将KV caches从disks中移到memory。

![image-20260401091636915](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401091636915.png)

### Scheduler-aware Eviction from Memory to Disks

当host memory中的空间耗尽，需要淘汰某些KV caches到disks中。如果disk也满了，那就需要把他们淘汰出system。

不同于FCFS or LRU，Eviction使用的是**scheduler-aware eviction scheme**

## Decoupled KV Cache Truncation

如果historical tokens超出了context window的limitation，LLM seriving engines将会实施token truncation。

但是这种truncation会让存储在CachedAttention中的KV cache失效，因为每个token的位置将会改变。

为了解决这个问题，CachedAttention通过解耦位置编码。

> CachedAttention needs to work with the relative position encoding (RPE) [42, 44, 54]. Unlike the absolute positional encoding (APE) in which positional encodings are added to the input, RPE directly embeds the positional encodings in the query (Q) and key (K) vectors

**传统的编码**是使用APE（绝对位置编码），在进入Attention层前，就直接把位置向量加到input embedding上。[**后果**：如果把前面的词删了，后面的词的绝对位置信息也将变化，原本的KV cache也将invalid]

文章使用**RPE（相对位置编码）**，单词向量进入attention层时并不带位置信息，位置信息是在计算Q和K的过程中实时注入的。[关注的是词与词之间的相对关系]

文章使用的解耦，存入KV cache的是没有位置信息的，是在取出之后再去注入位置信息。

![image-20260401103244205](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401103244205.png)

使用CachedAttention truncation流程

![image-20260401103553773](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401103553773.png)

同时，CachedAttention与KV Cache Compression策略协同工作。（缓存优化与缓存压缩）

传统的缓存是将所有生成的token都存起来，现在CachedAttention可以选择性保存找重要的token。哪些是重要的呢？句子开头的token or 计算过程中得分较高的tokens

KV cache Compression technique会提供token discarding  list（TDL）的创建方法。然后Cached Attention就会读取这份TDL，然后立即将TDL上的从AttentionStore里KV cache丢弃。最后，AttentionStore将会将这份pruned KV cache送给inference engine。



# Mooncake 分离KV

## Mooncake Architecture

![image-20260401155442165](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401155442165.png)

对于一个request：

1. 首先将尽可能多的可重用的KV Cache transfer到prefill instance中来重用
2. 在chunks/layers 中完成prefill stage，然后输出对应KV Cache给decoding instance （给某一层对应的decoding节点）
3. 加载KV Cache并且将request加入continuous batching 处理中

Mooncake使用的是PD分离架构，但是它与chunked prefill比较在大规模实战中的可用性还有待商榷。

Mooncake也是一个使用分布式KV Cache pool来在不同chat sessions和queries中共享KV cache的system。

## Design of Mooncake

Typical Workflow of a request

![image-20260401162033674](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401162033674.png)

1. KVCache Reuse:   被选中的prefill node(group) 将会收到raw input、可重用的prefix cache的block key（来自CPU memory加载到GPU memory）、以及分给这个请求分配的完整缓存的数据块键值(就是复用的prefix+新计算的)[这里的调度后续会具体提到]
2. Incremental Prefill: Prefill 节点会利用已经加载好的历史前缀缓存（prefix cache）作为基础，去计算用户输入中那些**全新、未被缓存过的 Token** 。计算完成后，模型会为这些新 Token 产出新的 KVCache（即“增量 KVCache”） 。接着，节点会把这部分新产生的数据存回（回写）到 CPU 内存中，放入全局的 MOONCAKE Store 分布式池里，以便下一步传输给 Decoding 节点 。但如果当前input非常长超出某个阈值（根据GPU计算性能），prefill stage就会将它分成多个chunks来流水线处理。【使用了pd分离后还使用chunked原因后续会解释】
3. KVCache Transfer: 传输KV cache会异步执行（不是一块一起而是分开），而且会与prefill step重叠，来节约时间。每个节点都部署了 MOONCAKE Store 来管理和传输这些cache。
4. Decoding: 当decoding node的CPU memory收到所有KV cache，request会用continuous 的方式加入batch。这些decoding node都是由Conducor基于TBT SLO选出的。

### MOONCAKE Store: Cache of KVCache

所有KV Cache都以paged blocks存放在分布式cache pool中。

1. 每个block都是由一个哈希key来标识的，这个hash key是由它自己的hash以及它prefix共同生成的。【因为单独一个block一样是没用的，要重用必须要求当前以及前缀的KV Cache都一致才行】
2. 对于hot-node：有些缓存块可能会被频繁调用，如果全网所有的请求都去同一个节点拉取这个热点缓存，那个节点就会因为网络拥堵而产生很高的延迟。为了缓解这种访问延迟，MOONCAKE 会通过缓存负载均衡策略，将同一个哈希键（即同一个热点 KVCache 数据块）在多个不同的物理节点上**复制多份副本（replicas）** 。这样，当大量并发请求到来时，系统就可以就近或分散地让不同节点提供这份缓存，避免单点过载 。

Transfer

1. Endpoint management 按需建立端点间的连接，创建端点池以及端点淘汰算法。

### MOONCAKE’s Prefill Pool

是否使用Prefill pool是有争议的。

## Scheduling

### Prefill Global Scheduling

不仅考虑load 还考虑prefix cache的hit长度以及可重用KV Cache blocks的分布。

对于每个新的request，有了匹配的prefix cache长度后，Conductor会估计相应的计算时间，然后再与估计的等待时间相加得到在某个instance上的TTFT时间。最后，Conductor再将request分发给有最短的TTFT的instace同时相应地更新对应instance的cache和queue的时间。

### Cache Load Balancing

heuristicbased automated hotspot migration scheme

**如果拥有“最长匹配缓存”的那台机器（最佳节点）因为请求太多、太忙了，系统该怎么办？** 

Conductor会把cache的位置和request都发给替代节点（如果估计的额外prefill 时间会比transfer时间短）。这个节点会检索KV Cache并且存在本地；如果这台替代机器本地其实已经有一部分历史缓存了，且远端节点拥有的最长缓存并没有比本地的长出很多（未超过预设的倍数阈值）节点会优先自己重新计算。



Mooncake function

1. 管理KV Cache storage scheme 以及 eviction policy

# 参考

1. Guided Decoding：https://zhuanlan.zhihu.com/p/31572085999
2. Prefix Cache：
3. 参考notes：https://se7en.mintlify.app/talks/03-prefix-caching
4. KV主机卸载：https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf
5. RPE相对位置编码：https://zhuanlan.zhihu.com/p/105001610