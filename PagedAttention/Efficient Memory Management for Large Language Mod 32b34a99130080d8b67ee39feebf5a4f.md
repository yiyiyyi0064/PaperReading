# Efficient Memory Management for Large Language  Model Serving with PagedAttention

原来vLLM是基于pagedattention搭建的。

# WHY

Memory-bound 内存受限

在使用LLM时，显卡内存是怎么分配的？

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image.png)

因为KV cache在Memory中占比很大，而且它是dynamic的，高效地管理它是一个很重大的课题。

传统system

1. 存储KV cache ：预分配一个连续chunk，大小为request的最大长度→【严重的internal fragmentation，其他requests不能用this part of chunk；External fragmentation 】
    
    <aside>
    💡
    
    内部碎片：已经被分配给某个任务，但**该任务用不掉**的那部分空间。
    
    外部碎片：存在于各个已分配块**之间**的空闲空间，它们太小、太分散，导致**无法装下任何一个新任务**。
    
    </aside>
    
2. 现存system不允许memory sharing。因为KV cache是存储在单独连续空间的。
    
    <aside>
    💡
    
    paralle sampling & beam search
    
    对一个request会有多种outputs，在这种情况下，其实这个request其实可以部分共享他们的KV cache的。
    
    </aside>
    

## Transformer-based & LLM Service

这两个part就是介绍LLM基于transformer这个背景，以及介绍prefill&decode这两个阶段的区别。

Prefill 的attention计算是可以parallel的，但是自回归生成的phase 不行，这也是浪费GPU和memory-bound的主要来源。

Batching Techniques for LLMs

# Page Attention

## Block

attention计算用的k、v vectors 存在block中，不连续的存储。

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image%201.png)

Block中存储的每个token是其单独的K、V，并不是有前文信息的KV。但是逻辑上，他们组成了一个完整的KV cache。

这样存储之后，attention的计算要如何实现呢？

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image%202.png)

个人理解：从表面来看，KV cache的结构没变，还是缓存起来等待attention计算。但是在真正取出这个cache进行计算的时候，和原来有区别，需要一个一个的去block中去取，来和新input的query来进行计算。又因为在存到block的时候是有顺序的，在block中也是按照顺序的，所以最后的计算也是按照原来的顺序的。

## Decoding with PagedAttention

这一段在论文中有很详细的解释。（但是懂了block怎么分的，这个看图就够）

唯一需要注意的是#filled 这个标识当前block还能不能继续fill的。

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image%203.png)

如图，对于第二次decoding，逻辑上需要一个新的block，vLLM也会在物理上为它分配block，并且在Block Table上进行mapping。

对于两个request的，每个block肯定得是各个request专用的。

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image%204.png)

**vLLM 的调度**

> Globally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching
> 

vLLM会从众多requests中选择，将它们进行batch，放在一起进行。

## Applications

**对于Parallel sampling，**

### Copy-on-write  写时复制

这个机制和OS中非常相似。

当共享同一个prompt的要产生新的token而要在block中写入时，显存会分配一个新的block，并将当前block的内容复制过去，在新的block中写入。同时，因为复制的只是当前block的内容（相比起原来很长的prompt还是省去了非常多的空间的）。

注意中间还有个reference count，就是指的是有多少个logical block指向当前physical block。当reference count降到1，尝试在这个block上写的sequence，可以直接写入，无需再复制到新的physical block。

### Beam search

写时复制在束搜索上的应用也很好。

![image.png](Efficient%20Memory%20Management%20for%20Large%20Language%20Mod/image%205.png)

### Share Prefix

system prompt 

prompt engineering

比如说要调用什么工具，调用工具的prompt就是system prompt，就是很多用户的share prefix。

这时paged attention就非常重要了。

## Scheduling and Preemption

当block不够了的时候，how to evict？

### **Which blocks should it evict?**

在传统os中，会将某个block移除（比如当前最少遍历or最远遍历）。但，考虑到LLM的特性，对于每个request如果移除某一个block，可能整个kv cache都没法使用了。同时，对于束搜索这种，multiple sequences within one request 被称为 **a sequence group** ，采取的策略都是 all-or-nothing，都是一起reschedule的。

### How to recover evicted blocks if needed again?

**Swapping**

将这些blocks复制到CPU memory中

同时当Preemption发生（GPU 显存被占满，但是还在不断生成token，有些sequence的blocks需要被evicted），vLLM就会停止接受新的requests，直到所有被抢占的sequences完成；一旦有request完成，被preempted的sequence就会被带回，继续产生下去。

<aside>
💡

Note that with this design, the number of blocks swapped to the CPU RAM never exceeds the number of total physical blocks in the GPU RAM

</aside>

**Recomputation**

如果不使用Swapping，直接把KV Cache丢掉，那就只能进行重新计算了。但是重算的时间比起原来的latency显著降低，因为只需要forward一次，并行相当于就是一次Prefill了。

### Trade-off

所以这两种方法应该选择哪一种呢？

取决于CPU RAM 和GPU memory的bandwidth以及GPU的算力。

## Distributed Execution

这部分也是在讲vLLM的。

# 思考

1. model weights可不可以放在不同的地方，要使用的时候再放一起，Parameters的组织方式也可以探讨一下。
2. 在Swapping时，有一个the number of blocksCPU RAM never exceeds the number of total physical blocks in the GPU RAM.这个地方是主动设计吗？
3. GPU worker 怎么并行计算的 这个部分知识要再补一补

# 参考

1. [PagedAttention](https://arxiv.org/abs/2309.06180)