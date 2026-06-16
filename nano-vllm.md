设置转发

ssh -NR 192.168.1.152:20100:127.0.0.1:7897 yww@192.168.1.152

# 学习顺序

- [x] nanovllm/models/qwen3.py
- [x] nanovllm/layers/embed_head.py
- [x] nanovllm/layers/layernorm.py
- [x] nanovllm/layers/linear.py
- [x] nanovllm/layers/rotary_embedding.py
- [x] nanovllm/layers/attention.py
- [x] nanovllm/layers/activation.py
- [x] nanovllm/layers/sampler.py
- [x] nanovllm/engine/llm_engine.py
- [x] nanovllm/engine/model_runner.py
- [x] nanovllm/engine/sequence.py
- [x] nanovllm/utils/context.py
- [x] nanovllm/engine/scheduler.py
- [x] nanovllm/engine/block_manager.py
- [x] 看懂nano-vllm的剩余代码
- [x] 结合flashattention的代码，思考nano-vllm是如何使用flashattention的接口的

# Qwen3.py

## Qwen3Attention

Transformer的核心：实现多头注意力机制[含**分组查询注意力机制GQA**]

这个部分需要补一下：

1. RoPE: https://zhuanlan.zhihu.com/p/647109286
2. RoPE缩放机制
3. GQA 分组查询注意力机制: https://zhuanlan.zhihu.com/p/686149289



一些**基础量**的理解：

`hidden_size` = `embedding_dim`   model会用hidden_size个维度去描述一个token的含义，那么后续使用不同head来分开的时候，某个head都会处理某些特定维度的信息含义（每个维度有不同的信息），最后head处理得到的是该token某些特征。

`total_num__heads` 表示总的head数量；`num_heads` 表示每个进程负责的头数

`total_num_kv_heads` 表示总的kv head数量；`num_kv_heads` 表示每个进程负责的KV头数



### GQA

```python
tp_size = dist.get_world_size()   #获取全局进程数 这里应该是指model parallel的进程数
self.total_num_heads = num_heads  #设置总的注意力头数
assert self.total_num_heads % tp_size == 0 #保证注意力总头数可以被进程数整除
self.num_heads = self.total_num_heads // tp_size #每个进程负责的注意力头数
self.total_num_kv_heads = num_kv_heads  
assert self.total_num_kv_heads % tp_size == 0 
self.num_kv_heads = self.total_num_kv_heads //tp_size 
self.head_dim = head_dim or hidden_size // self.total_num_heads
```

如果是一般的MHA，num_heads和num_kv_heads是不需要单独分开设置的。在nano-vllm中实现了GQA。

可以注意到这里使用了一个**短路：**当外界没有特别传入head_dim时，就会默认设置为 `hidden_size // self.total_num_heads`

最后还有个缩放因子 `scaling` 就是那个   $ \sqrt{d_k} $

`v = v.view(-1, self.num_kv_heads, self.head_dim)` 这个V，KV head 的数量与每个头的维度可以不同，1个KV头给多个Q头用。

### RoPE

计算流程：1. input先不加绝对位置编码，生成Q、K、V  2. 对Q、K进行旋转，注入位置信息  3. 再让QK互相做点积，得到Score

在Qwen3中，使用的是dim=128，那么实现中首先分为64组（因为2个dim构成一个平面，这样就可以应用推出来的公式），而这64个平面旋转的速度（角度大小）是不同的，model为这64个平面分配64个不同的基础角度 $\theta_i$ 。（m*theta就是表示第m个位置的信息）

![image-20260402094942807](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402094942807.png)

QK点积后得到：

![image-20260402100139827](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402100139827.png)

在实际计算中实现：

![image-20260402095324462](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402095324462.png)

可以先从二维的推起验证一下，然后推到128维，64个平面每个平面中都进行了旋转，这样计算之后就是128维的Q_rotate。最后再让旋转后的Q和K进行点积，这时候计算得到的结果就是64个平面的相对位置信息(n-m)的和。

但是还是涉及到的：max_position 以及 rope_scaling

### **RoPE缩放机制：上下文长度扩展**

max_position  指的是RoPE 预计算/支持的最大位置长度（最大 token 位置索引上限）

rope_scaling 指用于长上下文 RoPE 缩放

### RMSNorm

均方根归一化，为什么呢？Q和K的点积容易产生极端大的数值，所以在进入attention之前先把Q和K进行归一化，避免极端值。

## MLP

这与Transformer中简单的Linear -> ReLU -> Linear结构不同，这里使用的是**SwiGLU**，同时还使用分布式计算。

**Gate_up_proj Tutorial Link: **https://zhuanlan.zhihu.com/p/650237644

其实就是**SwiGLU**的实现，门控机制就是用来控制信息能够通过多少。

![image-20260402112039643](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402112039643.png)

其中特殊符号就是逐元素相乘。

在Qwen3中，直接使用了 `gate_up_proj` 将本来的gate和up要算两次矩阵乘法合成为一次[就是把矩阵拼在一起算一次]；然后送到 `act_fn` 也就是SwiGLU逐元素相乘（就是用gate来决定要多少信息）。最后再通过 `down_prj` 把高维特征压缩回标准的hidden_dim。

这里的intermediate_size 就是要升维的dim，里面有个拼接操作。

```python
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2, #升维到intermediate_size的两倍，一部分用于gate，一部分用于up
            bias=False,
        )
```

以上是实现部分，但很自然的会有疑问：它将这个运算拆分成了几个函数，**为什么不分开算or为什么要把这几个计算放在一起呢？**

Python 层面做好的封装，实际上是在显式地、强行地指引框架，去调用那个底层早就由人类专家写好的、高度优化的 CUDA 算子。

![image-20260402114346842](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402114346842.png)

## Decoder

![img](https://picx.zhimg.com/v2-b1284bd833708de368203be4e187ad83_r.jpg)

这里就是把前面几个part放在一起实现decoder layer。[和原来transformer的decoder很大不同了]

特别是，最后在每一层进入之前，都要将hidden_states 与 residual 来进行合并归一化。[第0层，只有hidden_states 自己归一化]

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

## Model

```python
for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
```

用一个for来遍历通过每一层decoderlayer，这里的position(每一层是一致的，旋转的信息，但是每层的hidden_state不同)用于RoPE旋转编码（上面提到过的），里面提供的就是每个token的位置信息，便**于后续旋转**，比如decode阶段的token，position会提供正确的位置信息，来与之建立正确的空间关联。

## ForCausalLM

这里涉及的就是把model输出的output转换回词表维度，同时经过softmax得出下一个词的预测。

这里比较特殊的是：

```python
#这个操作就是在做点积 但是由于矩阵太大了 采用词表并行的方式 并行计算
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data

```

这里决定是否使用**权重共享**，

1. 如果使用权重共享，那么这个输出翻译head就会和输入翻译head的权重一致，那么在显存中就不需要分两块空间。最后，在compute_logits中将model的output与lm_head中词表进行点积，去计算相似度，最后再得到的就是与每个token的相似度得分。
2. 如果不使用权重共享，那么最后这个lm_head使用的就是训练过程中得到的专门用于预测token的新矩阵，与输入端不同，预测计算是一致的。

在整个框架中， 涉及到了多次**并行化**操作，目前还没有具体去看它们的实现，在接下来的具体实现中将重点分析。将model部分看完后，将附上跳转link，更好的理解。

# Embed_head.py

## VocabParallelEmbedding

这是看到的第一个并行操作实现：词表并行。

```python
def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        #这里使用VocabParallel 词表并行
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        #堆了config.num_hidden_layers 个 decoderLayer
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        #在模型输出前进行最后一次归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

```

在Qwen3.py的Qwen3Model中使用了这个函数，作用是将token转换成hidden_size维，做embedding。

```python
def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

这里其实涉及到了并行的通信，比如这个all_reduce，要读懂它们的通信，从这个开始应该是再好不过。

一开始初始化时，就会将vocab中总的token均分给GPU进程，每个GPU处理对应部分的token，进行embedding。

```python
if self.tp_size > 1:
    mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) #检查x中每一个token [False, True,...] 
    x = mask * (x - self.vocab_start_idx) #False 处均为0
    y = F.embedding(x, self.weight) #拿着得到的位置信息去查表 
    #拿到对应向量 但是0是不需要的
```

`y = mask.unsqueeze(1) * y` 把查出来的脏数据，再与mask相乘不会影响后面allreduce，allreduce后每个token上只有自己的真实数据，不会有其他干扰。

NCCL会启动一个同步任务，所有参与计算的进程GPU会将张量y进行按位置求和，得到结果后会进行广播，会被立刻分发给参与计算的每一个显卡。



## ParallelLMHead

这里的并行是模型输出时转换回词表的操作。

```python
def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        #这个操作就是在做点积 但是由于矩阵太大了 采用词表并行的方式 并行计算
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

```

对于参数的理解：

context 是在输入前就固定好的，  `x` 就是model处理得到的那个hidden_states （如果是prefill阶段 输出就是sequence length；如果是decoding 输出就是单个token）

其实就是如果是prefill阶段，那么x对应的hiddenstates就是每一个token的预测，那么原来request的最后一个词的位置上就是对下一个token的预测；如果是decoding阶段，x对应的hiddenstates就是一个一个request的预测token，不需要去标识每个request有多长。

这里涉及到 `context.cu_seqlens_q`  为**Query的累积序列长度**。在底层实现中，会将所有request句子首尾相连，变成一个一维数组[不需要padding]。使用 `cu_seqlens_q ` 就会得到每个request的位置。



# LayerNorm.py

在nano-vllm中实现的是：RMSNorm。这与传统的Transformer中的decoder是不同的。总体来说就是将residual和hidden_states相加后归一化。

![img](https://pica.zhimg.com/v2-5b27f1e99fe10d1c7443c7c509445034_1440w.jpg)

具体实现比较好理解，但是其中涉及到一个**精度问题**，这是我从前没注意到过的。

```python
@torch.compile
def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

```

这里涉及到将参数拉高到float 32位单精度浮点的操作，是为了避免后续在进行平方计算的时候数值越界。最后再将x恢复到 `orig_dtype`  



# Linear.py

这个文件中主要是实现TP张量并行的部分。

## LinearBase

首先定义了所有线性层的父类，定义了分布式环境下的一些基本属性。

同时注意到，在基类中使用 `torch.empty` 初始化权重，但是不立即加载数据（权重占位），真正加载权重的操作在下面的 `weight_loader` ，但这是一个自定义的，在不同子类linear中有不同实现，从而来实现TP张量并行。

```python
self.weight = nn.Parameter(torch.empty(output_size, input_size))
        #就是进行变换
self.weight.weight_loader = self.weight_loader
```

## ReplicatedLinear

子类直接继承了基类，这个linear层的设计思路是：每个GPU都拥有当前层的所有权重参数，直接用 `copy_` 拷贝到 `param.data` 中 。

从forward中也能看出，GPU中单独计算出当前layer的的数据，不涉及GPU间的通信。

```python
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight, self.bias)
```

## ColumnParallelLinear

列并行线性层，这里就是论文**Megatron-LM** 实现MLP第一层的方式。

![image-20260402235404307](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402235404307.png)

实现方式就是将权重矩阵按列切分。

```python
        super().__init__(input_size,divide(output_size,tp_size),bias,0)
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
```

可以看出在init部分传入的 `tp_dim = 0` ，这也是和后来的row并行区分，取权重的切分方式不同。

`loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)` 	使用narrow来切分权重矩阵，取出对应GPU部分的权重，但这里 0 应该是按行分，但是存储的weight是需要的转置，所以我们按行分就是逻辑上的按照列切分。最后再将weight参数拷贝到预留的param.data

## RowParallelLinear

接下来继续讲行并行线性层，`weight_loader` 部分一致，只有init时传入的tp_dim参数为1。

有些不同的是 `forward` 部分，这里就是MLP实现的第二部分，最后完成之后需要进行通信all_reduce ，每张GPU上结果形状都是[batch,output_size] ，allreduce就会让所有output结果加在一起得到结果。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

## MergedColumnParallelLinear

这个部分比较难理解一些，我看了比较久时间。要搞懂先要明白在哪里调用了它。这个函数调用在 `qwen3.py/Qwen3MLP` 中，这个操作就是 `gate_up_proj` ，把gate和up的拼成一个大矩阵，然后forward GEMM一次，这样gate和up都进行了投影，从而避免了两次GEMM。当然其中涉及到了多个GPU的协作。

调用处：传入一个list

```python
self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2, #升维到intermediate_size的两倍，一部分用于gate，一部分用于up
            bias=False,
        )
```

继承了`ColumnParallelLinear` ，只是内部的 `weight_loader` 不同，因为内部是gate和up两个融合。

```python
  def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):  
      self.output_sizes = output_sizes
      super().__init__(input_size, sum(output_sizes), bias)
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
    param_data = param.data
    shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size 
    shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
    #以上操作都是为了找准位置存放权重，从哪里放起，放多少
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    #将权重分为tpsize份，取出对应rank的
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    #拷贝到刚刚腾出的空间
    param_data.copy_(loaded_weight)
```

首先我们来看 `output_sizes` ，这是一个列表，记录被合并进来的每一个逻辑层的原始的输出维度大小。`load_shard_id` 就是要加载的类型，0是Gate，1是up。

`shard_offset `这个量就是划定要取出来的权重在**当前GPU本地显存中的起始位置**，由于使用了并行，每个GPU都只分到了一部分。`shard_size` 则是决定要加载多少。

`param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)` 划出空间用来加载权重。

## QKVParallelLinear

这个部分和merged部分逻辑是类似的。看了很久这个地方，看这个的时候需要先看一下`loader.py` 这个加载model的。

根据`shard_id`来确定要存储参数的位置，值得注意的是接下里几行代码：

```python
param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
param_data.copy_(loaded_weight)
```

`param_data.narrow`不拷贝数据，只是在 `param` 上切出一个可写的视图，后面 `copy_` 会写进这段里。这里的`shard_offset`和`shard_size`是先根据qkv算出的。剩下的与merged是一样的。



# loader.py

`default_weight_loader` 就是默认的加载方式，没有分片什么，就是直接copy_，但就要求copy到的位置要对齐。

```python
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        #先在cpu上读取张量
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                #对每个 weight_name，先尝试走mapping
                for k in packed_modules_mapping:
                    #k是mapping中的key 看是否与外部参数checkpoint的name符合
                    if k in weight_name:
                        #v 外部参数名 qkv_proj\gate_up_proj shard_id 就是 q\k\v 这些
                        v, shard_id = packed_modules_mapping[k]
                        #找到model中对应要存放取出权重param的名字 用v换k
                        param_name = weight_name.replace(k, v)
                        #用这个name确定加载的权重放在model的哪部分
                        param = model.get_parameter(param_name)
                        #再把权重写进去
                        weight_loader = getattr(param, "weight_loader")
                        #这个param就是模型里已经创建好的参数对象（目标位置）
                        #f.get_tensor(weight_name) 从 checkpoint 读出来的权重（源数据）
                        #shard_id 就是说明具体放在param哪一块
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    #如果mapping中没有 说明直接使用checkpoint中的权重原始名即可
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

```

首先是`packed_modules_mapping = getattr(model, "packed_modules_mapping", {})`这个就是相当于从model实例中取这个`model.packed_modules_mapping`。涉及到这个mapping的就是，`Qwen3.py/Qwen3ForCausalLM` 中的：

```python
packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

这个mapping的含义是什么呢？

从外部文件加载参数时，外部文件只有`q_proj`这种权重命名，这里做一个mapping到model的内部合并层，同时携带子块标志，让 `weight_loader` 知道写进合并参数的哪一段。

# Rotary_embedding.py

这个部分也就是`Qwen3.py`里使用的RoPE旋转编码。

首先使用的`apply_rotary_emb` 就是给一对[x_1, x_2]进行编码。

## RotaryEmbedding

`inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))`

这段代码我们来逐步解析：

1. `torch.arange(0, rotary_dim, 2)` 总共的rotary_dim，每两个维度一组组成一个平面来旋转。生成 `0, 2, 4, ..., rotary_dim-2`。
2. `.. / rotary_dim` 归一化到 `[0, 1)` 区间
3. `base ** (...)` 得到一个按维度递增的指数序列（这个base是`qwen3.py`中设定给出的 常见 `base=10000`  ）
4. `1.0 / (...)`取倒数，得到 `inv_freq`（inverse frequency，逆频率）

这样设计的话，不同的维度就会有不同的角速度: 低维（前面的 pair）频率高，变化快；高维（后面的 pair）频率低，变化慢。-> **在同一个维度中通过m-n这种可以得到相对位置信息，在不同维度对间角度theta不同，也可以表示不同的信息。**正因此，不同维度间的theta关系其实是可以任意取得，两个维度的组成方式可以任意。

```python
#生成位置index 第几个token的位置
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
```

`freqs = torch.einsum("i,j -> ij", t, inv_freq)` 这个operation之前没有接触过，但其实就是：让刚生成t中的每个元素都与`inv_freq` 相乘，得到的就是

[ [0\*theta1,0\*theta2,...] , [1\*theta1,1\*theta2,...],..., [m*theta1,m\*theta2,...] ]

因为 RoPE 本质是“把 2 维组成一对做旋转”。“哪两维组成一对”可以有不同固定规则。

# Activation.py

既然上面把MLP的TP并行操作都看了，接下来就看MLP中激活函数的实现。

在`qwen3.py/Qwen3MLP`中传入的gate_up，是一个大小为 `[intermediate_size] * 2` 的矩阵，一半是给gate，一半是用于up，所以二者只有**gate需要激活**。

```python
self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2, #升维到intermediate_size的两倍，一部分用于gate，一部分用于up
            bias=False,
        )
def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

所以在使用激活函数时，需要首先将 `x` 分开， 沿着隐藏层维度（-1）将输入的张量均分为两块，前部分是 `x` 后部分是`y` 。再使用 `F.silu(x) * y` 逐元素相乘，门控激活（silu后相乘决定y部分中有多少信息流向下一层）。

```python
@torch.compile
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y

```

当然我们在这里也可以看到 `@torch.compile` 这是算子融合（Kernel Fusion），它会生成一个专门的 CUDA 内核，让 GPU 在寄存器里一次性完成“切分、激活、相乘”所有操作。如果不加这个，PyTorch 会先算一次 SiLU，把结果存回显存；再读出来算乘法。这涉及多次显存读写（I/O），在 GPU 上其实非常慢。

# Attention.py

这个文件也是model的核心，涉及到kv cache以及flash attention。

核心：将当前 KV 写入KV Cache，根据不同阶段选择不同Flash attention API。

这个文件是在 `Qwen3.py/Qwen3Attention` 的 `rotary_emb` 后调用，存入的是**新计算出**RoPE编码后的QK。

从`Forward`部分来理解：

1. `context = get_context()` 读取当前step的batch的上下文信息，是否是prefill
2. ` k_cache, v_cache = self.k_cache, self.v_cache` 首先把当前这一层在model上的GPU张量读到局部变量上，方便后面传。
3. `store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` 这里是存放计算出来的KV Cache。
   1. 写入源：`Qwen3Attention`中新计算出的KV，形状为：`[N, num_kv_heads, head_dim]`，`N` 为本步参与计算的token个数（所有序列拼成一条大的batch）
   2. 写入目标：`context.slot_mapping` 长度为N。每个 token 在 K/V 上占 `D = num_kv_heads * head_dim` 个连续元素；
4. 做FlashAttention：
   1. Prefill阶段：`engine`把多条request拼接成一条一维的长向量（而不是带padding的二维向量），然后用`前缀和 cu_seqlens_q`标出每条序列的边界。`cu_seqlens_q[-1]`、`cu_seqlens_k[-1]` 就是qkv的总token数。
   2. Decoding阶段：

有一些细节需要详解，Triton中是按照block操作的。

## store_kvcache

目的：将当前这一步forward算出的K/V，按照slot_mapping写入全局KV Cache的函数。

内容：校验内存分布 + 启动Triton Kernel



```python
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    #key: (batch_size, num_heads, head_dim)
    #这里的batchsize  prefill：当前 batch 中所有序列本步参与计算的新 token 总数
    # decode：通常每条序列 1 个 token，所以 N ≈ 当前活跃序列数
    N, num_heads, head_dim = key.shape
    #D:  每个token的维度 K/V的维度
    D = num_heads * head_dim
    #k.head_dim连续，v.head_dim连续 [0..D-1]连续
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    #k.num_heads步长为head_dim，v.num_heads步长为head_dim
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    #k_cache.num_heads步长为D，v_cache.num_heads步长为D
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    #每个待写 token 都必须有对应槽位。
    assert slot_mapping.numel() == N
    #调用kernel函数存储kv cache
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

```

#### 内核启动部分：

启动1D grid（一维空间），共N个program，每个program处理一个token(idx)。

- 传入 `key.stride(0)`、`value.stride(0)` 给 kernel，用来算第 `idx` 行基址。[指的是第idx个token]
- 传入 `D`（`constexpr`），让 kernel 知道每行要拷贝多少元素。

kernel内部操作：根据传入的地址把 `key[idx]`、`value[idx]` 整行拷贝到 `k_cache[slot]`、`v_cache[slot]`。

## store_kvcache_kernel

用 Python 语法写 Triton kernel。

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    #idx: 当前token的索引 一个batch中所有token索引
    idx = tl.program_id(0)
    #slot: 当前token的槽位索引 一个batch中所有token槽位索引
    slot = tl.load(slot_mapping_ptr + idx)
    #如果当前token的槽位索引为-1 则跳过
    if slot == -1: return
    #key_offsets: 当前token的key的偏移量  tl.arange是token内的偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    #value_offsets: 当前token的value的偏移量
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    #cache_offsets: 目标写入位置的偏移量
    cache_offsets = slot * D + tl.arange(0, D)
    #将当前token的key存储到kv cache中
    tl.store(k_cache_ptr + cache_offsets, key)
    #将当前token的value存储到kv cache中
    tl.store(v_cache_ptr + cache_offsets, value)


```

## 怎样调用 flash_attn 呢？

这个需要我们结合FlashAttention的代码一起来分析。

我的理解是flashattention就是将注意力分块进行运算，为什么调api可以实现呢？调用api是怎么实现这个分块计算的呢？

这和一般的TP不同，实现是在CUDA层面的：这里的`分块`是指每次从GPU的HBM中加载一小块tile进入SRAM，对这一块局部进行matmul + softmax，在softmax过程中，每来一个新tile，得到这一段的logits，然后去与全局当前的(`m`,`l`)合并，这样不需要在HBM中存每个token的整个logits。[具体原理参考flashattention论文]

=> 即：在HBM中不需要再去存完整的attention矩阵(N x N)，只需要维护每行的m、l这些小统计量。

### Prefill 阶段

```python
if context.block_tables is not None:    # prefix cache
  k, v = k_cache, v_cache
  #block table指明当前batch的kv cache的存储物理块
  o = flash_attn_varlen_func(q, k, v,                                                 max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
```

调用：[执行核心的**注意力计算**的地方]

1. q：当前步query（拼接后的总token维）
2. k, v：普通prefill 用当前forward算出的kv（没有reuse）；使用prefix cache（即block_tables != None）使用`k_cache,v_cache` (里面有之前session的kv)
3. `cu_seqlens_q` 、`cu_seqlens_k`给出每条序列的真实边界 [后取出seq进行计算]
4. `max_seqlen_q` 、`max_seqlen_k`给内核一个上界，方便决定 tile/block、shared memory 使用和循环范围；
5. `causal=True` 表示使用因果掩码（自回归掩码）
6. `block_table` 在使用prefix_cache中提供要reuse的KV Cache的block位置，不使用prefix就是None（因为没有复用块嘛）。必须给 `block_table`，否则内核不知道怎样按逻辑序列顺序去 gather。

### Decode 阶段

与prefill阶段的不同是：不再传 `cu_seqlens_q/k`，而是传 `cache_seqlens` 来告诉每条序列有效历史长度。

这是 带 KV cache 的 decode API，Q 只是一小步（通常 1 token），K/V 从 cache 读全历史。

- `k_cache/v_cache` 是分页存储的大池子，包含很多物理 blocks；
- `block_table` 指定某条序列对应哪些物理 block（以及顺序）；
- `cache_seqlens` 指定这条序列当前有效 token 长度 L，也就是只用前 L 个逻辑位置的 KV（超过的不参与）。

 调用：

1. `q.unsqueeze(1)`: 把 `q` 从 `[bs, nheads, headdim]` 变成 `[bs, 1, nheads, headdim]`（每序列1个新 token）
2. `k_cache`, `v_cache`: 整个历史 KV 缓存（已先通过 `store_kvcache` 写入当前 token 的 K/V）
3. `cache_seqlens=context.context_lens`:「包含历史 + 当前这一步在内的、该 request 的 总序列长度」
4. `block_table=context.block_tables`: 每条序列对应的物理块映射。既然是decode，那么一定已经有KV cache了，要知道block块知道从哪里取。

### 问题

1. 为什么有k/v还要使用kv_cache？decode必须使用 因为forward一次得到的kv 只有当前batch的kv 而decode需要用到历史kv。prefill如果不用prefix可以直接使用kv，使用prefix必须用，因为需要读取历史kvcache。













# Sampler.py

这个文件涉及的就是，预测token的最后一步，在得到的logits里面进行采样，得到的token就是最后输出结果。

`logits = logits.float().div_(temperatures.unsqueeze(dim=1))` 

这个操作有几步，第一步先转为float，然后再使用`温度缩放`来调节概率分布的尖锐程度。先除以T再进行softmax，比如：降温T=0.5，logits被放大分布更加尖锐，头部token更占优；升温T=2，分布更平坦，尾部 token 概率抬升（更随机）。

至于 `unsqueeze(dim=1` 这个操作，就是把`temperature`变成`[B] -> [B, 1]`，方便div。不然就是 `[1,B]`这样无法对齐。

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

最后这个部分就是在进行采样，不过使用的是随机采样，而不是贪心采样。实现就是，加入随机参数E，用`prob/E`，最后结果中的最大值，这与随机取样数学形式上是一致的。结果有随机性，但高概率词总体更常被选中。

# llm_engine.py

接下来进入scheduling的部分！

调用链：

`generate()` -> `add_request()` ->  `scheduler.add()` -> 循环 `step()` -> `scheduler.schedule()` -> `model_runner.run()` -> `scheduler.postprocess()` -> 输出完成序列

## Init

```python
#从对应model加载tokenizer
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
#从tokenizer中取出该模型的eos token id
config.eos = self.tokenizer.eos_token_id
```

注意这里要理清：1. Config.model 是一个目录路径，告诉程序去哪里找model 2. 这里从model目录加载完tokenizer后，才能用tokenizer去找eos，把它写到config中去。

## Generate()

使用进度条表示：

```python
if use_tqdm:
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

```

Sampling_params 采样控制参数 含：temperature / max_tokens / ignore_eos

```python
from dataclasses import dataclass
@dataclass
class SamplingParams:
    temperature: float = 1.0 #温度缩放
    max_tokens: int = 64 #最多生成多少个token
    ignore_eos: bool = False #是否忽略EOS 终止符

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"

```

在generate中支持两种调用方式，

1. 所有Prompt共用一套samplingparams
2. 每个prompt有自己的参数：给一个list[samplingparams]

然后就批量将prompt和samplingparams配对，然后调用`add_request`，加入队列。

只要队列中还有未完成的seq，就会继续执行。在while循环中，调用step执行一次，最后将得到的结果output的token_ids转换成text文本，按照seq_id排序存入outputs。

关于进度条实时吞吐

```python
if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
```



## add_request

在这个函数中，如果prompt是字符串会先tokenize（每个 token 映射成一个整数 id），然后构造一个Sequence，再将这个seq加入`scheduler.waiting` 队列。

核心思想：把所有待生成请求先入队，后续step再调度执行。

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

为什么先要构造成一个`Sequence`呢？方便进行调度。这里面封装了很多调度信息，这里只传入了一部分。

## Step

在step中，也是LLMengine主体操作。

1. 首先，由scheduler来调度组合batch，得到当前轮次要跑的序列列表`seqs`，以及决定当前轮次是`prefill` or `decode`
2. 执行模型推理`model_run`，得到每条seq生成的新token的ids
3. 调用scheduler中的postprocess，将token 添加到各序列，并判断是否完成，必要时释放KV block。‘
4. 收集本轮数据 (seq_id, completion_token_ids) 新生成的token_id
5. 生成统计信号量`num_tokens` prefill 返回正数token总量 decode 返回负的batch_size（这里就是generate部分要的）



# Config.py 

来统一保存engine启动和推理运行时需要的参数。

- 模型定位
  - `model`：模型目录路径
- 批处理与长度限制
  - `max_num_batched_tokens`
  - `max_num_seqs`
  - `max_model_len`
- 并行与执行模式
  - `tensor_parallel_size`
  - `enforce_eager`
- 显存/KV cache 策略
  - `gpu_memory_utilization`
  - `kvcache_block_size`
  - `num_kvcache_blocks`（运行时再计算）
- 模型与停止相关运行时信息
  - `hf_config`（由 `AutoConfig` 加载）
  - `eos`（默认占位，后续由 tokenizer 回填）

并且在post_init做合法检查

```python
def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

```

enforce_eager   hf_config

# Sequence.py

用来封装调度所需信息。

- 身份与状态
  - `seq_id`：全局递增 ID（`counter` 生成）
  - `status`：`WAITING / RUNNING / FINISHED`
- token 数据
  - `token_ids`：完整 token 序列（prompt + 已生成）
  - `last_token`：最后一个 token（decode 阶段常用）
  - `num_tokens`：当前总 token 数
  - `num_prompt_tokens`：prompt token 数
- KV cache / block 管理
  - `num_cached_tokens`：已写入 cache 的 token 数
  - `block_table`：逻辑 block 到物理 cache block 的映射
  - `block_size`（类变量，默认 256）
- 采样与终止配置（来自 `SamplingParams`）
  - `temperature`
  - `max_tokens`
  - `ignore_eos`
- 一些便捷属性/方法
  - `is_finished`
  - `num_completion_tokens`
  - `prompt_token_ids` / `completion_token_ids`
  - `num_blocks` / `last_block_num_tokens` / `block(i)`
  - `append_token(token_id)`



# model_runner.py

运行顺序：

`run`

## Init

里面有个捕获CUDA图。

## Run

这是模型运行的主体，

1.  根据prefill or decode 阶段构造 inputs 与 positions。[prepare_prefill & prepare_decode]
2. run_model 跑model forward得到logits输出。
3. 根据`run_model`得到的logits，进行温度缩放后采样得到`token_ids`。



## prepare_prefill  & prepare_decode 

这两个函数都是为了构造模型输入的，输入都是Sequences(`seqs: list[Sequence]` 就是一个batch有多个requests组成)

这里涉及到使用Prefix cache时，Q、K、V该如何决定。

`seqlen_q = seqlen - seq.num_cached_tokens`  不需要已经缓存的Q，只需要新增token的Q来与`旧K+新K=seqlen_k = seqlen`计算。

P.S 这里的`num_cached_tokens`是在`BlockManager` 里维护的。

![image-20260408231904566](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260408231904566.png)



使用prefix cache时，需要为sequence中prefill新增的token的kv cache分配slot，计算在物理内存的写入位置。

**这里写入的设计也是基于paged attetion的。这里`seq.num_blocks`指的是当前序列`request`所占据的总blocks数，这个量是计算出来的。**

1. `start = seq.block_table[i] * self.block_size` 计算当前写入的起始位置。这里不从别的块开始写，num_cached_tokens单位都是blocks，同时prefix cache也是block级别的命中。

2. 对于end，分两种情况：

   1. 不是最后一个block： 写入范围就是`block_size`

   2. 最后一个block：只写到`last_block_num_tokens `

      P.S 内部存储一定是连续的，而且中间块一定是填满的，但是尾部块不一定写完，所以有`last_block_num_tokens`，方便下一次的写入找到index。

3. `slot_mapping.extend(range(start, end))`把当前的block对应的物理槽位全部加入`slot_mapping`

那么就会有个问题：

1. 内部`slot_mapping`到底是怎样一个映射方式？
2. block_table 到底是怎么管理prefix cache命中的呢？Prefill阶段的KV cache都是从头开始存吗？

接着，构造block_table，明确seqs中所有序列使用的prefix cache的存放的物理block，方便后续flash attention调用。 

但是，这里的判断是用`cu_seqlens_k[-1] 和 cu_seqlens_q[-1]`，这两个量是在上面的遍历构造得到的。由于每次 `cu_seqlens_q.append` 的是 `seqlen_q = seqlen - seq.num_cached_tokens `，所以就会得出判断出是否使用了prefix。

`set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)` 把本轮prefill需要的元数据写进全局Context，model forward过程会通过`get_context()` 读到。比如在attention中，block_table就直接从context中读出。		

## prepare_block_tables

```python
    def prepare_block_tables(self, seqs: list[Sequence]):
        #确定max_len 即每个request的block_table长度最大值
        max_len = max(len(seq.block_table) for seq in seqs)
        #为每个request的block_table填充-1 直到长度达到max_len
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables 
```



## prepare_decode

整体逻辑与prefill是类似的，但是由于inputs的每个request每次只输入一个token，所以会简化一些。

## prepare_sample

这几个prepare都是把相关数据转化为tensor，移动到CUDA显存中。

## run_model

负责跑一次forward，输出得到的logits。

不过这里有个可以使用` CUDA Graph 加速`的选项。

什么是`CUDA Graph`



# Scheduler.py

调度中心。

在**WAITING中**的是准备进行prefill的（这里回看llmengine，加入等待队列的都是prompt即prefill阶段），decode阶段处理的都是已经开始生成、处于**RUNNING阶段**的。



# Block_manager.py

这个是比较重点的内容。

一步一步慢慢来看。

## Init

对于：`self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]` num_blocks是config中指定的总共有多少个blocks

1. `self.blocks: list[Block] ` 类型标注，self.blocks这个属性是 `Block`对象组成的列表。
2. `[Block(i) for i in range(num_blocks)]`对每个i都创建一个Block(i)，最终得到`[Block(0), Block(1), ..., Block(num_blocks-1)]`

```python
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        #初始化num_blocks个Block对象    这个num_blocks是配置文件中num_kvcache_blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        #创建一个hash到block_id的映射表 
        self.hash_to_block_id: dict[int, int] = dict()
        #创建一个空闲block_id队列
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        #创建一个已使用block_id集合
        self.used_block_ids: set[int] = set()
```

## compute_hash

作用：为了计算每个token的hash值。之前一直认为，算hash是给每个block的id来算hash，但是实现居然是每个token_ids来算hash，最后这些token的hash合在一起就是block对应的hash。

```python
if prefix != -1:
    #将prefix转换为8字节的bytes并更新hash值
    h.update(prefix.to_bytes(8, "little"))
```

这里如果有prefix的话，就是上一个块的hash就是prefix。会把这个prefix转换成8字节的小端序二进制，先把这个prefix给hash计算，再去算当前token的hash。

这和论文中提到的计算方式是一致的，当前token和block的hash一定要与上文有关，才能正确reuse。

## _allocate_block & _deallocate_block 

 作用：这两个函数就是对物理的block分配和释放。



## Can_allocate

```python
#判断能否分配block 当前队列中空闲block数量 >= 当前seq需要的block数量
def can_allocate(self, seq: Sequence) -> bool:
  return len(self.free_block_ids) >= seq.num_blocks
```

这里的`num_blocks`就是上面提到的算出来的当前seq需要多少block。

## allocate

作用：处理逻辑block与物理block的映射。

取出该序列第 `i` 个逻辑 block 对应的 token 切片。

```python
def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

然后将第i个逻辑block中的token进行hash计算，然后去字典 `hash_to_block_id` 里查这个哈希是否已经出现过，如果出现就返回对应的block_id（可能可以直接复用已有物理block），没有就返回-1，需要重新分配物理block。

这里分两种“复用”：

1. 命中且当前正在被使用（in used）
   直接共享同一物理块：`ref_count += 1`
2. 命中但当前不在 used（不在 used）
   说明这个 block 虽然在哈希表里，但现在处于空闲池（之前没人引用了）。
   这时需要把它重新激活为已使用状态，所以调用 `_allocate_block(block_id)`。这不是“新建内容”，而是把已有块从 free 挪到 used，并重置引用态。

重点要搞清的事：逻辑block和物理block是怎么映射的。

## deallocate

在函数最后，会清空当前seq的缓存kv cache数量以及block_table，为什么要这么操作呢？

```python
#清空seq的num_cached_tokens和block_table
        seq.num_cached_tokens = 0
        seq.block_table.clear()
```

## can_append

这个函数与`can_allocate`的区别在于，这个函数用于decode阶段，`len(seq) % self.block_size == 1`至少要求有一个空闲块。整体翻译过来就是：这一步是否需要新块；需要就检查至少 1 个空闲块，不需要就直接可过。

## May_append

作用：这个函数在`scheduler.py`中的decode阶段被调用，真正来更新block_table。

# Context.py

作用：设置每次forward时的为底层计算提供的上下文，每一个新的forward都有reset一次。

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()
```

## `Context` 里各字段做什么

- `is_prefill`：本步是 prefill 还是 decode（attention 路径可能不同）。
- `cu_seqlens_q` / `cu_seqlens_k`：变长序列的累积长度（前缀和），给 flash-attn 风格接口区分每条序列在拼起来的大序列里的边界；prefill 常用。
- `max_seqlen_q` / `max_seqlen_k`：本批 Q/K 的最大长度，很多 kernel 需要。
- `slot_mapping`：每个 token 的 K/V 要写入大块 KV cache里的线性槽位；prefill 可能是一段，decode 常是每序列一个。
- `context_lens`：每条序列当前上下文长度；decode 常用。
- `block_tables`：paged KV 的块表（逻辑块 → 物理块），prefill 有 prefix 时或 decode 时都会用。





跑bench

![image-20260401142623036](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260401142623036.png)

# 参考

1. nano-vllm overview：https://zhuanlan.zhihu.com/p/2008285806222132143
2. nano-vllm 代码地址：
