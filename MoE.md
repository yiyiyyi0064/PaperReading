# MoE

进一步scaling需要资源太多，太贵。

Related 

补一下MoE

# Model Architecture

![image-20260329100037171](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260329100037171.png)

Gating module[会在train中学习到激活] 会动态地为input token选择两个最相关的expert（64个中 右侧的grid）然后这两个专家输出的**加权平均**会传给下个Transformer。对于下个新的token，会选择两个不同的专家。[怎么选择experr呢？会先打分后softmax得到概率分布] -> 虽然MoE layer有很多参数，但是只有被选中的expert参数激活参与运算。

## 文章对Architecture做的modifications

1. 用per-layer relative positional bias 取代 standard positional embedding
2. 在非MoE的FFN层，将第一个linear projection层和激活函数替换为Gated Linear Unit和一个GELU 激活函数。
3. 使用2D sharing算法来划分参数









