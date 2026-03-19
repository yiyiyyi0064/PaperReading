# Attention is all you need

# Model Architecture

![image.png](Attention%20is%20all%20you%20need/image.png)

## Input Embedding

只有**#0** Encoder的input需要embedding（one-hot）

### Positional Encoding

虽然我们有self-attention机制，但它**关注的是token与token之间的相关性**，没有**考虑到input的sequential information**。

Transformer给出的方案是add a vector(contains positional information) to each input embedding

![image.png](Attention%20is%20all%20you%20need/image%201.png)

**那么这些positional vector是怎么得到的呢？**

在Transformer中提供的方法，使用的是

![image.png](Attention%20is%20all%20you%20need/image%202.png)

直觉的做法是使用绝对位置编码，但是论文认为使用这种正余弦的编码方式能够有更好的拓展性，输出不会局限于绝对位置编码的index内。

## Attention

### Scaled Dot-Product Attention（缩放点积注意力）

**注意到：**

![image.png](Attention%20is%20all%20you%20need/image%203.png)

Q、K、V是由embedding后的vector乘三个不同的矩阵得到的。**而这三个矩阵是可训练的。**

需要注意的是：由于使用的是Multi-head，在论文中使用H=8个的head，主dimension=512，通过线性投影得到 $d_k=64$ 。**保证总计算量不变。**

具体计算过程就是self-attention，用一个图来详细illustrate

![image.png](Attention%20is%20all%20you%20need/image%204.png)

计算过程有几个细节：1. Divide by $\sqrt{d_k}$ 这里去做scaling 防止scores过大进入softmax后梯度消失 2. 这里是在softmax之后再与 $W_v$ 来进行加权求和

**Matrix Form**

![image.png](Attention%20is%20all%20you%20need/image%205.png)

### Multi-Head

有多个head，每个head有单独的QKV，得到自己的Z。

但是**FFN**的input是一个matrix，所以需要将得到的多个Z合并得到一个matrix。实现方式是：先concat然后Multiply $W_o$ 得到的就是来自所有attention head的information。

![image.png](Attention%20is%20all%20you%20need/image%206.png)

## The Residuals

残差之后接一个LayerNorm，有一篇文章分析了使用LayerNorm的效果。

Link：https://arxiv.org/abs/2003.07845

![image.png](Attention%20is%20all%20you%20need/image%207.png)

## Decoder

### **那么Encoder与Decoder又是如何combine的呢？**

encoder的output为vectors K and V 这在decoder的“encoder-decoder attention” layer中被使用，这能帮助decoder去关注输出中合适的位置。而Q则还是来自Maked Multi-head的。

对于decoder，它是auto-regressive，得到的output会作为input来使用。

### Masked Multi-Head Attention

注意层只允许去看在改词之前的词，之后的词会被mask。

## **The Final Linear and Softmax Layer**

![image.png](Attention%20is%20all%20you%20need/image%208.png)

如图：

decoder stack的output应该是dimension=512的，通过linear映射到vocab_size。这个logits就是每个cell就是对应word的score，经过softmax得到probability，最后选择最高的probability对应的word。

# Train

下面有一些实现细节

### 翻译如何结束？

一种可能的做法，output中有END这种symbol。（这应该在一开始的vocab中包括）

### Beam Search 束搜索

有更多确定性

# 思考

1. 怎么理解QKV的含义?（这个问题related to为什么Decoder要使用来自encoder的K、V）
2. 使用不同head有什么作用？
3. Transformer中的weight Tying
4. positional encoding和self-attention有什么区别？
5. 在考虑positional encoding时，为什么加一个vector就能提供位置信息？
6. 为什么使用layernorm？不使用batchnorm？

# 参考

1. https://mp.weixin.qq.com/s?__biz=MzI4ODg3NDY2NQ==&mid=2247483679&idx=1&sn=26ec25fcb954332f2c21ca89c324278e&chksm=ec3688d9db4101cf99ed29500e1b3f50225159a7b1e3d318d4409caad1955fe354f7f502363f&scene=178&cur_album_id=1464771644039610372&search_click_id=#rd
2. http://jalammar.github.io/illustrated-transformer/
3. https://arxiv.org/abs/1706.03762
4. 代码实现