# Accelerating Large Language Model Decoding with Speculative Sampling

# Intro

**该方法的大致思路：**

draft model生成k个token之后 再将生成的token+原prompt交给target model来forward一遍，再与draft model得到的概率来进行比较，来选择要不要留下（因为target model forward一遍就可以得到所有预测token的概率，故这个部分理论上是可以并行的），如果不留下，就舍弃当前token以及之后生成的所有，重新选择一个。

接下来继续看看它的实现细节。

# Auto-regressive Sampling

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image.png)

模型从已经生成的x sequence 计算下一个x出现概率，并从中采样sampling得到 $x_{n+1}$ 并加入队列 继续预测下一个

# Speculative Sampling

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%201.png)

# Modified Rejection Sampling

其实这是我对于这个方法的核心疑问点。

### Target Model 如何根据 Draft Model 推理得到的k个token 进行打分？

target model打分是理解的关键，这个也是我初读时比较大的困惑点。

在一般的自回归的过程中前面已经生成的qkv计算是不重要的（我们只需要新的token给的qkv），所以会有kvcache这个技术，但是在投机采样里，利用mask却可以用来打分下一个生成的结果怎么样（这是个并行操作）。

**补：**以上是初读时的思考，其实draft model就是标准的kvcache，但是target model也是使用kvcache的，因为1. 在当前任务前肯定有其他任务的，不可能只有当前的token 2. 有回滚操作，当前会直接将K个由draft model预测的token交给target model，如果猜错了要回滚到错误token输入前那部分。

**再补**：draft model 应该也是要回滚的，否则接下来的预测问题会越来越大。【这里貌似涉及到pagedattention这样方案处理rollback】

**具体实现其实很简单：**

通过有masked的attention计算后，会得到“下一个token”（而且这个过程是并行的，因为target model得到的输入就是prompt+生成的k个token 共n+k个token，那么在进行计算的时候得到的矩阵就会是，有prompt时对下一个token的预测，prompt+draft model生成的第一个token时target model对下一个token的预测，以此类推至有第k个token）[input本就是不定长的 这里迷糊了 花了点时间]

最后再在target model得到的logits的词表中，找到draft model预测生成的token，取出其概率 $q(x_i)$ ，再与draft model在生成这个token时得到的概率 $p(x_i)$ 进行比对。

### 如何比对来决定是否要留下来or舍弃呢？

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%202.png)

为1则留下，不为1则按照概率判断是否留下，如果q很小，那么就会舍弃draft model当前以及后面所有生成的token。

### 那么问题又来了，接下来的这个token该怎么选择呢？

是要**直接选择**吗：在当前已经计算出来的logits中选出词表中概率最大的token [我的第一直觉hhh]

而**论文中**给出的方式是：（这样的话那么最后选择的x和target model预测得到的x概率是一致的）

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%203.png)

换句话说就是：1. 首先要在target model中得出的对下一个token的预测概率中减去draft model的预测概率 2. 然后再对新得到的概率分布进行norm，后随机抽取。

此时得到的概率分布：{接受小模型的词的概率} + {拒绝后从残差分布抽样的概率} = 大模型原始的概率q

<aside>
💡

证明part：

如果要 X = x（x为最终选择的预测），要么 $\tilde{x} = x$ （ $\tilde{x} ~ p$） 然后接受draft model的预测，要么在拒绝 $\tilde{x}$  后重采样。

利用全概率公式，分为两个part：

![接受draft model的预测](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%204.png)

接受draft model的预测

resample rule 重新采样的分布（这是设定的）

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%205.png)

拒绝 $\tilde{x}$  

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%206.png)

最后将合并计算得：

![image.png](Accelerating%20Large%20Language%20Model%20Decoding%20with%20Sp/image%207.png)

</aside>

# 思考

1. 考虑到在两个model中kv cache的回滚操作以及预测长度，大模型和小模型要是一直预测下去的话，上限在哪里呢？
2. 那么同样生成50个token，k=50和k=5*10次，哪个的正确率会更好呢？

# 参考

1. 贪心解码[还未读]：https://arxiv.org/abs/2211.17192 
2. 投机推理：https://arxiv.org/abs/2302.01318