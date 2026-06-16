这里探讨几种并行方式：PP、TP、EP

由于之前看过一些讲解博客，论文看了重点部分。

# PP并行

![image-20260402232327881](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402232327881.png)

核心思想：就是将model的layer划分为多个部分。1. Model Partitioning：将一个包含L层的网络划分为K个连续的cell，并将每个单元放置在独立的GPU中（论文里写的是accelerator应该就是说的GPU）2. Batch Splitting：将一个Mini-batch进一步拆分为更小的batch，**实现流水线**。

一些其他优化操作：1. Re-materialization：为了节省显存，在forward中，每个GPU都只存分区边界的输出激活值，不存中间结果； backward中，当需要某一层的数据来算梯度时，发现显存里是空的。此时，加速器会利用之前保留的分区输入端数据，**重新跑一遍前向计算** 。从而显著降低了峰值激活内存需求 。 2. Gpipe使用的是同步微批次梯度下降，梯度在整个微批次的所有微批次中累积，在micro-batch结束的时候再统一update。

# TP并行

重点看了Model Parallel Transformers部分

## MLP Block 

![image-20260402235404307](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260402235404307.png)

这里就是涉及到基础的矩阵乘法，第一个权重矩阵按照列分，相当于一个GEMM分成两次算。第二个按照行分，直接利用第一次计算的结果，输出结果。[为什么这里可以直接这样呢？]

![image-20260403000405844](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260403000405844.png)

一般需要两次allreduce才能解决的，这里只需要最后Dropout前来通信即可。

## Self-Attention  Block

![image-20260403134152410](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260403134152410.png)

直接把QKV分成多个头，每个头的计算只需要对应的Q、K、V分量，所以在计算过程中是不需要通信的。最后head计算完后，进入Dropout这样一个linear layer来使用行并行，也是不需要GPU间的通信的。

![image-20260403140625997](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260403140625997.png)

综上可以看出，在Transformer layer中，在forward中一个MLP和一个Self-Attention只需要两个All-reduce通信。

## Logits

Megatron-LM在这里做出训练的优化，做一个融合计算（GEMM+Cross Entropy），每个GPU在本地计算自己负责的那部分词表的交叉熵，最后all-gather汇总的时候，就只需要传输一个标量，规模为`b*s`而不是传统方案的 `b*s*v` 。

但是推理没法减少，因为最后采样必须要那个logits的概率分布。







# 参考

1. 关于几种并行：https://zhuanlan.zhihu.com/p/2003423046342554380