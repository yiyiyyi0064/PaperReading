由于Flash attention有3个版本，我们放在一起来看，对比并分析一下各个版本之间的创新与tradeoff。

# FlashAttetion-1



Memory Hierarchy 

![image-20260406211209049](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260406211209049.png)

之前的工作，重点是在减少计算量or内存消耗，比如稀疏注意力（重点是在attention的部分）；但是文章优化的方向是从IO的方向上去优化。

主要目标就是避免从HBM中读或写attention矩阵。

使用Tilling，将输入分割成多个块，并且在这些输入块上进行多次遍历。同时，在CUDA上进行实现，实现对memory的fine-grained control，并且将所有的attention操作到一个GPU核上。在标准的Pytorch上，每执行一步就会启动一个独立的GPU Kernel，每个Kernel都会把算出的中间结果写回到慢速的HBM上，下一个Kernel再从HBM读出来，这种频繁读写会导致速度大幅变慢。

## 原始attention

![image-20260406233531244](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260406233531244.png)

![image-20260406211023319](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260406211023319.png)

## FlashAttention

我们重点探讨**foward**的部分。

在attention这个部分，我们的最终目标就是将attention output写入HBM，所以我们可以减少HBM accesses。

### Tiling

![image-20260406234429083](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260406234429083.png)

这是论文中给出的算法逻辑。在这个过程中存在的IO变成了：















# 参考

1. 讲解：https://www.bilibili.com/video/BV1UT421k7rA?spm_id_from=333.788.recommend_more_video.0&trackid=web_related_0.router-related-2479604-9shrk.1775480661790.588&vd_source=0025ea9b72b20ffd57a9e9a9bec2f297