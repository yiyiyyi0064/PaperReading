# Overview

量化技术常被用于降低显存占用、减少计算量与数据传输开销。这其实也是一种Tradeoff，就是显存占用计算量与精度的取舍。

在模型计算过程中，数据类型可以统一为一种，但更常见的是混合使用多种类型。例如，logits采用FP32，attention_score和router采用FP16，线性层运算则使用FP8。这种混合策略正是精度与速度之间权衡取舍的结果。

在推理阶段，量化计算通常不会应用于所有网络层，而是主要集中在非线性层，如Attention中的线性投影（QKV project、O project）以及MLP层的线性运算（up、down、gate）等。

更进一步，在同一运算中，两个输入也可采用不同精度，即权重（Weight, W）和激活值（Activation, A）的数据类型可以不同。推理时，通过组合二者的数据类型，可产生多种量化形态，例如：

- W8A8：权重和激活值均采用FP8类型；
- INT W8A8：权重和激活值均采用INT8类型；
- W4A8：权重采用INT4，激活值采用FP8；
- INT4 W4A16：权重采用INT4，激活值采用FP16。

![image-20260417152337249](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260417152337249.png)

优点：1. 保持精度，相当于引入噪声；2. 提升速度

## 问题

1. 为什么模型量化技术能够对实际部署起到加速作用？
2. 为什么不训练的时候就训练低精度模型？
3. 什么情况下不应该/应该使用模型量化？

### 量化方法

![image-20260417152753821](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260417152753821.png)

### 原理

模型量化桥接了定点与浮点，建立了一种有效的数据映射关系，使得以较小的精度损失代价获得较好收益。

PTQ Dynamic 动态离线量化

仅将模型中特定算子的权重从FP32类型映射成INT8/16类型

PTQ Static 静态离线量化

使用少量无标签校准数据，核心是**计算量化比例因子**。使用静态量化后的模型进行预测，在此过程中量化模型的缩放因子会根据输入数据的分布进行调整。

![image-20260417155630663](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260417155630663.png)

#### KL散度校准法

### 端侧量化推理部署

端侧量化推理的结构方式主要有3种，

1. FP32输入FP32输出 
2. FP32输入int8输出
3. int8输入int32输出

![image-20260417161248227](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260417161248227.png)

量化

![image-20260417161432902](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260417161432902.png)



# AWQ

AWQ量化是一种基于激活值分布(activation distribution)挑选显著权重(salient weight)进行量化的方法。











# 参考

1. 基础速览：https://zhuanlan.zhihu.com/p/2005335401469083798
1. AWQ: https://zhuanlan.zhihu.com/p/697761176
