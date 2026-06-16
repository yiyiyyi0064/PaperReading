Tutorial:

1. 一文读懂cuda stream与cuda event: https://zhuanlan.zhihu.com/p/699754357
2. 一文读懂cudagraph: https://zhuanlan.zhihu.com/p/700224642
3. GPU原理: https://www.bilibili.com/video/BV1Ms4y1N7RL?spm_id_from=333.788.videopod.sections&vd_source=0025ea9b72b20ffd57a9e9a9bec2f297



## 一个CPU thread与2个GPU执行单元

![img](https://pica.zhimg.com/v2-4d8699d73bf9789a2c7d2e95c2edf6d2_1440w.jpg)

这里要注意，kernel 1和kernel 2能不能并发执行，需要由用户来决定，硬件不能擅作主张，否则万一kernel 2要读取kernel 1算出来的数据，那并发执行的结果就是错的。

为了给用户提供这种控制权，于是我们就有了`stream`的概念。一个`stream`就对应于一个`执行队列（加一个执行单元）`，用户可以自行决定是否把两个kernel分开放在两个队列里。

由此，我们可以总结得到：**一个GPU kernel可以执行的必要条件是它在stream的队列首位且存在执行kernel的空闲硬件资源。**

我们不仅可以从CPU thread对stream下发kernel（最常见的stream操作），还可以：

- 查询stream里的队列是否为空，即`cudaStreamQuery`
- 等待直到stream里的队列为空，即`cudaStreamSynchronize`

## stream之间的操作: cuda event 

例子1： 用于stream之间的等待

![img](https://pica.zhimg.com/v2-cac7adb67a2c5bf46df4775cc3a170c8_1440w.jpg)

它实现的功能是：

- 在stream 1上面执行kernel 1 和 kernel 3
- 在stream 2上面执行kernel 2，但是必须等到stream 1上面的kernel 1执行结束之后才能开始

为此，我们在stream 1上面launch kernel 1之后，创建一个event来记录当前stream 1的状态（具体来说就是队列里有哪些kernel还没执行），然后把这个event放进stream 2的队列里，让stream 2执行一个想象中的“wait for event 1”的kernel，这个kernel直到kernel 1执行结束之后才离开队列。于是，在这个event之后加入队列的kernel 2，就必须等到kernel 1执行结束之后才能开始了。

这样的好处在于，我们不仅控制了kernel 1和 kernel 2的执行顺序，而且把kernel 2和kernel 3并发执行了，节省了时间（蓝色区域为两个kernel同时执行的时间段）。

例子2: 测量GPU kernel执行结束的时间

`cudaEventRecord`相当于把一个event插入到stream的队列里（fake event kernel），记录这个event执行的时刻。

# CUDA Graph

## 为什么需要CUDA Graph？

因为**CPU-bound**。在大模型推理的 **Decode（逐词生成）阶段**，由于每次只生成一个词，矩阵运算量其实非常小。整个网络可能包含几千个极小的算子（层归一化、加法、小矩阵乘法）。 此时，**GPU 真正用于计算的时间只有 10%，而 90% 的时间都在等待 CPU 慢吞吐吐地发送下一条指令。**

## 什么是CUDA Graph？

**算子打包与回放(Capture and Replay)**

就像是给 GPU 录制了一段宏操作（Macro）：

- **捕捉阶段 (Capture)：** 系统先“假跑”一遍模型。在这个过程中，CPU 不会让 GPU 真正去算，而是默默记录下所有的指令、内存分配、核函数调用以及它们之间的前后依赖关系。最终，把它画成一张**有向无环图（Graph）**。
- **回放阶段 (Replay/Launch)：** 当真正开始推理时，CPU 只需要对 GPU 喊一句话：“**执行第 3 号 Graph！**” 然后 CPU 就可以去喝茶了。**GPU 硬件内部的任务调度器会自动接管**，按照预先设定好的图结构，一个接一个地执行完这几千个算子，中间**不需要与 CPU 进行任何通信**。 



# GPU原理

AI计算模式与线程的关系

## GPU基础概念

![image-20260416174308801](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260416174308801.png)

### 硬件基本概念

![img](https://pic2.zhimg.com/v2-aee3e46616d860823f1291433771f6e7_r.jpg)

GPC  图形处理簇 

TPC texture processed Cluster  纹理处理簇

SM 流多处理器 stream Multiprocessors

HBM  高带宽存储器 Hign Bandwidth Memory

#### SM stream Multiprocessors

CUDA Core  Tensor core

在CUDA中的作用，可以并发地执行数百个线程，一个block上线程是放在同一个SM，一个SM的有限Cache制约了每个block的线程数。

SP Stream Processor 最基本的处理单元。后续CUDA Core也被淘汰了

Warp 线程束

逻辑上所有Thread是并行，但是从硬件角度来说，并不是所有Thread能够在同一时刻执行，这里就需要Warp引入。

Warp是SM的基本执行单元，一个Warp包含32个并行的Thread，这32个Thread执行与SIMT模式。Thread以锁步的方式执行同一条指令。

### CUDA Compute Unified Device Architecture 

#### 线程层次结构

核 Kernel

CUDA执行流程中最重要的过程是，调用CUDA的核函数来执行并行计算。

在CUDA程序架构中，Host代码部分在CPU上执行，是c代码；当遇到数据并行处理的部分，CUDA就会将程序编译成GPU能执行的程序，并传送到GPU，这个程序在CUDA里称为核Kernel。

网络 grid

kernel在device(GPU)上执行时，实际上是启动很多线程，一个Kernel启动的所有线程称为一个网络grid。同一个grid上的线程共享相同的全局内存空间。

线程块 Block

Grid分为多个线程块(block)，一个block里包含很多线程。

Block间并行执行，并且无法通信，也没有执行顺序。每个block包含共享内存。

线程 thread

CUDA并行程序，实际上会被多个threads来执行，多个threads会被群组成一个线程block，多个block中threads可以同步，也可以通过shared memory进行通信。

CUDA 跟 NVIDIA 硬件架构关系

Block线程块只在一个SM上通过Wrap进行调度。一旦在SM上调起了Block线程块，就会一直保留到执行完Kernel。SM可以同时保存多个Block线程块，块间并行执行。

然后多个SM堆叠得到Grid，得到GPC，TPC。

## 峰值算力

![image-20260416180944159](/Users/yiwangwei/Library/Application Support/typora-user-images/image-20260416180944159.png)

















