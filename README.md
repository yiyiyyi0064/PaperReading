### PaperReading
## 1. Conformer
Conformer 是一种卷积和Transformer相结合的架构，也算是我认真读完的第一篇论文（因为比较短hhh），可能是我积累的比较少，里面module的设计感觉没有特别精妙的地方，给人感觉更像是trick的堆砌，但自己也在VoxCeleb2的数据上train了一下。确实结果比较好。对于这篇文章的其他感想，等我精读完Attention is all you need再一起来说。
## 2. Attention is all you need
觉得title实在是太有范了，整个文章阐述了Transformer架构，没有使用CNN or RNN，这个架构基于Self-attention，由于没有像RNN那样的Sequentia operations所以给出了并行化的答案。这篇文章在看原文的时候还同时看了很多参考的解读，但要进一步再去理解还需要自己去复现以及调参数试试。而且在看的时候自己其实也有些思考，也找了很多相关的论文准备继续看看，看能不能解答我的困惑。
越看越觉得要学习的东西实在太多啦，不过很有意思还挺喜欢这种感觉哈哈哈。
## 3. Accelerating Large Language Model Decoding with Speculative Sampling
刚开始其实觉得这个想法看起来不新鲜，但是读完后觉得文章的具体实现part很精彩，还有一篇很related的work是Fast inference from transformers via speculative decoding，读完之后再对比一下。先写到这，具体实现还没完成，完成后再调一调看能不能解决自己的问题，特别是与kvcache相关的东西，貌似还涉及到了另一个技术pagedattention，和上过的system课的内容很相似。
## 4. Efficient Memory Management for Large Language  Model Serving with PagedAttention
这篇其实内容量不小，而且涉及到vLLM，目前读完主要是对Paged attention这个深入了点，其他部分看完了有点仅停留在概念的感觉。打算做一下Nano-vLLM，结合代码来把论文中一些东西理解一下。