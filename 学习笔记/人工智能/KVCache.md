# KV Cache 原理

前置知识：[《机器学习中的Attention机制》](./Attention.md) [《Transformer结构》](./transformer.md) [《什么是BERT？》](./BERT.md)

KV Cache 是针对 Transformer 中 Decoder 组件的 masked self-attention 模块推理过程进行性能优化的一个常用技术，该技术以增加显存占用为代价提高 masked self-attention 推理性能，且不影响其计算精度，多用于 decoder-only 的大模型。

![](./i/0__wkGO1660kIWEV1M.gif)

## 复习：masked self-attention 计算

Transformer 架构中的 decoder 运行流程：接收一串 token 输入，输出一个 token，输出的 token 会与输入 token 拼接在一起，然后作为下一次推理的输入，不断反复直到遇到终止符。

![](./i/transformer_decoding_2.gif)

当今的大语言模型通常都是 decoder-only，没有 encoder：

![](./i/v2-3d6d31bba8b5b9cc97e7aa8e50c5b6ea_1440w.gif)

而在 decoder 中，主要的性能瓶颈为 masked self-attention 模块的计算。

在[《Transformer结构》](./transformer.md)中，我们已推导出每次新增一个 token 后，输出矩阵中只需要计算一个新增行$MaskedAttention(Q_{1:t},K_{1:t},V_{1:t})_t$即可，其计算公式为：

$$MaskedAttention(Q_{1:t},K_{1:t},V_{1:t})_t=\sum_{i=1}^tsoftmax\left(\left[\frac{Q_tK^{\top}_1}{\sqrt{d_k}},\frac{Q_tK^{\top}_2}{\sqrt{d_k}},\cdots,\frac{Q_tK^{\top}_t}{\sqrt{d_k}}\right]\right)_iV_i\tag{1}$$

![](./i/1_8xqD4AYTwn6mQXNw0uhDCg.gif)

而[《机器学习中的Attention机制》](./Attention.md)中我们也分析了 self-attention 中的$K$、$Q$、$V$来源，即由输入的词向量$s_i$乘上3个矩阵$W^Q$、$W^K$、$W^V$得来：
$$
\begin{aligned}
Q_i=s_iW^Q\\
K_i=s_iW^K\\
V_i=s_iW^V\\
\end{aligned}
$$

## KV Cache 在 Cache 什么？

从公式$(1)$中可以看出，计算这个新增行需要之前的所有$K_i,V_i$，但是只需要最新的$Q_t$，所以直接就能理解 KV Cache 在干嘛：保存之前的所有$K_i,V_i$，避免每次都要重新计算$K_i=s_iW^K$和$V_i=s_iW^V$。

在没有KV Cache的情况下，每新增一个单词都需要花费$O(n)$计算之前的所有 token 的$K,V$，再花费$O(n)$计算新增行$MaskedAttention(Q,K,V)_i$；而有了KV Cache之后，虽然内存占用的增长随 token 数量增长呈$O(n)$线性增加，但每新增一个 token 就只需要花费$O(1)$计算这个最新 token 对应的$K_i,V_i$即可，再花费$O(n)$计算$MaskedAttention(Q,K,V)_i$。这样用空间的$O(n)$为代价换走了一个$O(n)$的计算过程，还是很划算的。