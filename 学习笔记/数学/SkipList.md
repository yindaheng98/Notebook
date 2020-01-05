# 跳表(SkipList)的数学思考

## [跳表是什么](https://blog.csdn.net/pcwl1206/article/details/83512600)

## 关于跳表中随机函数

### 限制

* 用户向每次跳表中插入的数据大小满足均匀分布
* 现有一个随机数生成器，其输出的随机数分布由用户指定
* 跳表每次进行插入操作时都使用上述随机数生成器的输出向下取整作为此次插入节点的Index个数
* 跳表每次进行插入操作时都能知道表中已有的元素个数

### 问题

如何指定随机数生成器的概率分布，使得跳表满足以下条件：
* 跳表第1层索引中的索引数量是跳表内元素总数的$1/C$
* 跳表每一层索引中的索引数量都是上一层的$1/C$

#### 注：此类跳表的复杂度分析

##### 时间复杂度

显然，以此设置的跳表查询效率与$C$分查找相同：若记跳表元素总数为$N$，$C=2$就相当于二分查找，时间复杂度为$O(2log_2N)$、$C=3$就相当于三分查找，时间复杂度为$O(3log_3N)$。

依此类推，共有log_CN层索引要查，在每一层索引中要进行至少一次至多$C$次比较，故时间复杂度为

$$O(\frac{C+1}{2}log_CN)=O(Clog_CN)$$

##### 空间复杂度

显然，此设置下跳表的各索引层中的索引数量加上原始链表中的数据量共构成一等比数列，首项为1、末项为$N$、公比为$C$。故其总数据量为：

$$\frac{NC-1}{C-1}=N+\frac{N-1}{C-1}$$

空间复杂度为：

$$O(N+\frac{N-1}{C-1})=O(N)$$

### 解决

假设：
* 跳表中所存入的元素集合$E=\{e_1,e_2,\cdots e_N\}$
* 跳表的索引有$L$层，编号为$1\sim L$
* 第$l$层索引中的索引个数为$_Lk_l$
* 元素$e_i$上的索引个数为$_Ek_i$，且仅分布在第$1\sim{}_Ek_i$层索引中

则问题中所述的条件可表示为：

* ${}_{L}k_1=N/C$
* $\forall(1\le l\le L, l\in \mathbb N){\quad}_{L}k_{l+1}={}_Lk_{l}/C$

显然，前述述条件等效于：

$\forall(1\le l\le L, l\in \mathbb N){\quad}_{L}k_l=N/C^l$

$\because$ 由假设显然，第$l$层索引中的索引个数等于跳表中索引个数不小于$l$的元素的个数，即：

$$_Lk_l=\left| \{e_i|e_i\in E, {}_Ek_i\ge l \}\right|\quad(1\le l\le L, l\in \mathbb N)$$

令$_Lk_0$表示跳表中索引个数不小于$0$的元素的个数，则有：

$$_Lk_l=\left| \{e_i|e_i\in E, {}_Ek_i\ge l \}\right|\quad(0\le l\le L, l\in \mathbb N)$$

$\because$ 显然索引个数为正整数，即 $\forall(e_i\in E){\quad}_Ek_i\ge 0$

$\therefore$ $_Lk_0=\left| \{e_i|e_i\in E, {}_Ek_i\ge 0 \}\right|=N$

$\therefore$ 前述条件等效于：在跳表中所存储的这$N$个节点中，索引个数大于等于$l$的概率为

$$
\begin{aligned}
P({}_Ek_i\geq l)&=\frac{\left| \{e_i|e_i\in E, {}_Ek_i\ge l \}\right|}{N}\\
&=\frac{_Lk_l}{N}\\
&=1/C^l\\
&(0\le l\le L, l\in \mathbb N)
\end{aligned}
$$

$\therefore$ 限制中对随机函数的输出$X$进行向下取整作为插入节点的索引个数，即$l=\lfloor X\rfloor$

$\therefore$ 前述条件等效于：在$N$次数据插入操作中，随机数生成器的输出值$X$满足

$$\forall(0\le l<L+1, l\in \mathbb N)\quad P(X\geq l)=1/C^l$$

$$P(X\ge L+1)=0$$

$\because$ 显然：

$$
\begin{aligned}
P(l\le X< l+1)&=P(X\ge l)-P(X\ge l+1)\\
&=1/C^l-1/C^{l+1}\\
&(0\le l<L-1, l\in \mathbb N)
\end{aligned}
$$
以及
$$
\begin{aligned}
P(L\le X<L+1)&=P(X\ge L)-P(X\ge L+1)\\
&=1/C^{L}
\end{aligned}
$$

$\therefore$ 前述条件等效于：在$N$次数据插入操作中，随机数生成器的输出值$X$应满足：

$$
P(l\le X< l+1)=
\left\{
\begin{aligned}
&1/C^l-1/C^{l+1}&(0\le l<L-1)\\
&1/C^{l}&(l=L)\\
&0&(l\ge L+1)
\end{aligned}
\right.
$$

其中 $l\in \mathbb N$

由此方程组可以看出，在这$N$次数据插入操作中，随机数生成器的输出值$X$应满足的条件只与$N$次插入后跳表的索引层数有关，且输出值$X$在$[0,L-1)$间的概率分布为定值。

因此我们可以有此想法：当插入了足够多的数据使跳表层数$L$接近$+\infin$时，上面的方程组就可以近似为：

$$P(l\le X<l+1)=1/C^l-1/C^{l+1}\quad(l\in \mathbb N_+)$$

此式即可将随机数生成器的概率唯一确定，在实际中适用跳表元素数量很多而不可预估的情况。

### 但是如果跳表长度不长呢？

显然，如果跳表中的元素数量可以预估，那直接使用前面那个方程组即可。使用那个方程组就需要通过$N$确定$L$，即通过预估的跳表元素数量确定跳表层数：

$\because$ 由前述条件易得，在这$N$次数据插入操作完成后，跳表最上层索引中的索引个数应满足：$_Lk_L=N/C^L\leq 1$

$\therefore$ 跳表总层数$L\geq log_CN$

简单地，由用户输入想要的$C$值和预估的跳表元素总量$N$，即可计算跳表总层数$L=\lceil log_CN\rceil$