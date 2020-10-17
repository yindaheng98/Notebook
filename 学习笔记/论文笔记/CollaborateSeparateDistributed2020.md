# Collaborate or Separate? Distributed Service Caching in Mobile Edge Clouds

```bibtex
@inproceedings{xuCollaborateSeparateDistributed2020,
  title = {Collaborate or {{Separate}}? {{Distributed Service Caching}} in {{Mobile Edge Clouds}}},
  shorttitle = {Collaborate or {{Separate}}?},
  booktitle = {{{IEEE INFOCOM}} 2020 - {{IEEE Conference}} on {{Computer Communications}}},
  author = {Xu, Zichuan and Zhou, Lizhen and {Chi-Kin Chau}, Sid and Liang, Weifa and Xia, Qiufen and Zhou, Pan},
  year = {2020},
  month = jul,
  pages = {2066--2075},
  publisher = {{IEEE}},
  address = {{Toronto, ON, Canada}},
  doi = {10.1109/INFOCOM41043.2020.9155365},
  isbn = {978-1-72816-412-0},
  language = {en}
}
```

## 背景

* 将内容放到边缘侧的5G基站，使得用户在访问时延迟更低
* 将内容永久放置于边缘侧不划算，因为边缘侧的计算资源非常昂贵
* 可以只将经常访问的资源放到边缘侧

## 研究对象：边缘缓存系统

* 多个网络服务提供者
* 基础设施提供商构建的边缘端的移动服务市场
  * 服务提供商通过租用虚拟机的方式使用边缘服务器
  * 多个服务提供商可以共享一个虚拟机
* 有限的计算和带宽资源
  * 多个服务提供商在边缘端竞争有限的资源，且它们只关心自己的收益，不能统一规划
  * 一个边缘服务器上运行着多个服务
  * 一个边缘服务器上只运行一个服务

## 研究问题

* 设计一个分布式的资源配置算法
  * 使得每个服务提供商都有动力参与其中
  * 没有任何服务提供商能通过违反规定增加自己的效用
* 网络服务提供商的自私行为导致网络偏离social optimum最佳状态 **（应该是博弈论术语？？）**
  * 需要有一个偏离social optimum最佳状态的近似最优解 **（应该是博弈论术语？）**
  * 设计算法使得服务提供商获得非负收益
* 降低服务缓存的代价
  * 为了降低消耗，几个服务提供商可能会共用一个虚拟机
  * 设计资源分配算法尽可能降低多个服务提供商共用虚拟机带来的消耗
* 选择合适的5G服务缓存位置

## 已有的相关研究

* 以内容为中心的网络：专注于有限存储容量边缘节点上的内容缓存，不关注内容处理的过程（意思应该是这些内容都是静态的不需要计算）
  * 优化路径选择
  * 优化服务放置
  * 优化内容冗余策略
  * ......
* 任务卸载和服务放置：假定服务只部署于移动边缘端，没有考虑从远程数据中心缓存服务和更新缓存的情形

## 创新点

* 公式化描述了云边缘的服务缓存问题，包括多个服务提供商共享资源和不共享资源的情况
* 对于不共享资源的情况，公式化描述了一个整数线性规划解法，并且设计了一个较好近似的随机算法，同时保持了良好的资源冲突 **（maintaining moderate resource violations？难道还有算法能消除资源冲突？这个violations到底指什么？）**
* 为资源共享设计了一个新的coalition formation game，这个game的目标是最小化服务提供商的总消耗
* 对于可证明纳什均衡的coalition formation game，设计了一种机制，guarantees the worst-case performance gap between the obtained social cost and the optimal one **（确保获得的社会成本与最优成本之间的最坏情况下的绩效差距？？？什么意思）**
* 通过模拟评估了算法的性能

## 研究方法

### 一个边缘服务器上只运行一个服务的情况

* Integer Linear Program (ILP) and a randomized
* rounding algorithm

### 一个边缘服务器上运行着多个服务的情况

* 多个服务的资源共享策略：distributed and stable game-theoretical mechanism
* 引入一种新的cost sharing模型和联盟博弈模型，最小化所有网络服务的social cost

## 建模

### 在边缘服务器缓存应用相比不缓存时增加的利润（不与其他服务商共享边缘服务器时）

$$
u_l^{dft}(CL_i)=v_l(d_l^{DC}-d_{l,i})-c_{l,i}
$$

* $u_l^{dft}(CL_i)$：在边缘服务器$CL_i$部署服务$S_l$所产生的利润
* $v_l$：服务$S_l$在被用户访问时每减小单位时间的延迟对服务提供商所产生的利润
* $d_l^{DC}$：向云计算中心请求服务$S_l$的延迟
* $d_{l,i}$：向边缘服务器$CL_i$请求服务$S_l$的延迟
* $c_{l,i}$：在边缘服务器$CL_i$部署服务$S_l$所产生的成本

### 在边缘服务端部署应用的成本

$$
c_{l,i}=c^p_{l,i}C^{vm}_i+c^b_{l,i}B^{vm}_i
$$

* $c_{l,i}$：在边缘服务器$CL_i$部署服务$S_l$所产生的成本
* $c^p_{l,i}$：服务$S_l$在边缘服务器$CL_i$使用单位计算量所产生的成本
* $C^{vm}_i$：虚拟机在边缘服务器$CL_i$所使用的计算量
* $c^b_{l,i}$：服务$S_l$在边缘服务器$CL_i$使用单位带宽所产生的成本
* $B^{vm}_i$：虚拟机在边缘服务器$CL_i$所使用的带宽

### 在边缘服务器缓存应用相比不缓存时增加的利润（与其他服务商共享边缘服务器时）

注：**每个服务提供商$sp_l$只提供一种服务$S_l$**

$$
u_l^{coll}(g_i)=v_l(d_l^{DC}-d_{l,i})-p_l(g_i)
$$

* $u_l^{coll}(g_i)$：共享边缘服务器$CL_i$联盟$g_i$对服务提供商$sp_l$产生的利润
* $p_l(g_i)$：服务提供商$sp_l$（或也可以看作服务$S_l$）留在联盟$g_i$的成本

### 在边缘服务器缓存应用时，与其他服务商共享边缘服务器相比不共享增加的利润

$$
\begin{aligned}
u_l(g_i)&=u_l^{coll}(g_i)-u_l^{dft}(CL_{i'})\\
&=v_l\left(d_{l,i}-d_{l,i'}\right)-\left(p_l(g_i)-c_{l,i'}\right)\\
\end{aligned}
$$

* $u_l(g_i)$：服务提供商$sp_l$共享边缘服务器$CL_i$组成联盟$g_i$相比于不共享时所产生的额外利润
* $u_l^{coll}(g_i)$：共享边缘服务器$CL_i$组成联盟$g_i$对服务提供商$sp_l$产生的利润
* $u_l^{dft}(CL_{i'})$：不共享边缘服务器时在边缘服务器$CL_i$部署服务$S_l$对服务提供商$sp_l$产生的利润
* $d_{l,i}$：共享边缘服务器$CL_i$时访问服务$S_l$的延迟
* $d_{l,i'}$：不共享边缘服务器$CL_i$时访问服务$S_l$的延迟
* $c_{l,i'}$：不共享边缘服务器$CL_i$时时在边缘服务器$CL_i$部署服务$S_l$的成本

### 合作成本

$$
c_l^{coll}(g_i)=p_l(g_i)-v_l\left(d_{l,i}-d_{l,i'}\right)
$$

* $c_l^{coll}(g_i)$：合作成本

因此有

$$
u_l(g_i)=c_{l,i'}-c_l^{coll}(g_i)
$$

## 优化

### 不共享边缘情况下，成本敏感的服务缓存问题

在系统容量的限制范围内使多个服务提供商边缘部署的总成本降到最低

$$
\begin{aligned}
    min&\sum_{l=1}^L\sum_{i=1}^{|\mathcal{CL}(S_l)|}x_{li}c_{l,i}&\\
    subject\ to:&&\\
    &\sum_{i=1}^{|\mathcal{CL}(S_l)|}x_{li}=1&\forall S_l\in\mathcal S\\
    &\sum_{l=1}^Lx_{li}\leq M&\forall CL_i\in\mathcal{CL}(S_l)\\
    &x_{li}\in\{0,1\}&
\end{aligned}
$$

* $x_{li}$：是否在边缘服务器$CL_i$部署服务$S_l$
* $\mathcal{S}$：所有服务提供商所提供的应用的集合（再次强调每个服务提供商只提供一种服务）
* $\mathcal{CL}(S_l)$：服务提供商$sp_l$可以使用的边缘服务器集合
* $M$：一个边缘服务器最多可以由$M$服务提供商使用

解题思路需要凸优化才能理解，目前略

### 共享边缘情况下，成本和延迟敏感的服务缓存问题

在系统容量的限制范围内使边缘部署的social cost $c_l^{coll}(g_i)$降到最低

公式和思路需要学习博弈论才能理解，目前略