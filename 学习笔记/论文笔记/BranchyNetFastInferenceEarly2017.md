# BranchyNet：Fast Inference via Early Exiting from Deep Neural Networks

```bibtex
@article{BranchyNetFastInferenceEarly2017,
  title = {{{BranchyNet}}: {{Fast Inference}} via {{Early Exiting}} from {{Deep Neural Networks}}},
  shorttitle = {{{BranchyNet}}},
  author = {Teerapittayanon, Surat and McDanel, Bradley and Kung, H. T.},
  year = {2017},
  month = sep,
  archivePrefix = {arXiv},
  eprint = {1709.01686},
  eprinttype = {arxiv},
  journal = {arXiv:1709.01686 [cs]},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning,Computer Science - Neural and Evolutionary Computing},
  language = {en},
  primaryClass = {cs}
}
```

## 为什么需要分支网络

* 神经网络的层数越来越多
  * AlexNet 8 层
  * VGGNet 19 层
  * ResNet 152 层
* 准确率的进一步提升所付出的时间和能量代价越来越大
  * 在Titan X GPU上运行VGGNet的能量和时间消耗是AlexNet的约20倍，但是性能只高了约4%
  * ResNet能量还要消耗更高
* 高能耗和长推断时间使得现在最先进的神经网络技术很难应用于现实中
  * 尤其是能量有限和延迟敏感的场合

## 引入

* 在主要网络的层与层之间加一些分支
  * 这些分支以主要网络中的某一层为输入，输出一个推断结果和置信度
  * 如果某个分支输出了一个置信度较高的分类结果，则可以提前退出而不需要执行完整推断

## 创新点

* 通过提前退出的分支实现边缘端快速推理
  * 大部分样本都能提前退出，由此能耗下降、延迟降低
* 基于联合优化的归一化操作 Regularization via Joint Optimization
  * BranchyNet的联合优化联合了所有退出点位置的梯度，进而也联合了所有位置的归一化量，进而减少了过拟合 **（归一化惩罚大权重防止过拟合，但是为什么叠加归一化量就能叠加防止过拟合的效果？）**
* 减少了梯度消失 Mitigation of Vanishing Gradients
  * BranchyNet的联合优化联合了所有退出点位置的梯度，进而缓解梯度消失的问题

## 相关工作：BranchyNet之前的优化方式

* 网络压缩：在保证正确率的情况下减小模型中的参数数量
  * 压缩后的网络可能不再具有适合GPU计算的明显的规律，因此使用标准的GPU实现很难将网络压缩转化为明显的加速（Converting that reduction into a significant speedup is difficult using standard GPU implementations due to the lack of high degrees of exploitable regularity and computation intensity in the resulting sparse connection structure **什么意思？**）
  * 提取卷积层之间的共享信息并执行等级选择（extract shared information between convolutional layers and perform rank selection **什么意思？**）
    * 与BranchyNet有一定的共同点，有可能是BranchyNet的进一步的优化方法 **（如何实现？）**
* 实现优化：在算法上对计算的过程进行优化
  * 面向多CPU的代码优化
  * 用快速傅里叶变换加速卷积层
  * 更快速的卷积算法
* 防止过拟合：
  * Dropout、L1/L2正则化
  * 通过添加Softmax分支防止过拟合
    * 与BranchyNet防止过拟合原理大致相同
* 减少梯度消失
  * normalized network initialization
  * 归一化层
  * 跨层连接的神经网络模型：ResNet、Highway Networks、Deep Networks with Stochastic Depth等
* Conditional Deep Learning：在网络中的卷积层上加线性分类器，通过输出决定是否提前退出
  * 只有线性分类
  * 没有对分类器进行联合训练

## BranchyNet推断过程

### 第$n$个分支的Softmax层输入

$$
\bm z_{exit_n}=f_{exit_n}(\bm x;\theta)
$$

* $\bm z_{exit_n}=f_{exit_n}(\cdot)$：第$n$个分支的Softmax层输入（第$n$个退出点的输出）
  * 为一维数组，为所有可能的识别结果输出一个数（基础知识，不多讲）
* $\bm x$：神经网络的输入
* $\theta$：模型的所有参数

### 第$n$个分支的识别结果

$$
\hat{\bm y}_{exit_n}=softmax(\bm z_{exit_n})=\frac{e^{\bm z_{exit_n}}}{\sum_{c\in\mathcal{C}}e^{z_{(exit_n,c)}}}
$$

* $\hat{\bm y}_{exit_n}$：第$n$个分支的识别结果（第$n$个分支的Softmax层输出的分类标签）
  * 为一维数组，为所有可能的识别结果打分，表示$x$属于该类别的“可能性”
* $\mathcal{C}$：所有可能的识别结果
* $z_{(exit_n,c)}$：函数$\bm z_{exit_n}=f_{exit_n}(\cdot)$为可能的识别结果$c\in\mathcal{C}$输出的数

### 交叉熵

$$
entropy(\bm y)=\sum_{c\in\mathcal{C}}y_clog(y_c)
$$

### 退出判定过程

$n\in[1,N]\cap\mathbb{N}$

从入口开始一层一层计算，对于每个退出点：
* 如果$entropy(\hat{\bm y}_{exit_n})<T_n$，则输出$\hat{\bm y}_{exit_n}$，停止计算；
* 否则继续计算，到下一个退出点，重复上一步。

其中，$T_n$为每个点的交叉熵阈值（交叉熵越小，说明$\hat{\bm y}_{exit_n}$中的值高低差别越大，说明可信度越高。基础知识，不多讲）。

## BranchyNet训练过程：以交叉熵代价函数为例

理论上讲，分支上可以再出分支，为简便起见，本文不考虑这种情况。

### 第$n$个分支的识别代价函数（交叉熵）

$$
L(\hat{\bm y}_{exit_n},\bm y;\theta)=-\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}y_clog(\hat{y}_{(exit_n,c)})
$$

* $L(\hat{\bm y}_{exit_n},\bm y;\theta)$：第$n$个分支的识别代价函数（交叉熵）
* $\bm y$：输入样本的正确分类标签
* $y_c=\{0,1\}$，格式同$\hat{\bm y}_{exit_n}$（基础知识，不多讲）

### 用于联合优化的代价函数

$$
L_{BranchyNet}(\hat{\bm y},\bm y;\theta)=\sum_{n=1}^Nw_nL(\hat{\bm y}_{exit_n},\bm y;\theta)
$$

* $L_{BranchyNet}(\hat{\bm y},\bm y;\theta)$：用于联合优化的代价函数
* $N$：退出点的数量
* $w_n$：为退出点$exit_n$选的权重

## 实验：超参数选择问题

* 为退出点$exit_n$选的权重$w_n$：为靠近入口的退出点选大权重使得后面分支的分类准确率更高（因为归一化效果）
* 交叉熵阈值$T_n$：根据应用的不同而定
* 退出点位置：第一个退出点位置的最优选择依赖于数据集难度。数据集越难，第一个退出点的位置应该越靠后
* 分支层数：靠近入口的分支层数应该更高（直觉上感觉很正确）

## 思考

* 高能耗和长推断时间使得现在最先进的神经网络技术很难应用于现实中
  * 高能耗->低能耗的场合不能用->移动设备上用不了->但是现在民用设备中移动设备占主流->不能走入寻常百姓家
  * 长推断时间->即使移动设备将计算放到云端降低了能耗，实时性要求也满足不了->不能走入寻常百姓家
  * 如果这些问题无法解决，神经网络技术终将被下一代移动网络所抛弃
* 一种构建神经网络的全新思路，由此生发开去，通过分支可以衍生的结构无穷无尽，由此所产生的分支优化问题也无穷无尽
  * 从何处产生分支？分支使用何种结构？分支退出条件如何确定？实际场景下各个部件应该通过何种方式连接？......
  * 本文可能是神经网络实用化走向移动计算的起点

### 关于训练方式

本文在训练分支时会将所有分支的代价算在一起进行梯度下降，因此其训练过程不太方便分布式地进行

* 分支的判断结果之间不会互相影响，如果完全按照树的思想进行训练，梯度下降应该是从分支开始到主干：
  * 分支的代价函数互相独立互不影响
  * 分支的代价反向转播到主干，与主干上的已有的代价叠加传播
* 既然分支的判断结果之间不会互相影响，那为什么要算在一起？
  * 因为每个分支都会输出完整的分类结果
  * 如果按照上述树的思想进行训练，那么必然导致靠近入口处的分支和主干训练的结果和一个层数不足的神经网络相近，或者导致分支间的误差传播在主干上互相干扰（叠加）

设想一种不会产生干扰的**树形神经网络**训练方法（效果可能不行）：
* 将分支看成是独立的神经网络，其输入是主干某层的输出
* 分支是独立的神经网络，互相之间的误差传播独立进行，互不影响
* 分支的误差不传播到主干

（树形神经网络是否已有研究？待确认）

### 由BranchNet开始的“分布式人工智能基础设施”胡思乱想

基于上一节中的树形神经网络设想一种大范围的多功能神经网络模型：

1. 不同的移动设备在不同的区域，其所接受到的输入也不一定会遵循同样的分布（飞机场的摄像头识别飞机更多，机场公路的摄像头识别车更多，机场入口的摄像头识别人更多），如果这三处摄像头部署的分支不同，那么是否能设计后续的网络使得这三个处神经网络识别不出来的数据发往同一个后续网络进行进一步识别？
   * **和分别用三个分支网络相比，这个好像没有太大优势**
   * 当入口附近的神经网络结构有多种时（有的模型更擅长识别车，有的更擅长识别飞机，等等），应该如何对模型进行训练？
     * 上一节中的树形神经网络或许能作为参考
2. 如果可以，这样的网络如何执行分布式训练？
3. 更进一步，如果多个不同的分支网络可以使用同一个网络作为后续识别的网络，那么则可以建立一个这样的训练系统：
  * 每一个网络部署单元由两个网络和其所需的后续网络的标记构成$D_i=(M_i(\cdot),N_i(\cdot),j)$
    * $M_i$的输出是识别结果和置信度
    * $N_i$的输出用于输入到后续网络中
    * 一个通常的分支网络中，$M_i$和$N_i$应该共用前几层
  * 对于某个输入$x$，若$M_i(x)$输出的置信度达到要求，则返回结果
  * 否则，计算$x'=N_i(x)$，根据标记$j$查找后续网络$D_j$，将$x'$传输到$D_j$中
  * 此过程不断级联，直到最后得到了一个识别结果
4. 这样$D_i$不需要知道$D_j$后续如何运作，$D_j$也不需要知道$D_i$的数据来自何方，一个分支网络可以被任意地拆分并放置到不同的地方
5. “基础设施”在边缘端或云端提供了一系列不同结构的$D_i$，开发者可以根据自己的需要在终端开发不同的$D_i$，并根据需要使用边缘端或云端的$D_i$
6. “基础设施”负责考虑各种功耗延迟之类的限制在云端和边缘端优化$D_i$的调度方案，使得有人要用某个$D_i$的时候能在尽量短的延迟时间用上（包括传输延迟和计算延迟）
