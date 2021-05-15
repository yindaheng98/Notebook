# Deep Unfolding Network for Image Super-Resolution

```bibtex
@inproceedings{DeepUnfoldingNetworkImage2020,
  title = {Deep {{Unfolding Network}} for {{Image Super}}-{{Resolution}}},
  booktitle = {2020 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  date = {2020-06},
  pages = {3214--3223},
  publisher = {{IEEE}},
  location = {{Seattle, WA, USA}},
  doi = {10.1109/CVPR42600.2020.00328},
  url = {https://ieeexplore.ieee.org/document/9157092/},
  urldate = {2021-05-14},
  archiveprefix = {arXiv},
  eprint = {2003.10428},
  eprinttype = {arxiv},
  eventtitle = {2020 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  isbn = {978-1-72817-168-5},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Electrical Engineering and Systems Science - Image and Video Processing},
  langid = {english}
}
```

## 概括

### 研究现状

* 基于模型的方法：基于图像降质的光学模型对图像进行超分辨率
  * 不需要训练，也没有超参数
  * 能适应不同的缩放尺度、模糊核和噪声
  * 可解释性好
* 基于机器学习的方法：使用机器学习中的一些方法对图像进行超分辨率
  * 需要大量数据进行训练
  * 训练数据决定了适用范围，单一个模型无法适应多种不同的缩放尺度、模糊核和噪声
  * 效果比基于模型的方法好

### 本文贡献

结合基于模型的方法和基于机器学习的方法进行超分辨率

## 本文使用的基于模型的方法：Deep Unfolding

Deep Unfolding 是前人已有的研究，是一种基于光学模型的迭代求解方法

参考论文：《Learning Deep CNN Denoiser Prior for Image Restoration》

### 图像降质的光学模型

$$\bm y=(\bm x\otimes\bm k)\downarrow_{\bm s}+\bm n$$

1. $(\bm x\otimes\bm k)$：原始高清图像$\bm x$与一个模糊核$\bm k$卷积
2. $\downarrow_{\bm s}$：进行$\bm s\times\bm s$下采样（其实就是池化）
3. $\bm n$：噪声

### MAP(最大后验概率) framework 估计高清图

已知模糊核$\bm k$，使$E(\bm x)$最小的高清图像$\bm x$就是最大后验概率估计得到的高清图

$$E(\bm x)=\frac{1}{2\sigma^2}\parallel\bm y-(\bm x\otimes\bm k)\downarrow_{\bm s}\parallel^2+\lambda\Phi(\bm x)$$

1. $\frac{1}{2\sigma^2}\parallel\bm y-(\bm x\otimes\bm k)\downarrow_{\bm s}\parallel^2$：保真项
2. $\lambda\Phi(\bm x)$：用于处理噪声的惩罚项，$\lambda$为trade-off参数

### 半二次分裂(half-quadratic spliting, HQS)算法

原问题：

$$min_x f(x)+g(x)$$

半二次分裂近似问题：

$$\begin{aligned}
&min_{x,z}&f(x)+g(z)\\
&s.t.&x=z
\end{aligned}$$

增广拉格朗日：

$$\mathcal L(x,z;\mu)=f(x)+g(z)+\frac{\mu}{2}\parallel x-z\parallel^2$$

半二次分裂迭代求解：

$$\left\{\begin{aligned}
z_{k+1}&=argmin_z f(x_k)+g(z)+\frac{\mu}{2}\parallel x_k-z\parallel^2\\
x_{k+1}&=argmin_x f(x)+g(z_k)+\frac{\mu}{2}\parallel x-z_k\parallel^2\\
\end{aligned}\right.$$

固定$x$优化$z$、固定$z$优化$x$，如此循环往复逼近最优解。

### MAP framework + HQS = Deep Unfolding

原问题：

$$min_{\bm x} E(\bm x)=\frac{1}{2\sigma^2}\parallel\bm y-(\bm x\otimes\bm k)\downarrow_{\bm s}\parallel^2+\lambda\Phi(\bm x)$$

半二次分裂近似+增广拉格朗日：

$$\mathcal L(\bm x,\bm z;\mu)=\frac{1}{2\sigma^2}\parallel\bm y-(\bm z\otimes\bm k)\downarrow_{\bm s}\parallel^2+\lambda\Phi(\bm x)+\frac{\mu}{2}\parallel\bm z-\bm x\parallel^2$$

迭代：

$$\left\{\begin{aligned}
\bm z_{k}&=argmin_z \parallel\bm y-(\bm z\otimes\bm k)\downarrow_{\bm s}\parallel^2+\mu\sigma^2\parallel\bm z-\bm x_{k-1}\parallel^2\\
\bm x_{k}&=argmin_x \lambda\Phi(\bm x)+\frac{\mu}{2}\parallel\bm z_k-\bm x\parallel^2\\
\end{aligned}\right.$$

## Deep Unfolding + 机器学习

### Data Module

### Prior Module

### Deep Unfolding 步骤1：找出模糊核$\bm k$

$$\bm k_{bicubic}^{\times\bm s}=argmin_{\bm k}\parallel(\bm x\otimes\bm k)\downarrow_{\bm s}-\bm y\parallel$$

用大量数据学出最合适的$\bm k$