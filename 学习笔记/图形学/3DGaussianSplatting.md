# 3D Gaussian Splatting

## Splatting

抛雪球算法的英文名称为Splatting方法，也称为足迹法（Footprint），它与光线投射法不同，是反复对体素的投影叠加效果进行运算。它用一个称为足迹的函数计算每一体素投影的影响范围，用高斯函数定义点或者小区域像素的强度分布，从而计算出其对图像的总体贡献，并加以合成，形成最后的图像。由于这个方法模仿了雪球被抛到墙壁上所留下的一个扩散状痕迹的现象，因而取名为“抛雪球法”。

## 光栅化

## $\alpha$-Blending

## 如何表示Gaussian点

## 如何训练Gaussian点云

### 如何移动Gaussian点

位置梯度

### 如何增加Gaussian点

![](i/20231218204732.png)

需要增加Gaussian点的区域分两种：

* “欠重构区”(under-reconstruction)：缺少Gaussian点的区域
* “过重构区”(over-reconstruction)：每个Gaussian点覆盖很大范围的区域

寻找这些区域：

* “欠重构区”(under-reconstruction)：位置梯度很大但值很小的Gaussian点
* “过重构区”(over-reconstruction)：位置梯度很大且值很大的Gaussian点

增加Gaussian点：

* “欠重构区”(under-reconstruction)：复制这个Gaussian点，再沿位置梯度进行移动
* “过重构区”(over-reconstruction)：减小这个Gaussian点的值，再复制这个Gaussian点，再沿位置梯度进行移动

### 如何减少Gaussian点

减少Gaussian点主要是为了：

* 删除不重要的Gaussian点以减小计算量
* 应对相机面前的错误点(floaters close to the input cameras)

减少Gaussian点的方法：

* 删除不重要的Gaussian点以减小计算量：在训练过程中会有一些Gaussian点的透明度不断下降到小于给定阈值，这些Gaussian点在训练过程中删除
* 应对相机面前的错误点：
  1. 每隔一段时间就将所有Gaussian点的透明度设为0
  2. 禁用点的增减和移动功能，仅对Gaussian点的参数进行一段时间训练
     * 有些Gaussian点会在训练过程中变大，盖住其他点或者盖住相机，删除这些点
  3. 训练完成后会有一些Gaussian点透明度仍然接近0，删除这些点

#### 相机面前的错误点(floaters close to the input cameras)是什么

理论上讲，在每个输入视角面前摆上输入图片，就可以让NeRF训练误差降到0。
这也是NeRF的一种过拟合，正因为如此，NeRF在训练过程中会有概率在输入视角附近训练出一些错误点。
尤其是在相机参数的有误差的时候，输入图像上的某些点时无解的，这是如果用DNN较大的NeRF进行过度的训练，NeRF不管怎么样都没法在场景中找到某些点的最优解，就会将点放在相机面前。

## 可微的Gaussian点云快速光栅化方法
