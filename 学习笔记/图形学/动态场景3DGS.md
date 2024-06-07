# 3D Gaussian Splatting 用于动态场景表示

## (3DV 2024) Dynamic 3D gaussians: Tracking by persistent dynamic view synthesis

## (ICLR 2024) Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

![](i/teaser.png)

3D Gaussian形状参数里的位置、rotate和scale都变成4D，相当于3D Gaussian（椭圆）加一个时间维度变成了4D椭圆，渲染是**在时间轴上采样从而将这个4D椭圆投影到3D空间**。

文中介绍的3D Gaussian：

![](i/20240606204542.png)

拓展到4D Gaussian：

![](i/20240606204622.png)

缩放3D变4D在数学上是3x3对角矩阵变成4x4对角矩阵$S$；旋转3D变4D在数学上是两个啥矩阵相乘得到$R$；
从而4D Gaussian均值（中心点坐标）变成4维$\mu$、协方差也变成4x4矩阵$\Sigma$：

![](i/20240606204813.png)

最后，每个时刻的3D Gaussian是从这个4D Gaussian中采样而来（4D椭圆于t时刻在3D空间中的一个投影）：

![](i/20240606205959.png)

球谐系数加上一个维度用傅里叶级数组成的函数表示：

![](i/20240606204458.png)

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

![](i/pipeline_00.jpg)

高斯点云只有一个，用Triplane存储运动信息。

任意高斯点位置xyz和时间t输入Triplane得到位移/旋转/缩放的变化情况，从而对高斯点进行变换。

（只有形状方面的变换，没有颜色和球谐系数的变化）