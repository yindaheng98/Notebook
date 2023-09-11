# Nerf运行流程解析

上接：[体渲染](./体渲染.md)

前置知识：[光线行进算法](./光线行进算法.md)、[相机的内外参数](./相机参数.md)

从论文中给出的系统结构图可知，Nerf中的DNN输入是相机位置$(x,y,z)$和射线朝向$(\theta,\phi)$，其组成5元组表示一条光路；DNN输出是这条光路的各离散采样区间$\delta_n$内的粒子颜色$c_n$和其粒子密度$\sigma_n$（即光学厚度），之后，按照上述离散化体渲染公式进行积分操作，即得到这条光路射出的光线颜色，就像这样：

![](./i/2f54150145095c59f605941a1170f071.gif)

于是，渲染过程如下：
1. 指定相机的外参（位置、方向等）和内参（焦距、视野、分辨率等）根据外参内参计算出需要采样的各光路的$(x,y,z,\theta,\phi)$
2. 将每个光路的$(x,y,z,\theta,\phi)$输入DNN，计算得到光路上每个采样点的各离散采样区间$\delta_n$内的颜色$c_n$和粒子密度$\sigma_n$
3. 按照离散化体渲染公式进行积分操作，即得到每条采样光路的颜色
4. 根据这些采样光路的颜色和相机内参，计算出相机拍到的图像
5. 上面这个离散化体渲染公式很显然是可微的，所以将计算得到的图像和Ground truth作差进行反向传播训练。

![](./i/NerfFlow.png)


## 根据相机外参计算需要采样的光路

函数[get_rays](https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L153)和[get_rays_np](https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L165)
  * `H, W `是分辨率
  * [内参矩阵](./相机参数.md)输入 `K = np.array([[focal, 0, 0.5W], [0, focal, 0.5H], [0, 0, 1]])`是焦距之类的
  * [外参矩阵](./相机参数.md)输入 `c2w`表示“Camera-to-world transformation matrix”，一个3x4矩阵
  * 输出`rays_o, rays_d`大小都是`(H, W, 3)`分别表示图像上各像素点对应需要采样的光线的原点和方向

