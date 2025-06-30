# MPEG Immersive Video (MIV) 简介

![](i/home_panel_06.jpg)

MIV 是 Moving Picture Experts Group (MPEG) 出品的一种3D视频格式，其原理是用多视角下的色彩图(texture)和深度图(geometry)表示场景，从而将3D内容映射为2D内容并使用传统2D视频编码器进行编码。
此类技术又称 Multi-view Video 。

👇一段介绍，来自 C. Zhu, G. Lu, B. He, R. Xie and L. Song, “Implicit-Explicit Integrated Representations for Multi-View Video Compression,” in IEEE Transactions on Image Processing, vol. 34, pp. 1106-1118, 2025

>Over the past decade, the Moving Picture Experts Group (MPEG) has been committed to the development of multi-view video coding standards. Popular coding standards, such as [3D-HEVC](http://hevc.info/3dhevc) [1] and [MIV](https://mpeg-miv.org/) [2], rely on disparity to eliminate inter-view redundancy.
>
>[1] G. Tech, Y. Chen, K. Müller, J.-R. Ohm, A. Vetro, and Y.-K. Wang, “Overview of the multiview and 3D extensions of high efficiency video coding,” IEEE Trans. Circuits Syst. Video Technol., vol. 26, no. 1, pp. 35–49, Jan. 2016.
>
>[2] J. M. Boyce, “MPEG immersive video coding standard,” Proc. IEEE, vol. 109, no. 9, pp. 1521–1536, Sep. 2021.

## MIV 的前身：Video-based Point Cloud Compression (V-PCC)

MIV 的前身是 V-PCC，二者技术一脉相承，但彼时的 V-PCC 还只能做到对 3D Object 编码，而 MIV 现在已经可以处理各种复杂大场景。

V-PCC 的核心思想类似物体三视图，其将3D物体（点云）在不同视角的多个平面上投影，并分解为一大堆补丁，放入2D视频帧中用2D视频编码器进行编码：

![](i/vpcc.png)

在每个平面上，V-PCC先根据法线对物体表明进行切割，将深度相近的点分到一块补丁，然后用一个合并算法将小块合并为大块：

![](i/vpcc-steps.png)

最后，对于每块补丁，分别存储其中的每个点的颜色(Attribute)和相对于选定平面的深度(Geometry)，放入2D图像上，并用一张 Occupancy map 指示哪块是有用的数据：

![](i/vpcc-atlas.png)