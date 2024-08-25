# Dense Reconstruction in Multi View Stereo (MVS)

一言以蔽之，Dense Reconstruction就是根据输入的RGB图像和稀疏重建结果重建出高密度点云。
其中包括深度图估计、点云融合及过滤操作。

![](i/uz61k20i93.jpeg)

## Colmap中的Dense Reconstruction

![](zhimg.com/v2-9b7cccdf799d3b2fdcb24d2f253c223a_r.jpg)

在[《【摘录】相机位姿估计相关代码解读》](./相机代码.md)的开头我们进行了相机位姿估计并生成了稀疏点云，还基于估计出的相机参数将输入图像扭回了针孔相机模式。
接下来即可在Colmap中进行Dense Reconstruction。

还记得`image_undistorter`生成了两个`.sh`文件？里面就写着Dense Reconstruction的例程。

`run-colmap-geometric.sh`:

```sh
# You must set $COLMAP_EXE_PATH to 
# the directory containing the COLMAP executables.
$COLMAP_EXE_PATH/colmap patch_match_stereo \
  --workspace_path . \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size 2000 \
  --PatchMatchStereo.geom_consistency true
$COLMAP_EXE_PATH/colmap stereo_fusion \
  --workspace_path . \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path ./fused.ply
$COLMAP_EXE_PATH/colmap poisson_mesher \
  --input_path ./fused.ply \
  --output_path ./meshed-poisson.ply
$COLMAP_EXE_PATH/colmap delaunay_mesher \
  --input_path . \
  --input_type dense \
  --output_path ./meshed-delaunay.ply
```

`run-colmap-photometric.sh`:

```sh
# You must set $COLMAP_EXE_PATH to 
# the directory containing the COLMAP executables.
$COLMAP_EXE_PATH/colmap patch_match_stereo \
  --workspace_path . \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size 2000 \
  --PatchMatchStereo.geom_consistency false
$COLMAP_EXE_PATH/colmap stereo_fusion \
  --workspace_path . \
  --workspace_format COLMAP \
  --input_type photometric \
  --output_path ./fused.ply
$COLMAP_EXE_PATH/colmap poisson_mesher \
  --input_path ./fused.ply \
  --output_path ./meshed-poisson.ply
$COLMAP_EXE_PATH/colmap delaunay_mesher \
  --input_path . \
  --input_type dense \
  --output_path ./meshed-delaunay.ply
```

其中的四个指令分别是：
* `patch_match_stereo`深度估计: 利用 Patch Match 在输入图像之间进行Patch匹配从而估计深度。这一步会在`stereo/depth_maps`和`stereo/normal_maps`中生成深度图和法线图
* `stereo_fusion`多目融合: 每张深度图都对应一个点云，这一步把各图像的点云融合成一个点云
* `poisson_mesher`: 一种网格生成算法，使用 Poisson 表面重建生成彩色点云 Mesh
* `delaunay_mesher`: 一种网格生成算法，基于 Delaunay 三角剖分生成Delaunay Mesh

可以看到geometric和photometric就只有`--PatchMatchStereo.geom_consistency`和`--input_type`参数不一样。

在`patch_match_stereo`中，其深度一开始是随机赋值的，在对于纹理缺少的地方，无法获得准确深度信息而残留随机值，所以初始计算的深度图在纹理缺失的光滑表面是不光滑的。这里的`--PatchMatchStereo.geom_consistency`就是通过光学一致性和几何一致性约束，对初始深度图进行过滤。详情可见[*Pixelwise View Selection for Unstructured Multi-View Stereo*](https://demuc.de/papers/schoenberger2016mvs.pdf)

## Bundle Adjustment

Bundle Adjustment中文译为 光束法平差、束调整、捆集调整、捆绑调整 等等。

Bundle Adjustment的本质是一个优化模型，其目的是**最小化重投影误差**

所谓bundle，来源于bundle of light，其本意就是指的光束，这些光束指的是三维空间中的点投影到像平面上的光束，而重投影误差正是利用这些光束来构建的，因此称为光束法强调光束也正是描述其优化模型是如何建立的。剩下的就是平差，那什么是平差呢？借用一下百度词条 测量平差 中的解释吧。

>由于测量仪器的精度不完善和人为因素及外界条件的影响，测量误差总是不可避免的。为了提高成果的质量，处理好这些测量中存在的误差问题，观测值的个数往往要多于确定未知量所必须观测的个数，也就是要进行多余观测。有了多余观测，势必在观测结果之间产生矛盾，测量平差的目的就在于消除这些矛盾而求得观测量的最可靠结果并评定测量成果的精度。测量平差采用的原理就是“最小二乘法”。

### 重投影误差

![](i/7985bf628f696afde4b763409c198048.jpeg)

* 投影：利用这些图像对一些特征点进行三角定位(triangulation)，计算特征点在三维空间中的位置
* 重投影：利用相机参数和投影计算得到的三维点的坐标，将这些三维点投影到相机成像平面上
* 重投影误差：输入图像上的像素点位置和其在重投影图像上的位置之差

设相机$i$的内参为$\bm K_i$（已知）、外参为$[\bm R_i|t_i]$（优化变量）、点$X_j$（优化变量）在相机$i$中拍摄到的图像归一化坐标系上的坐标为$[u_{ij},v_{ij}]$（已知）。
则该点在重投影图像上的位置为$z[\hat u_{ij},\hat v_{ij},1]^T=\bm K_i(\bm R_iX_j+t_i)$。
于是重投影误差为：

$$E(T_i,X_j)=\sum_{i}\sum_{j}\sigma_{ij}|[\hat u_{ij},\hat v_{ij},1]-[u_{ij},v_{ij},1]|=\sum_{i}\sum_{j}\sigma_{ij}|\frac{1}{z}\bm K_i(\bm R_iX_j+t_i)-[u_{ij},v_{ij},1]|$$

其中当点$X_j$在相机$i$中有投影时$\sigma_{ij}=1$，否则为$\sigma_{ij}=0$。

于是Bundle Adjustment的优化问题为：

$$\min\limits_{\bm R_i,t_i,X_j}E(T_i,X_j)$$

求解方法：[梯度下降法、Newton型方法、Gauss-Newton方法、Levenberg-Marquardt方法等](https://optsolution.github.io/archives/58892.html)

#### 梯度下降法

懂得都懂，详略

最速下降法保证了每次迭代函数都是下降的，在初始点离最优点很远的时候刚开始下降的速度非常快，但是最速下降法的迭代方向是折线形的导致了收敛非常非常的慢。

#### Newton型方法

现在先回顾一下中学数学，给定一个开口向上的一元二次函数，如何知道该函数何处最小？这个应该很容易就可以答上来了，对该函数求导，导数为0处就是函数最小处。Newton型方法也就是这种思想。

首先将函数利用泰勒展开到二次项：

$$f(\bm x+\Delta \bm x) \approx f(\bm x)+\nabla f(\bm x)\Delta \bm x+\frac{1}{2}\Delta \bm x^T\bm H(\bm x)\Delta \bm x$$

$\bm H$为Hessian矩阵，是二次偏导矩阵。

也就是说Newton型方法**将函数局部近似成一个二次函数进行迭代**，令$\bm x$在$\Delta \bm x$方向上迭代直至收敛，接下来自然就对这个函数求导了：

$$f'(\bm x)=\lim_{\delta\rightarrow 0}\frac{f(\bm x+\Delta \bm x)-f(\bm x)}{\Delta \bm x}\approx \nabla f(\bm x)+\bm H(\bm x)\Delta \bm x$$

于是优化问题为求$\bm x$使得：

$$\nabla f(\bm x)+\bm H(\bm x)\Delta \bm x=0$$

Newton型方法收敛的时候特别快，尤其是对于二次函数而言一步就可以得到结果。但是该方法有个最大的缺点就是Hessian矩阵计算实在是太复杂了，并且Newton型方法的迭代并不像最速下降法一样保证每次迭代都是下降的。

#### 其他方法

更多详情可见[《Bundle Adjustment简述》](https://blog.csdn.net/OptSolution/article/details/64442962)

* Gauss-Newton方法：Gauss-Newton方法避免了求Hessian矩阵，并且在收敛的时候依旧很快。但是依旧无法保证每次迭代的时候函数都是下降的（虽然从上式可以推导出来是下降方向，但是步长可能过长）。
* 利用矩阵稀疏性减少Gauss-Newton方法的计算量：Gauss-Newton方法中需要求解超定参数方程，SVD等方法计算量是$O(n^3)$，面对BA这种超大规模的优化有点不太实用。但是要求解的求解超定参数方程中有很多$\sigma_{ij}=0$可通过稀疏矩阵的Cholesky分解求近似解大幅减少计算量
* Levenberg-Marquardt(LM)方法：LM方法就是在以上方法基础上的改进，通过参数的调整使得优化能在最速下降法和Gauss-Newton法之间自由的切换，在保证下降的同时也能保证快速收敛。
* 李群求解：详情可见[《Bundle Adjustment原理及应用（附代码）》](https://www.bilibili.com/read/cv9304256/)