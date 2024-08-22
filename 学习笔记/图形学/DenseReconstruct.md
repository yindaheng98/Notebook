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
* `poisson_mesher`: 一种种网格生成算法，使用 Poisson 表面重建生成彩色点云 Mesh
* `delaunay_mesher`: 两种网格生成算法，基于 Delaunay 三角剖分生成Delaunay Mesh

可以看到geometric和photometric就只有`--PatchMatchStereo.geom_consistency`和`--input_type`参数不一样。

在`patch_match_stereo`中，其深度一开始是随机赋值的，在对于纹理缺少的地方，无法获得准确深度信息而残留随机值，所以初始计算的深度图在纹理缺失、光滑表面是不光滑的。这里的`--PatchMatchStereo.geom_consistency`就是通过光学一致性和几何一致性约束，对初始深度图进行过滤。详情可见[*Pixelwise View Selection for Unstructured Multi-View Stereo*](https://demuc.de/papers/schoenberger2016mvs.pdf)

## Bundle Adjustment