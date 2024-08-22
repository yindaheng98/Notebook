# Dense Reconstruction

一言以蔽之，Dense Reconstruction就是根据输入的RGB图像重建出高密度点云

## Colmap中的Dense Reconstruction

在[《【摘录】相机位姿估计相关代码解读》](./相机代码.md)的开头我们进行了相机位姿估计并生成了稀疏点云，还基于估计出的相机参数将输入图像扭回了针孔相机模式。
接下来即可进行Dense Reconstruction。

还记得`image_undistorter`生成了两个`.sh`文件？里面就写着Dense Reconstruction的代码。

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
  --output_path .\fused.ply
$COLMAP_EXE_PATH/colmap poisson_mesher \
  --input_path .\fused.ply \
  --output_path .\meshed-poisson.ply
$COLMAP_EXE_PATH/colmap delaunay_mesher \
  --input_path . \
  --input_type dense \
  --output_path .\meshed-delaunay.ply
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
  --output_path .\fused.ply
$COLMAP_EXE_PATH/colmap poisson_mesher \
  --input_path .\fused.ply \
  --output_path .\meshed-poisson.ply
$COLMAP_EXE_PATH/colmap delaunay_mesher \
  --input_path . \
  --input_type dense \
  --output_path .\meshed-delaunay.ply
```

可以看到就只有`--PatchMatchStereo.geom_consistency`参数不一样。