# 使用 VGGT + VGGSfM 强化 Colmap 原理解析

前置知识：[《VGGT 原理解析》](./VGGT.md)

注意，作者在[demo_colmap.py#L148-L154](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L148)注释中已说明是因为原版VGGT Tracker的实现不够efficiency才用的VGGSfM，并且指出这里的VGGSfM还可以换成任意Point Tacking或匹配模型，比如CoTracker。

```python
# Predicting Tracks
# Using VGGSfM tracker instead of VGGT tracker for efficiency
# VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
# Will be fixed in VGGT v2

# You can also change the pred_tracks to tracks from any other methods
# e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
```

从代码里看，VGGT在这里负责输出相机位姿和像素点深度信息，VGGSfM负责寻找每个输入图片上提取到的像素点在其他图片上的像素位置。
所以这其实是相机位姿+深度估计+点追踪三个独立过程拼起来的。
因此，VGGT + Colmap的正确实现应该是推断一次VGGT然后用其输出的特征推断一次[`DPTHead`](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)得到相机位姿和像素点深度信息，再推断一次[`TrackHead`](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/track_head.py)得到点追踪结果。
但是代码里没写出来这个，运行[`DPTHead`](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)+[`TrackHead`](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/track_head.py)要推断两次VGGT，可能这就是作者说的efficiency问题吧？