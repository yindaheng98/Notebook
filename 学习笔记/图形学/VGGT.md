# VGGT 原理解析

![](./i/vggt.png)

原版[VGGT](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba)由一个骨干网络和4个Head构成：

```python
self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
```

其运行时输入为一批图片和一批用于track的查询点，输出为相机位置编码和深度+点云及其置信度，如果有track查询点输入还会输出查询点在每张图片上的像素位置、可见性及置信度：

```python
def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
    """
    Forward pass of the VGGT model.

    Args:
        images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
            B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
            Shape: [N, 2] or [B, N, 2], where N is the number of query points.
            Default: None

    Returns:
        dict: A dictionary containing the following predictions:
            - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
            - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
            - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
            - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
            - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
            - images (torch.Tensor): Original input images, preserved for visualization

            If query_points is provided, also includes:
            - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
            - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
            - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
    """        
    # If without batch dimension, add it
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
        
    if query_points is not None and len(query_points.shape) == 2:
        query_points = query_points.unsqueeze(0)

    aggregated_tokens_list, patch_start_idx = self.aggregator(images)

    predictions = {}

    with torch.cuda.amp.autocast(enabled=False):
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            predictions["pose_enc_list"] = pose_enc_list
            
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

    if self.track_head is not None and query_points is not None:
        track_list, vis, conf = self.track_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
        )
        predictions["track"] = track_list[-1]  # track of the last iteration
        predictions["vis"] = vis
        predictions["conf"] = conf

    if not self.training:
        predictions["images"] = images  # store the images for visualization during inference

    return predictions
```

VGGT的结构决定了它只能输入固定尺寸的图片。
原版VGGT参数输入尺寸为518x518，patch_size为14x14。
在原版代码中，对于任意尺寸的图片，在输入VGGT之前是先由[`load_and_preprocess_images_square`](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/utils/load_fn.py#L13)padding为正方形再缩放到1024x1024，再在[`run_VGGT`](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L72)里缩放到518x518。

## `Aggregator` 结构解析

### `patch_embed`

图片输入先经过一个`patch_embed`：

```python
B, S, C_in, H, W = images.shape

if C_in != 3:
    raise ValueError(f"Expected 3 input channels, got {C_in}")

# Normalize images and reshape for patch embed
images = (images - self._resnet_mean) / self._resnet_std

# Reshape to [B*S, C, H, W] for patch embedding
images = images.view(B * S, C_in, H, W)
patch_tokens = self.patch_embed(images)

if isinstance(patch_tokens, dict):
    patch_tokens = patch_tokens["x_norm_patchtokens"]
```

这个`patch_embed`就是DINOv2或者卷积：

```python
if "conv" in patch_embed:
    self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
else:
    vit_models = {
        "dinov2_vitl14_reg": vit_large,
        "dinov2_vitb14_reg": vit_base,
        "dinov2_vits14_reg": vit_small,
        "dinov2_vitg2_reg": vit_giant2,
    }

    self.patch_embed = vit_models[patch_embed](
        img_size=img_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        block_chunks=block_chunks,
        init_values=init_values,
    )

    # Disable gradient updates for mask token
    if hasattr(self.patch_embed, "mask_token"):
        self.patch_embed.mask_token.requires_grad_(False)
```

### `camera_token`和`register_token`

再拿两个`camera_token`和`register_token`给拼在`patch_embed`后面：

```python
_, P, C = patch_tokens.shape

# Expand camera and register tokens to match batch size and sequence length
camera_token = slice_expand_and_flatten(self.camera_token, B, S)
register_token = slice_expand_and_flatten(self.register_token, B, S)

# Concatenate special tokens with patch tokens
tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
```

这里的`slice_expand_and_flatten`就是根据`patch_embed`的尺寸把`token_tensor`复制几遍：

```python
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
```

这两个`camera_token`和`register_token`就是两个可训练的`nn.Parameter`：

```python
# Note: We have two camera tokens, one for the first frame and one for the rest
# The same applies for register tokens
self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
```

并且经过`normal_`初始化参数：
```python
# Initialize parameters with small values
nn.init.normal_(self.camera_token, std=1e-6)
nn.init.normal_(self.register_token, std=1e-6)
```

所以，这两个`camera_token`和`register_token`就是把可训练的两个token拼接在图片的`patch_embed`后面输入给transformer，每张图片后面都拼了一个相同的token。

### patch位置信息

接下来获取patch的位置信息：

```python
pos = None
if self.rope is not None:
    pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

if self.patch_start_idx > 0:
    # do not use position embedding for special tokens (camera and register tokens)
    # so set pos to 0 for the special tokens
    pos = pos + 1
    pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
    pos = torch.cat([pos_special, pos], dim=1)
```

其实就是每个patch在图像上的坐标：

```python
class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
```

### Attention计算

经过多个Attention模块，每个Attention模块里都有几个全局attention和帧内attention子模块，根据`aa_order`决定是全局attention还是帧内attention，最后`output_list`输出attention后的所有token：

```python
# update P because we added special tokens
_, P, C = tokens.shape

frame_idx = 0
global_idx = 0
output_list = []

for _ in range(self.aa_block_num):
    for attn_type in self.aa_order:
        if attn_type == "frame":
            tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                tokens, B, S, P, C, frame_idx, pos=pos
            )
        elif attn_type == "global":
            tokens, global_idx, global_intermediates = self._process_global_attention(
                tokens, B, S, P, C, global_idx, pos=pos
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

    for i in range(len(frame_intermediates)):
        # concat frame and global intermediates, [B x S x P x 2C]
        concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
        output_list.append(concat_inter)

del concat_inter
del frame_intermediates
del global_intermediates
return output_list, self.patch_start_idx
```

默认值为一个帧内加一个全局Attention：

```python
aa_order=["frame", "global"]
```

帧内Attention和全局Attention模型结构都一样：

```python
self.frame_blocks = nn.ModuleList(
    [
        block_fn(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=self.rope,
        )
        for _ in range(depth)
    ]
)

self.global_blocks = nn.ModuleList(
    [
        block_fn(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=self.rope,
        )
        for _ in range(depth)
    ]
)
```

区别在于推断时`token`的重排方式不一样：

```python
def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
    """
    Process frame attention blocks. We keep tokens in shape (B*S, P, C).
    """
    # If needed, reshape tokens or positions:
    if tokens.shape != (B * S, P, C):
        tokens = tokens.view(B, S, P, C).view(B * S, P, C)

    if pos is not None and pos.shape != (B * S, P, 2):
        pos = pos.view(B, S, P, 2).view(B * S, P, 2)

    intermediates = []

    # by default, self.aa_block_size=1, which processes one block at a time
    for _ in range(self.aa_block_size):
        if self.training:
            tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
        else:
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
        frame_idx += 1
        intermediates.append(tokens.view(B, S, P, C))

    return tokens, frame_idx, intermediates

def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    """
    Process global attention blocks. We keep tokens in shape (B, S*P, C).
    """
    if tokens.shape != (B, S * P, C):
        tokens = tokens.view(B, S, P, C).view(B, S * P, C)

    if pos is not None and pos.shape != (B, S * P, 2):
        pos = pos.view(B, S, P, 2).view(B, S * P, 2)

    intermediates = []

    # by default, self.aa_block_size=1, which processes one block at a time
    for _ in range(self.aa_block_size):
        if self.training:
            tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
        else:
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
        global_idx += 1
        intermediates.append(tokens.view(B, S, P, C))

    return tokens, global_idx, intermediates
```

注意这两个函数唯一的区别在于开头几行：

```python
def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
    ......
    if tokens.shape != (B * S, P, C):
        tokens = tokens.view(B, S, P, C).view(B * S, P, C)

    if pos is not None and pos.shape != (B * S, P, 2):
        pos = pos.view(B, S, P, 2).view(B * S, P, 2)

    ......

def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    ......
    if tokens.shape != (B, S * P, C):
        tokens = tokens.view(B, S, P, C).view(B, S * P, C)

    if pos is not None and pos.shape != (B, S * P, 2):
        pos = pos.view(B, S, P, 2).view(B, S * P, 2)

    ......
```

帧内Attention的batch维是`B*S`，序列长度是`P`（单帧内部token），因此每个batch是在同一帧内计算attention。
全局Attention的batch 维是`B`，序列长度是`S*P`（所有帧token串起来），因此每个batch是在所有帧的所有token间计算attention。