# D4RT 原理解析

![](i/D4RT.png)

[D4RT](https://d4rt-paper.github.io/) 是一个用统一查询接口完成动态场景 4D 重建和跟踪的前馈视频模型。本文解析的是 [`Open-d4rt/src/model`](https://github.com/Lijiaxin0111/Open-d4rt/tree/bead824a40f2d719246469aed3acad101287a201/src/model) 中的 PyTorch 实现。

如果只看模块，OpenD4RT 由一个视频编码器、一个查询编码器、一个独立查询解码器和一组任务 Head 构成：

```python
self.encoder = VideoPatchTransformerEncoder(...)
self.memory_proj = nn.Identity() if dec_hidden == enc_hidden else nn.Linear(enc_hidden, dec_hidden)
self.query_embedder = QueryEmbedder(...)
self.decoder = IndependentQueryDecoder(...)
self.heads = D4RTHeads(hidden_dim=dec_hidden)
```

相比于 [VGGT](VGGT.md) D4RT 最大的区别是换了一种输出范式：

- [VGGT](VGGT.md) 把整组图像一次性解码为相机、深度、点云和轨迹等**稠密**结果。
- D4RT 先把整段视频编码为一个可复用的 **scene memory**，再用查询
  $q=(u,v,t_{src},t_{tgt},t_{cam})$ 去询问这个场景。

一个查询的含义是：

> 位于源帧 $t_{src}$ 的像素 $(u,v)$ 所对应的场景点，在目标时刻 $t_{tgt}$ 位于哪里；并把它的三维结果表示在时刻 $t_{cam}$ 的相机坐标系中。

这里 $u,v$ 是归一化到 $[0,1]$ 的坐标，$u$ 对应图像宽度方向，$v$ 对应高度方向。三个时间变量承担不同职责：

| 字段 | 含义 |
| --- | --- |
| `t_src` | 从哪一帧、哪个像素定义要跟踪的场景点 |
| `t_tgt` | 想知道这个点在哪一个目标时刻的状态 |
| `t_cam` | 三维坐标、位移和法线用哪一帧的相机坐标系表达 |

模型对每个查询输出 13 个标量：

| 输出 | 维度 | 含义 |
| --- | ---: | --- |
| `xyz_3d` | 3 | 目标点在 `t_cam` 相机坐标系中的三维位置 |
| `uv_2d` | 2 | 目标点在 `t_tgt` 图像中的归一化像素坐标 |
| `visibility` | 1 | 目标点在 `t_tgt` 中的可见性 logit |
| `displacement` | 3 | 从 `t_src` 到 `t_tgt`、表达在 `t_cam` 中的三维位移 |
| `normal` | 3 | 目标点的表面法线 |
| `confidence` | 1 | 三维预测的置信度 logit |

模型的顶层 [`forward`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/model/d4rt.py#L366-L371) 非常简洁：

```python
def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    video = batch["video"]
    if video.ndim != 5:
        raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")
    memory = self.encode_video(video=video, aspect_ratio=batch.get("aspect_ratio"))
    return self.decode_queries(video=video, query=batch["query"], memory=memory)
```

输入和输出的 shape 可以概括成：

```text
video:               [B, T, 3, H, W]
query 中每个字段:     [B, M]
scene memory:        [B, N(+1), C_dec]
decoded query token: [B, M, C_dec]
输出:                 [B, M, output_dim]
```

其中 $M$ 是查询数量，$N$ 是视频 patch token 的数量。下面沿着这条数据流逐步分析。

## D4RT 的核心：视频只编码一次，查询可以分批解码

`D4RTModel` 特意把推断拆成了两个公开函数：

```python
def encode_video(self, video, aspect_ratio=None):
    extra_tokens = self._project_aspect_ratio_token(video=video, aspect_ratio=aspect_ratio)
    return self.memory_proj(self.encoder(video, extra_tokens=extra_tokens))

def decode_queries(self, video, query, memory):
    query_tokens = self.query_embedder(video=video, ...)
    decoded = self.decoder(query_tokens, memory)
    return self.heads(decoded)
```

因此一次视频推断可以写成：

```python
memory = model.encode_video(video, aspect_ratio)

for query_chunk in query_chunks:
    pred_chunk = model.decode_queries(video, query_chunk, memory)
```

最昂贵的视频 self-attention 只执行一次。后续无论询问一百个稀疏轨迹点，还是逐像素询问整幅图，都复用同一个 `memory`。

这也是 `IndependentQueryDecoder` 中刻意删除 query self-attention 的原因：不同查询之间不通信，一个 query chunk 的结果不会依赖同批次中还放了哪些其他查询，所以可以按照显存大小自由分块。

---

## [`VideoPatchTransformerEncoder`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/model/encoder.py)：把视频压成 scene memory

从**可训练参数的来源**来说，`VideoPatchTransformerEncoder` 可以概括为：**本质上是一个使用预训练 VideoMAE2 ViT-g 参数初始化、再由 D4RT 端到端 fine-tune 的视频 encoder**。

当前 encoder 的全部可训练参数类型都在加载函数中有对应的 VideoMAE2 映射：

```text
patch_embed:                         2 个参数张量
40 × Transformer block:     40 × 12 = 480 个参数张量
final_norm:                          2 个参数张量
---------------------------------------------------
合计：                                     484 个参数张量
```

每个 block 的 LayerNorm、QKV、attention output projection 和两层 MLP 都有映射；`q_bias`、`v_bias` 也会被重新组合成 PyTorch `MultiheadAttention` 的 QKV bias。因此只要使用的是代码所预期的 `vit_g_hybrid_pt_1200e.pth` checkpoint，这 484 个 encoder 参数张量都应当来自 VideoMAE2，而不是保留随机初始化。

不过，这里的“本质上是预训练 VideoMAE2”是指**参数初始化来源**，不代表当前 forward 与原版 VideoMAE2 逐项完全相同。它没有 import 或直接实例化 VideoMAE2 的原始模型类，而是用 PyTorch 原生模块重新搭建：

```python
self.patch_embed = nn.Conv3d(...)
self.blocks = nn.ModuleList([SelfAttentionBlock(...) for _ in range(num_layers)])
self.final_norm = nn.LayerNorm(hidden_dim)
```

默认配置与 VideoMAE2 ViT-g 对齐的部分包括：

```text
hidden dim = 1408
num heads  = 16
num layers = 40
MLP dim    = 6144，即 mlp_ratio = 6144 / 1408
```

参数之外仍有以下 forward 差异：

| 项目 | 原版 VideoMAE2 ViT-g | 当前 `VideoPatchTransformerEncoder` |
| --- | --- | --- |
| 空间 patch size | 14×14 | 16×16 |
| patch kernel | 原始 14×14 权重 | 将 VideoMAE2 kernel 三线性插值为 16×16 |
| attention 范围 | 每个 block 都做 joint space-time attention | 40 层按 local/global 交替执行 |
| local 层 | 无这种单独重排 | 每个时间 patch 内独立做空间 attention |
| token 数控制 | 固定预训练输入规格 | 超过 `max_tokens` 时自适应空间池化 |
| 附加 token | 原版 encoder 没有 D4RT 的宽高比查询条件 | global 层加入 `aspect_ratio_token` |
| LayerNorm | 官方 ViT-g 配置使用 `eps=1e-6` | `nn.LayerNorm` 默认 `eps=1e-5` |
| Dropout | 由 VideoMAE2 原配置决定 | attention 和 MLP 固定使用 0.1 |

其中最重要的不是模块换了名字，而是 **VideoMAE2 的同一组 attention/MLP 权重在 D4RT 中被放进了不同的 token 交互模式**：偶数层只处理各时间 patch 内部，奇数层才处理整个视频。[官方 VideoMAE2 ViT-g 定义](https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_pretrain.py)使用 14×14 patch、1408 hidden dim、40 层和 16 个 head；其[补充材料](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_VideoMAE_V2_Scaling_CVPR_2023_supplemental.pdf)说明 backbone 是使用 joint space-time attention 的 vanilla ViT。当前实现则用 local/global 交替降低并组织长视频 attention 的计算。

因此本文后面使用下面这句话：

> `VideoPatchTransformerEncoder` 在**参数来源上本质是预训练 VideoMAE2 ViT-g encoder**，但经过了 patch-size、attention pattern 和场景条件 token 等 D4RT 化改造，并不是原版 VideoMAE2 forward 的原封不动复用。

### VideoMAE2 参数在哪里加载

第一层加载发生在 [`D4RTModel.__init__`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/model/d4rt.py#L169-L229) 的末尾：

```python
self._load_pretrained_encoder_weights(encoder_cfg.get("pretrained", None))
```

[`D4RTModel._load_pretrained_encoder_weights`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/model/d4rt.py#L286-L364) 读取配置中的 VideoMAE2 checkpoint，再调用 [`_structured_pretrained_tensor`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/model/d4rt.py#L107-L166) 把 VideoMAE2 参数名和 shape 转换为当前 encoder 的格式，最后只加载到 `self.encoder`：

```python
payload = torch.load(path, map_location="cpu")
src_state = _unpack_state_dict(payload)

# 逐项建立 matched: 当前 encoder 参数名 -> VideoMAE2 tensor
...

self.encoder.load_state_dict(matched, strict=False)
```

对应的模型配置是：

```yaml
encoder:
  variant: vit-g
  pretrained:
    enabled: true
    type: videomae_v2
    path: checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth
    strict: true
    must_succeed: true
```

训练脚本还会用 `VIDEOMAE2_CKPT` 覆盖上述路径：

```bash
CONFIG_OVERRIDES+=(
  --override "model.encoder.pretrained.path=${VIDEOMAE2_CKPT}"
)
```

因此在 `build_model(...)` 构造 `D4RTModel` 时，encoder 已经先获得 VideoMAE2 权重，而 query embedder、decoder、Head 等 D4RT 专有模块仍使用各自的 PyTorch 默认初始化。

### 48 帧训练实际还有第二层初始化

仓库提供的 48 帧训练脚本不只是从原始 VideoMAE2 开始。它还向 `train.py` 传入一个已经训练过的 32 帧 OpenD4RT checkpoint：

```bash
INIT_CKPT="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"

torchrun ... train.py \
  --init-model "$INIT_CKPT" \
  --init-timestep-embed-resize linear
```

[`train.py`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/train.py#L500-L526) 的加载顺序是：

```python
# ① build_model 内部先加载 VideoMAE2 encoder 权重
model = build_model(model_cfg["model"])

# ② 再加载整个 32 帧 OpenD4RT 模型
payload = load_checkpoint(args.init_model, map_location="cpu")
init_state_dict, ... = _prepare_init_state_dict(model, state_dict, ...)
model.load_state_dict(init_state_dict, strict=False)
```

第二次加载覆盖所有名称和 shape 能匹配的模型参数，其中也包括 encoder。因此对于这个 48 帧训练 recipe，真正进入第一个训练 step 的 encoder 通常是 **32 帧 OpenD4RT 已经 fine-tune 过的 encoder**，而不是刚载入后尚未经过 D4RT 训练的原始 VideoMAE2 encoder。只有 32→48 的三张时间 embedding 表因为长度变化，需要线性插值扩展：

```text
query_embedder.t_src_embed.weight
query_embedder.t_tgt_embed.weight
query_embedder.t_cam_embed.weight
```

如果使用 `--resume`，[`Trainer.resume`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/engine/trainer.py#L534-L554) 还会在更后面恢复完整 model、optimizer、scheduler 和训练步数；此时 resume checkpoint 的状态拥有最终优先级。

所以完整的加载优先级是：

```text
PyTorch 随机初始化
    ↓
VideoMAE2 checkpoint：只初始化 encoder
    ↓
--init-model：初始化整个 OpenD4RT，匹配参数覆盖前面的值
    ↓
--resume：恢复完整训练状态，若提供则优先级最高
```

### Encoder 在 D4RT 训练中冻结吗

**没有冻结，整个 encoder 会和 decoder、query embedder、Head 一起端到端 fine-tune。**

理由有三点：

1. `src/model` 和训练入口中没有对 encoder 调用 `requires_grad_(False)`，也没有把它放进 `torch.no_grad()`；`nn.Conv3d`、attention、MLP 和 LayerNorm 参数默认都是 `requires_grad=True`。
2. [`Trainer.__init__`](https://github.com/Lijiaxin0111/Open-d4rt/blob/bead824a40f2d719246469aed3acad101287a201/src/engine/trainer.py#L239-L245) 直接把 `self.model.parameters()` 全部交给同一个 AdamW：

```python
self.optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=float(lr_cfg.get("peak_lr", 1e-4)),
    weight_decay=float(optim_cfg.get("weight_decay", 0.03)),
)
```

3. 训练循环对总 loss 执行 backward 和 optimizer step：

```python
outputs = self.model(batch)
loss, metrics = self.loss_fn(outputs, batch)
self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
```

梯度会从六个预测 Head 依次反传经过 decoder、scene memory 和 encoder。当前代码也没有为 encoder 单独设置更小的 learning rate；所有可训练参数共用 warmup + cosine schedule。默认 48 帧训练配置的 peak LR 为 `4e-6`，final LR 为 `4e-7`。

这里的“使用 VideoMAE2”指的是 **用 VideoMAE2 预训练参数作为初始化，再使用 D4RT 的几何/跟踪监督继续 fine-tune**；仓库并不会在 D4RT 训练阶段继续优化 VideoMAE2 原本的 masked-video-reconstruction 预训练目标。

### 第 1 步：用 `Conv3d` 同时切分时间和空间 patch

输入视频首先从 `[B,T,C,H,W]` 调整成 PyTorch `Conv3d` 使用的 `[B,C,T,H,W]`：

```python
x = video_b_t_c_h_w.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
x = self.patch_embed(x)
```

`patch_embed` 不是逐帧的二维卷积，而是 kernel 和 stride 相同的三维卷积：

```python
self.patch_embed = nn.Conv3d(
    in_channels=in_channels,
    out_channels=hidden_dim,
    kernel_size=patch_size_t_h_w,
    stride=patch_size_t_h_w,
)
```

默认配置使用：

```yaml
patch_size_t_h_w: [2, 16, 16]
hidden_dim: 1408
```

所以每个 token 不只覆盖一个 $16\times16$ 图像区域，还同时覆盖相邻 2 帧。假设输入为 `[B,48,3,256,256]`，卷积输出就是：

```text
[B, 1408, 24, 16, 16]
```

token 总数为：

$$
N=T'H'W'=\left\lfloor\frac{T}{P_t}\right\rfloor
\left\lfloor\frac{H}{P_h}\right\rfloor
\left\lfloor\frac{W}{P_w}\right\rfloor
=24\times16\times16=6144.
$$

三维 patch embedding 的意义是，局部短时运动从模型的第一层就已经被压入 token，而不是先把每帧独立编码后再建立时间联系。

需要注意，卷积没有 padding。如果 $T,H,W$ 不能整除 patch size，末尾不足一个完整 patch 的部分不会生成 token。

### 第 2 步：`max_tokens` 限制 token 数量

视频分辨率或帧数增大时，全局 attention 的平方复杂度会迅速增长。代码用 `_token_cap` 限制 token 数量：

```python
def _token_cap(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, T', H', W']
    b, c, tp, hp, wp = x.shape
    token_count = tp * hp * wp
    if token_count <= self.max_tokens:
        return x
    scale = math.sqrt(self.max_tokens / float(token_count))
    out_h = max(1, int(round(hp * scale)))
    out_w = max(1, int(round(wp * scale)))
    return F.adaptive_avg_pool3d(x, output_size=(tp, out_h, out_w))
```

它保持时间分辨率 $T'$ 不变，只在空间上做自适应平均池化。缩放比例同时作用于高和宽：

$$
s=\sqrt{\frac{N_{max}}{T'H'W'}},\qquad
H_{out}\approx sH',\quad W_{out}\approx sW'.
$$

48 帧、256×256 的默认配置刚好产生 6144 个 token，与 `max_tokens: 6144` 相等，因此不会池化。32 帧 checkpoint 则产生 $16\times16\times16=4096$ 个 token。

### 第 3 步：展平 token 并添加正弦位置编码

三维特征图随后被展平：

```python
b, c, tp, hp, wp = x.shape
video_tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
token_count = video_tokens.shape[1]
pos = sinusoidal_position_embedding(token_count, self.hidden_dim, video_tokens.device)
video_tokens = video_tokens + pos.unsqueeze(0)
```

位置编码使用标准 Transformer 的正弦余弦形式：

$$
PE(p,2i)=\sin\left(p\cdot10000^{-2i/C}\right),
$$

$$
PE(p,2i+1)=\cos\left(p\cdot10000^{-2i/C}\right).
$$

这里有一个容易误读的实现细节：当前代码并没有分别编码 $(t,h,w)$ 三个坐标，而是对展平后的序列下标 $p\in[0,N-1]$ 使用一维位置编码。由于 `flatten` 的顺序固定，时间和空间位置仍然一一对应到序列位置，但它不等同于显式的三维位置编码。

### 第 4 步：加入 `aspect_ratio_token`

输入图像通常会统一缩放到 256×256，原图的宽高比信息可能因此丢失。配置打开 `use_aspect_ratio_token` 时，原始宽高比 $W/H$ 会经过一个两层 MLP：

```python
self.aspect_ratio_proj = nn.Sequential(
    nn.Linear(1, enc_hidden),
    nn.GELU(),
    nn.Linear(enc_hidden, enc_hidden),
)
```

推断时：

```python
return self.aspect_ratio_proj(aspect_ratio).unsqueeze(1)
```

于是每个视频多出一个 `[B,1,C_enc]` 的 special token。没有传入 `aspect_ratio` 时，代码使用 1，即假设原图是正方形：

```python
if aspect_ratio is None:
    aspect_ratio = torch.ones((video.shape[0], 1), ...)
```

这个 token 不是直接加到每个 patch 上，而是只在全局 attention 层与视频 token 拼接。因此它可以向全局场景表示提供相机成像比例信息，同时不会进入逐时间块的 local attention。

### 第 5 步：帧内 local attention 与全局 attention 交替

每个 `SelfAttentionBlock` 都是标准的 Pre-Norm Transformer block：

```python
def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    q = self.norm_attn(tokens)
    attn_out, _ = self.attn(q, q, q, need_weights=False)
    x = tokens + attn_out
    x = x + self.ff(self.norm_ff(x))
    return x
```

对应公式为：

$$
X'=X+\operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
X''=X'+\operatorname{MLP}(\operatorname{LN}(X')).
$$

默认 `vit-g` 配置有 40 层、16 个 attention head，MLP ratio 为 $6144/1408\approx4.3636$。`attention_pattern` 被规范化为 `interleaved_local_global` 后，各层模式为：

```python
return ["local" if (i % 2 == 0) else "global" for i in range(num_layers)]
```

即：

```text
第 0 层 local
第 1 层 global
第 2 层 local
第 3 层 global
...
```

#### Local attention

local 层把时间维并入 batch 维：

```python
local = video_tokens.reshape(b, tp, spatial_tokens, c)
local = local.reshape(b * tp, spatial_tokens, c)
local = block(local)
video_tokens = local.reshape(b, tp, spatial_tokens, c)
video_tokens = video_tokens.reshape(b, tp * spatial_tokens, c)
```

此时 attention 的 batch size 是 $B\times T'$，序列长度是 $H'W'$。因此每个时间 patch 内部的所有空间 token 可以互相交流，不同时间 patch 之间暂时隔离。

注意这里所谓 “framewise” 严格来说是 **时间 patch 内**：默认 $P_t=2$，一个时间 token 已经融合了两帧。

#### Global attention

global 层直接对 `[B,T'H'W',C]` 做 attention：

```python
if extra_tokens is None:
    video_tokens = block(video_tokens)
else:
    merged = torch.cat([video_tokens, extra_tokens], dim=1)
    merged = block(merged)
    video_tokens = merged[:, :token_count]
    extra_tokens = merged[:, token_count:]
```

这时所有时间和空间 token 都处于同一个序列中，所以任意位置都能和任意时刻的信息交互。如果存在 aspect-ratio token，它也在这些层中参与 self-attention。

local/global 交替的作用可以概括为：

- local 层以较低成本充分建模单个时间片内部的空间结构；
- global 层建立跨时间对应、相机运动和动态物体运动关系；
- aspect-ratio token 在 global 层不断汇总并广播全局成像信息。

最后将视频 token 和 aspect-ratio token 拼接，并做一次 LayerNorm：

```python
if extra_tokens is None:
    encoded = video_tokens
else:
    encoded = torch.cat([video_tokens, extra_tokens], dim=1)
return self.final_norm(encoded)
```

### 第 6 步：把 encoder memory 投影到 decoder 维度

默认 encoder hidden dim 是 1408，decoder hidden dim 是 1280，两者并不相同：

```python
self.memory_proj = (
    nn.Identity()
    if dec_hidden == enc_hidden
    else nn.Linear(enc_hidden, dec_hidden)
)
```

所以默认 48 帧配置最终得到：

```text
encoder 输出: [B, 6145, 1408]  # 6144 video tokens + 1 aspect token
memory 输出:  [B, 6145, 1280]
```

这个 `[B,N+1,1280]` 张量就是后续所有查询共同读取的 scene memory。

### VideoMAE2 预训练权重的参数映射细节

当前 encoder 用的是 PyTorch 原生 `Conv3d`、`MultiheadAttention` 和自定义 block 名称，而 VideoMAE2 checkpoint 的命名与结构不同，因此 `D4RTModel` 做了一层显式映射。

例如：

```python
direct_map = {
    "norm_attn.weight": f"{base}.norm1.weight",
    "attn.in_proj_weight": f"{base}.attn.qkv.weight",
    "attn.out_proj.weight": f"{base}.attn.proj.weight",
    "norm_ff.weight": f"{base}.norm2.weight",
    "ff.0.weight": f"{base}.mlp.fc1.weight",
    "ff.3.weight": f"{base}.mlp.fc2.weight",
}
```

VideoMAE2 分开的 `q_bias` 和 `v_bias` 会被拼成 PyTorch MHA 所需的 `in_proj_bias`，中间的 key bias 用零填充：

```python
k_bias = torch.zeros_like(q_bias)
in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
```

如果预训练 patch kernel 的时空尺寸不同，代码还会用三线性插值调整五维卷积核：

```python
resized = F.interpolate(
    kernel,
    size=(dt, dh, dw),
    mode="trilinear",
    align_corners=False,
)
```

### 一句话总结

`VideoPatchTransformerEncoder` = **3D 卷积切视频 patch** → **限制空间 token 数量** → **加入展平序列位置编码** → **交替执行时间片内 local attention 和跨时空 global attention** → **得到可被所有查询反复读取的全局 scene memory**。

---

## `QueryEmbedder`：把一个 5D 几何问题翻译成 Transformer token

视频 memory 说明“场景里有什么”，query token 则说明“我们到底想问什么”。一个 query token 由三类信息相加而成：

```text
query token = UV Fourier token
            + 三个时间 embedding
            + 源像素附近的局部 RGB patch token
```

对应代码：

```python
uv_token = self.uv_proj(self.uv_encoder(uv))
time_token = (
    self.t_src_embed(t_src)
    + self.t_tgt_embed(t_tgt)
    + self.t_cam_embed(t_cam)
)

token = uv_token + time_token
if self.patch_proj is not None:
    patches = self._extract_local_patches(video, u, v, t_src)
    token = token + self.patch_proj(patches)

return self.out_norm(token)
```

### `u,v` 的 Fourier Features

直接把两个坐标送进线性层，很难高效表达高频空间变化。`FourierFeatures` 为每个坐标使用指数增长的频率：

```python
self.frequencies = 2.0 ** torch.arange(num_bands) * torch.pi
```

对于坐标 $x\in\{u,v\}$ 和频带 $k$：

$$
\gamma_k(x)=\left[\sin(2^k\pi x),\cos(2^k\pi x)\right].
$$

实现还保留原始的 $u,v$：

```python
x = uv.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
sin = torch.sin(x)
cos = torch.cos(x)
out = [uv.unsqueeze(-1), sin, cos]
return torch.cat(out, dim=-1).flatten(start_dim=2)
```

默认 `num_bands=8`，因此输出维数为：

$$
2+2\times8\times2=34.
$$

随后用 `Linear(34,1280)` 投影到 decoder token 空间。

Fourier Features 让网络能同时区分大尺度位置与细微坐标差异：低频项表达“在图像左边还是右边”，高频项表达“相邻像素之间的区别”。

### 三个时间使用三套独立 Embedding

代码没有把 `t_src,t_tgt,t_cam` 当三个普通数字拼接，而是分别使用三张可学习的 embedding 表：

```python
self.t_src_embed = nn.Embedding(clip_frames, hidden_dim)
self.t_tgt_embed = nn.Embedding(clip_frames, hidden_dim)
self.t_cam_embed = nn.Embedding(clip_frames, hidden_dim)
```

三者最后相加：

```python
time_token = self.t_src_embed(t_src) \
           + self.t_tgt_embed(t_tgt) \
           + self.t_cam_embed(t_cam)
```

使用独立参数非常重要，因为同一个整数在三个字段中的语义完全不同：

- `t_src=5` 表示从第 5 帧定义场景点；
- `t_tgt=5` 表示询问这个点第 5 帧的状态；
- `t_cam=5` 表示要求输出在第 5 帧相机坐标系中表达。

时间下标会先被截断到 checkpoint 支持的范围：

```python
def _clamp_t(self, t):
    return t.clamp(min=0, max=self.max_frames - 1)
```

因此 48 帧模型的合法时间 embedding 是 0–47，32 帧模型则是 0–31。超出范围不会报错，而是全部映射到最后一个 embedding；正常调用方应当在构造 query 时保证索引有效。

### 源帧局部 RGB patch

只给 $(u,v)$ 和时间下标，query token 知道“从哪里出发”，却不知道“那里长什么样”。因此默认配置还会在 `t_src` 帧的 $(u,v)$ 周围采样一个 9×9 RGB patch。

先根据每个 query 取出对应源帧：

```python
batch_idx = torch.arange(bsz, device=video.device).view(-1, 1).expand(-1, num_queries)
src_frames = video[batch_idx, t_src]  # [B, M, C, H, W]
src_frames = src_frames.reshape(-1, channels, height, width)
```

然后构造以 query 坐标为中心的采样网格：

```python
p = self.local_patch_size
offsets = torch.linspace(-(p - 1) / 2.0, (p - 1) / 2.0, p)
dx = offsets * (2.0 / max(1, width - 1))
dy = offsets * (2.0 / max(1, height - 1))
grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")
base_grid = torch.stack([grid_x, grid_y], dim=-1)
```

`grid_sample` 使用 $[-1,1]$ 坐标，所以中心坐标需要变换：

```python
centers_x = u * 2.0 - 1.0
centers_y = v * 2.0 - 1.0
```

最后双线性采样：

```python
patches = F.grid_sample(
    src_frames,
    grid,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
```

边界外使用 `padding_mode="border"`，即复制最近的边界像素，而不是补零。9×9 RGB patch 被展平成 $3\times9\times9=243$ 维向量，再经过两层 MLP 投影到 1280 维：

```python
self.patch_proj = nn.Sequential(
    nn.Linear(243, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

这块局部外观信息相当于告诉 decoder：“请在 scene memory 中寻找和这个局部 RGB 模式对应的场景点”。它在纹理相似、遮挡和大位移情况下，是对纯坐标查询的重要补充。

### 为什么三种 token 用加法而不是拼接

UV、时间和局部外观都被投影到相同的 1280 维语义空间，然后逐元素相加并 LayerNorm：

$$
q=\operatorname{LN}\left(E_{uv}+E_{time}+E_{rgb}\right).
$$

这样每个查询始终只对应一个 token `[B,M,1280]`，不会因为增加字段而增加 query 序列长度。代价是各种信息必须在相同通道空间里叠加，网络需要自己学习如何把它们分解使用。

### 一句话总结

`QueryEmbedder` = **用 Fourier Features 描述源像素位置** + **用三套独立 embedding 描述源时间、目标时间和相机坐标系** + **用局部 RGB patch 描述要找的点长什么样**。

---

## `IndependentQueryDecoder`：每个查询独立读取全局场景

`IndependentQueryDecoder` 由 8 个结构相同的 cross-attention block 构成：

```python
self.blocks = nn.ModuleList(
    [CrossAttentionBlock(...) for _ in range(num_layers)]
)
```

每个 block 的 forward 是：

```python
def forward(self, query_tokens, memory_tokens):
    q = self.norm_q(query_tokens)
    kv = self.norm_kv(memory_tokens)
    attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
    x = query_tokens + attn_out
    x = x + self.ff(self.norm_ff(x))
    return x
```

其中：

```text
Q = query token: [B, M, 1280]
K = scene memory: [B, N+1, 1280]
V = scene memory: [B, N+1, 1280]
```

cross-attention 计算：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

直观地说，每个 query token 都拿自己的问题去和视频的所有时空 patch 做匹配，然后按相关性加权读取整段视频的信息。

例如，对于一个固定的源像素，`t_tgt` 从 0 变到 47 时，query token 会发生变化。不同目标时间的 query 会在同一个 scene memory 中关注不同的时空位置，从而得到这个场景点的完整轨迹。

### 为什么叫 `IndependentQueryDecoder`

普通 Transformer decoder 往往包含两种 attention：

1. query 之间的 self-attention；
2. query 对 encoder memory 的 cross-attention。

这里明确只有第 2 种：

```python
class CrossAttentionBlock(nn.Module):
    """Decoder block with cross-attention only (no query self-attention)."""
```

虽然 `query_tokens` 的 shape 是 `[B,M,C]`，但在 cross-attention 中，每个 query 的 softmax 都只沿 memory token 维度计算。第 $i$ 个 query 的输出可写为：

$$
y_i=f(q_i,\mathcal M),
$$

而不是：

$$
y_i=f(q_1,q_2,\ldots,q_M,\mathcal M).
$$

block 后面的 MLP 也是逐 token 运算，因此不同查询从头到尾都不交换信息。

这带来三个直接结果：

- **可分块**：一次解码 4096 个 query 与分四次各解码 1024 个 query，eval 模式下结果应相同；
- **查询数可变**：稀疏跟踪只放少量 query，稠密重建可以放几十万 query；
- **计算线性增长**：decoder 对 query 数量 $M$ 的复杂度主要是 $O(MN)$，没有 query self-attention 的 $O(M^2)$ 项。

### 残差与 MLP

每层 cross-attention 后都有残差连接和前馈网络：

$$
Q'=Q+\operatorname{CrossAttn}(\operatorname{LN}(Q),\operatorname{LN}(M)),
$$

$$
Q''=Q'+\operatorname{MLP}(\operatorname{LN}(Q')).
$$

默认 decoder 有 8 层、16 个 head、hidden dim 1280、MLP ratio 3.5，因此 FFN 中间维数是：

$$
1280\times3.5=4480.
$$

8 层反复读取同一份 scene memory：前几层可能建立粗略的时空匹配，后几层再逐步把读取结果变成适合几何回归的 query feature。最后再做一次 LayerNorm：

```python
for block in self.blocks:
    x = block(x, memory_tokens)
return self.out_norm(x)
```

### 一句话总结

`IndependentQueryDecoder` = **让每个 query 独立地对整段视频 memory 做 8 轮 cross-attention**；查询之间不做 self-attention，因此同一份场景表示可以被任意数量、任意分块方式的查询读取。

---

## `D4RTHeads`：一个 query feature 同时回答六类问题

decoder 输出 `[B,M,1280]` 后，没有复杂的上采样或迭代细化模块，而是直接接 6 个并行线性层：

```python
self.xyz_head = nn.Linear(hidden_dim, 3)
self.uv_head = nn.Linear(hidden_dim, 2)
self.visibility_head = nn.Linear(hidden_dim, 1)
self.displacement_head = nn.Linear(hidden_dim, 3)
self.normal_head = nn.Linear(hidden_dim, 3)
self.confidence_head = nn.Linear(hidden_dim, 1)
```

forward 为：

```python
return {
    "xyz_3d": self.xyz_head(decoded_queries),
    "uv_2d": self.uv_head(decoded_queries),
    "visibility": self.visibility_head(decoded_queries).squeeze(-1),
    "displacement": self.displacement_head(decoded_queries),
    "normal": self.normal_head(decoded_queries),
    "confidence": self.confidence_head(decoded_queries).squeeze(-1),
}
```

### `xyz_3d`

`xyz_3d` 是 D4RT 查询接口最核心的输出。它不是固定 world coordinate 中的点，而是目标时刻 $t_{tgt}$ 的场景点，在 $t_{cam}$ 相机坐标系中的位置：

$$
\mathbf p^{(t_{cam})}_{t_{tgt}}=(x,y,z).
$$

把 `t_cam` 设为不同值，就能要求同一个目标点在不同相机坐标系中表达。这个设计把动态物体运动和相机运动统一进同一个函数中。

### `uv_2d`

`uv_2d` 预测目标点在 `t_tgt` 图像中的归一化坐标。固定 `(u,v,t_src)`，枚举不同 `t_tgt`，就得到二维像素轨迹：

$$
\left\{(u_{t},v_{t})\mid t=0,\ldots,T-1\right\}.
$$

当前 Head 没有 sigmoid，所以输出在网络层面不被强制限制到 $[0,1]$。训练目标位于 $[0,1]$，模型通过 loss 学习这个范围；调用端不能因为它叫 UV 就假设代码已经自动 clamp。

### `visibility`

`visibility` 输出 raw logit。训练时使用：

```python
F.binary_cross_entropy_with_logits(
    outputs["visibility"],
    target["visibility"],
)
```

推断时需要显式转换：

$$
p_{vis}=\sigma(l_{vis})=\frac{1}{1+e^{-l_{vis}}}.
$$

例如仓库推断代码用 `sigmoid(logit) > 0.5` 得到布尔可见性。

### `displacement`

`displacement` 表示场景点从 `t_src` 到 `t_tgt` 的三维运动，并在 `t_cam` 坐标系中表达。它把“点在哪里”和“点移动了多少”作为两个相关但独立的监督任务。

### `normal`

Head 输出未经归一化的 3D 向量，单位化发生在 normal loss 中：

```python
pred = F.normalize(outputs["normal"], dim=-1)
gt = F.normalize(target["normal"], dim=-1)
cos = F.cosine_similarity(pred, gt, dim=-1)
```

所以推断时若需要单位法线，也应执行 `F.normalize(pred["normal"], dim=-1)`。

### `confidence`

`confidence` 同样是 raw logit，loss 中通过 sigmoid 得到 $c\in(0,1)$：

```python
c = torch.sigmoid(outputs["confidence"]).clamp(1e-4, 1.0 - 1e-4)
```

它主要用于衡量或加权 `xyz_3d` 的可靠性，不是 `visibility` 的重复：一个点可以在画面中可见，但三维坐标预测仍不够可靠。

### Head 配置与实际构造

配置文件中写有：

```yaml
heads:
  output_dim_total: 13
  xyz_3d: 3
  uv_2d: 2
  visibility: 1
  displacement: 3
  normal: 3
  confidence: 1
```

但当前 `D4RTModel` 构造 `D4RTHeads` 时只传入 `hidden_dim`，并不会读取这些开关或维数。也就是说六个 Head 在当前实现中总会全部创建，13 维只是它们输出维数之和，而不是一个真正的单层 `Linear(C,13)`。

### 一句话总结

`D4RTHeads` 用同一个 query feature 并行回归 **三维位置、二维位置、可见性、三维位移、表面法线和置信度**。Head 本身非常轻，真正的几何推理已经在 encoder memory 和 decoder cross-attention 中完成。

---

## 48 帧默认配置的完整 shape 流

假设：

```text
B = batch size
T = 48
H = W = 256
M = 每个样本的 query 数量
encoder dim = 1408
decoder dim = 1280
patch size = (2,16,16)
```

整个模型的张量变化如下：

```text
video
[B,48,3,256,256]
    │ permute
    ▼
[B,3,48,256,256]
    │ Conv3d(kernel=stride=(2,16,16))
    ▼
[B,1408,24,16,16]
    │ flatten spatial-temporal dimensions
    ▼
[B,6144,1408] video tokens
    │ 40 层 local/global self-attention
    │ + [B,1,1408] aspect-ratio token
    ▼
[B,6145,1408]
    │ Linear(1408,1280)
    ▼
[B,6145,1280] scene memory

u,v,t_src,t_tgt,t_cam: [B,M]
    │ Fourier UV + time embeddings + 9×9 RGB patch
    ▼
[B,M,1280] query tokens
    │ 8 层 cross-attention，K/V 均为 scene memory
    ▼
[B,M,1280] decoded query features
    │ 6 个并行 Linear Head
    ▼
xyz_3d:      [B,M,3]
uv_2d:       [B,M,2]
visibility:  [B,M]
displacement:[B,M,3]
normal:      [B,M,3]
confidence:  [B,M]
```

这个 shape 流也揭示了主要显存来源：

- encoder global self-attention 与 $N^2$ 成正比；
- decoder cross-attention 与 $MN$ 成正比；
- 删除 query self-attention 后，不再有 $M^2$ 项；
- scene memory 可复用，所以增大 query 数量时可以只分块 decoder。

---

## 同一个查询接口如何实现不同任务

D4RT 没有为深度、跟踪、动态重建分别设计三套完全不同的网络。它主要通过改变 query 的组合方式完成任务。

### 单帧深度或点图

在一帧上建立像素网格，对每个像素设置：

```text
t_src = t_tgt = t_cam = t
```

此时 `xyz_3d[...,2]` 就是相机前向轴方向的深度，整个 `xyz_3d` 则是该帧相机坐标系中的稠密 point map。

### 二维和三维点跟踪

固定一个源点：

```text
(u, v, t_src) 固定
t_tgt = 0, 1, 2, ..., T-1
```

读取所有目标时刻的 `uv_2d` 就得到二维轨迹；读取 `xyz_3d` 得到三维轨迹；`visibility` 判断各时刻是否可见。

如果令：

```text
t_cam = t_tgt
```

每个三维点都在目标帧自身的相机坐标系中表达，适合局部几何或深度预测。

如果令：

```text
t_cam = 0
```

所有时刻都在第 0 帧相机坐标系中表达，便于直接形成统一参考系下的 3D 轨迹。

### 4D 场景重建

对许多源像素以及所有目标时刻组合 query：

```text
空间维：枚举大量 (u,v)
时间维：枚举 t_tgt
参考系：固定 t_cam，或令 t_cam=t_tgt
```

模型输出的 `xyz_3d` 就构成随时间变化的三维点集合。再结合 `visibility`、`confidence` 和 `normal`，可以过滤不可见或低置信度点并进行渲染。

### 相机参数并不是一个直接 Head

与 [VGGT](VGGT.md) 的 `CameraHead` 不同，当前 `src/model` 中没有显式输出内参或外参的 Head。配置给出的几何解码策略是从大量三维查询结果反推相机：

```yaml
geometry_decoding:
  extrinsics_from_queries:
    method: umeyama
  intrinsics_from_queries:
    method: median_from_fx_fy_estimates
```

外参可以通过不同 `t_cam` 下的对应三维点，用 Umeyama 对齐求得；内参则可以利用 point map 与像素坐标的投影关系估计。例如在归一化、主点位于 0.5 的约定下，配置写为：

$$
f_x=\frac{p_z(u-0.5)}{p_x},\qquad
f_y=\frac{p_z(v-0.5)}{p_y}.
$$

因此 D4RT 的相机估计也是“查询几何 → 后处理求相机”，而不是一个专门的相机回归 token。

---

## 几个容易忽略的实现细节

### 1. `local` 不是单 RGB 帧，而是单时间 patch

默认时间 patch size 是 2。`local attention` 的每一个时间单元已经融合相邻两帧，所以更准确的说法是“时间 patch 内的空间 attention”。

### 2. scene memory 中包含 aspect-ratio token

aspect token 不只是 encoder 的临时条件。最后它会与视频 token 一起返回，decoder 的每个 query 都可以直接 cross-attend 到它。

### 3. 位置编码是展平后的 1D 编码

代码使用序列长度调用 `sinusoidal_position_embedding(token_count, dim)`，没有显式构造 $(t,h,w)$ 三轴编码。分析实现时不能把配置或论文中的“时空 token”自动等同于“三维位置编码”。

### 4. 输出 Head 基本都没有激活

`visibility` 和 `confidence` 是 logits，`normal` 未归一化，`uv_2d` 未 sigmoid/clamp，`xyz_3d` 和 `displacement` 也是无界线性输出。激活和规范化发生在 loss 或推断调用端。

### 5. query 的独立性成立于 eval 模式

结构上 query 不互相 attention；但训练时 block 中的 dropout 会为不同调用采样不同随机掩码，所以把同一批 query 拆成多次训练 forward，数值不会逐位一致。推断时 `model.eval()` 关闭 dropout 后，分块才具有确定性的等价关系。

### 6. 时间 embedding 有固定上限

时间索引由 `clip_frames` 决定。代码会 clamp 越界索引而不是报错，因此使用 32 帧 checkpoint 查询第 40 帧不会产生新的“第 40 帧语义”，而会复用第 31 帧 embedding。

### 7. `_token_cap` 只缩小空间维

它不减少 $T'$，这有利于保留时间分辨率，但也意味着极长视频即使空间降到 1×1，token 数仍至少为 $T'$。此外输出高宽经过 round，最终 token 数是近似受限，而非数学上严格保证不超过 `max_tokens`。

### 8. 配置项并非都直接控制代码分支

例如 `heads` 中的维数、`decoder.query_self_attention: false` 等字段描述了模型规格，但当前构造函数并没有逐项读取它们。真正决定行为的是 Python 实现：decoder 始终没有 self-attention，六个 Head 也始终全部存在。

---

## 总结

D4RT 的核心不是某个特殊的几何公式，而是把重建与跟踪统一成一个条件函数：

$$
F\left(V,u,v,t_{src},t_{tgt},t_{cam}\right)
\rightarrow
\left(\mathbf p_{3D},\mathbf p_{2D},v,\Delta\mathbf p,\mathbf n,c\right).
$$

它的完整工作流可以压缩为：

1. 用 3D patch embedding 和 local/global Transformer 把整段视频编码为 scene memory；
2. 用 UV Fourier Features、三个时间 embedding 和源帧局部 RGB patch 构造 query token；
3. 让每个 query 独立地 cross-attend 到整段视频 memory；
4. 用轻量线性 Head 同时输出三维位置、二维对应、可见性、位移、法线和置信度；
5. 通过枚举不同的空间坐标、目标时间和相机参考系，把同一个模型接口扩展为深度估计、点跟踪、4D 重建和相机恢复。

如果说 [VGGT](VGGT.md) 的思路是“看完所有图片后，一次性把几何结果全部画出来”，那么 D4RT 的思路就是“先把整段视频理解成一个可查询的 4D 数据库，再针对任意时空问题逐条作答”。
