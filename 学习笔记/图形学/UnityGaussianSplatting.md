# `aras-p/UnityGaussianSplatting` Runtime 源码浅析

>In the game object that has a `GaussianSplatRenderer` script, **point the Asset field to** one of your created assets.

按照 README 所述，主要调用逻辑位于 [package/Runtime/GaussianSplatRenderer.cs](https://github.com/aras-p/UnityGaussianSplatting/blob/2f371e121db7c56159be634545d5bf9c5b2ce55b/package/Runtime/GaussianSplatRenderer.cs) 中，将 `GaussianSplatRenderer.cs` 脚本绑定到任意 Game Object 上并且将3DGS文件设置为其 Assert 即可实现3DGS渲染。

[package/Runtime/GaussianSplatRenderer.cs](https://github.com/aras-p/UnityGaussianSplatting/blob/2f371e121db7c56159be634545d5bf9c5b2ce55b/package/Runtime/GaussianSplatRenderer.cs) 有两个类：`GaussianSplatRenderSystem` 类和其内部的 `GaussianSplatRenderer` 类。
其中，`GaussianSplatRenderSystem` 类为单例类，全局唯一，通过 `GaussianSplatRenderSystem.instance` 进行调用；而每个3DGS对象上都会绑定一个 `GaussianSplatRenderer` 类，`GaussianSplatRenderer` 类通过调用全局唯一的 `GaussianSplatRenderSystem` 单例类的类方法执行渲染操作，从而实现所有3DGS对象中的Gaussians统统合并到一起进行渲染。

## `GaussianSplatRenderer.OnEnable` 中的初始化过程

3DGS渲染初始化过程位于 `GaussianSplatRenderer.OnEnable` 中：

```c#
public void OnEnable()
{
    m_FrameCounter = 0;
    if (!resourcesAreSetUp)
        return;

    EnsureMaterials();
    EnsureSorterAndRegister();

    CreateResourcesForAsset();
}
```

可见其包含三项步骤：`EnsureMaterials` 初始化材质、`EnsureSorterAndRegister` 注册 Gaussians 并初始化排序 `CreateResourcesForAsset` 创建渲染时要用到的相关资源。

`EnsureMaterials` 初始化了一批材质：
```c#
public void EnsureMaterials()
{
    if (m_MatSplats == null && resourcesAreSetUp)
    {
        m_MatSplats = new Material(m_ShaderSplats) {name = "GaussianSplats"};
        m_MatComposite = new Material(m_ShaderComposite) {name = "GaussianClearDstAlpha"};
        m_MatDebugPoints = new Material(m_ShaderDebugPoints) {name = "GaussianDebugPoints"};
        m_MatDebugBoxes = new Material(m_ShaderDebugBoxes) {name = "GaussianDebugBoxes"};
    }
}
```
### `GaussianSplatRenderer.EnsureSorterAndRegister`

`EnsureSorterAndRegister` 初始化了排序用的类并在全局 `GaussianSplatRenderSystem` 单例中注册了3DGS对象：
```c#
    public void EnsureSorterAndRegister()
    {
        if (m_Sorter == null && resourcesAreSetUp)
        {
            m_Sorter = new GpuSorting(m_CSSplatUtilities);
        }

        if (!m_Registered && resourcesAreSetUp)
        {
            GaussianSplatRenderSystem.instance.RegisterSplat(this);
            m_Registered = true;
        }
    }
```

其中，`GpuSorting` 是用于GPU排序的类，定义于 [package/Runtime/GpuSorting.cs](https://github.com/aras-p/UnityGaussianSplatting/blob/2f371e121db7c56159be634545d5bf9c5b2ce55b/package/Runtime/GpuSorting.cs) 中，包含排序时用到的变量和方法，其底层是调用 [package/Shaders/DeviceRadixSort.hlsl](https://github.com/aras-p/UnityGaussianSplatting/blob/2f371e121db7c56159be634545d5bf9c5b2ce55b/package/Shaders/DeviceRadixSort.hlsl) 中定义的几个 Kernel实现GPU并行排序；

### `GaussianSplatRenderSystem.RegisterSplat`

`RegisterSplat` 是 `GaussianSplatRenderSystem` 中用于注册3DGS对象的方法，具体来说是将 `GaussianSplatRenderSystem.OnPreCullCamera` 方法绑定在 `Camera.onPreCull` 事件中，并在`m_Splats`中为当前的 `GaussianSplatRenderer` 新建一块 `MaterialPropertyBlock`：

```c#
public void RegisterSplat(GaussianSplatRenderer r)
{
    if (m_Splats.Count == 0)
    {
        if (GraphicsSettings.currentRenderPipeline == null)
            Camera.onPreCull += OnPreCullCamera;
    }

    m_Splats.Add(r, new MaterialPropertyBlock());
}
```

这个 `OnPreCullCamera` 函数就是3DGS渲染的主程序，绑定到 `Camera.onPreCull` 将令其在摄像机开始裁剪阶段之前被触发，这里的判断条件保证了 `OnPreCullCamera` 只被绑定一次；`MaterialPropertyBlock` 是Unity提供的一种轻量级容器，用于存储材质属性的覆盖值。它允许开发者为单个Renderer实例设置特定的材质属性值，同时保持原始材质的共享引用，维护GPU批处理的优势；`m_Splats` 是 `GaussianSplatRenderSystem` 一个以`GaussianSplatRenderer` 为key，`MaterialPropertyBlock` 为value的`Dictionary`，它通过 `RegisterSplat` 记录了场景中的所有3DGS对象。

### `GaussianSplatRenderer.CreateResourcesForAsset`

`CreateResourcesForAsset` 根据Gaussians的数量 `m_SplatCount = asset.splatCount` 初始化了几个GPU Buffer `m_GpuPosData`、`m_GpuOtherData`、`m_GpuSHData`、`m_GpuColorData`、`m_GpuChunks`，`m_GpuIndexBuffer`，调用从`asset.xxXX.GetData` 读取数据并用 `GetData` 给这些GPU Buffer设置了数据，最后调用 `InitSortBuffers` 初始化了GPU排序用到的各种Buffers：

```c#
void CreateResourcesForAsset()
{
    if (!HasValidAsset)
        return;

    m_SplatCount = asset.splatCount;
    m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
    m_GpuPosData.SetData(asset.posData.GetData<uint>());
    m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
    m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
    m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
    m_GpuSHData.SetData(asset.shData.GetData<uint>());
    var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
    var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
    var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
    tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
    tex.Apply(false, true);
    m_GpuColorData = tex;
    if (asset.chunkData != null && asset.chunkData.dataSize != 0)
    {
        m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
            (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
            UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
        m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
        m_GpuChunksValid = true;
    }
    else
    {
        // just a dummy chunk buffer
        m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
            UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
        m_GpuChunksValid = false;
    }

    m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize);
    m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
    // cube indices, most often we use only the first quad
    m_GpuIndexBuffer.SetData(new ushort[]
    {
        0, 1, 2, 1, 3, 2,
        4, 6, 5, 5, 6, 7,
        0, 2, 4, 4, 2, 6,
        1, 5, 3, 5, 7, 3,
        0, 4, 1, 4, 5, 1,
        2, 3, 6, 3, 7, 6
    });

    InitSortBuffers(splatCount);
}
```

其中用到的`asset` 变量为 `GaussianSplatAsset` 类，`GaussianSplatAsset` 定义于[package/Runtime/GaussianSplatAsset.cs](https://github.com/aras-p/UnityGaussianSplatting/blob/2f371e121db7c56159be634545d5bf9c5b2ce55b/package/Runtime/GaussianSplatAsset.cs)，继承于 `ScriptableObject`。

>`ScriptableObject` 是 Unity 提供的一个数据配置存储基类，是一个可以用来保存大量数据的数据容器，我们可以将它保存为自定义的数据资源文件。
`ScriptableObject` 类的实例会被保存成资源文件（`.asset`文件），和预制体，材质球，音频文件等类似，都是一种资源文件，存放在 `Assets` 文件夹下，创建出来的实例也是唯一存在的。

## `GaussianSplatRenderSystem.OnPreCullCamera` 中的渲染主程序

在 `OnEnable` 中绑定的 `OnPreCullCamera` 函数为3DGS渲染的主程序，核心思想就是使用 Unity 中提供的`CommandBuffer` 对渲染管线进行扩展，在渲染过程中插入3DGS的渲染逻辑：

```c#
void OnPreCullCamera(Camera cam)
{
    if (!GatherSplatsForCamera(cam))
        return;

    InitialClearCmdBuffer(cam);

    m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
    m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
    m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

    // We only need this to determine whether we're rendering into backbuffer or not. However, detection this
    // way only works in BiRP so only do it here.
    m_CommandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture, BuiltinRenderTextureType.CameraTarget);

    // add sorting, view calc and drawing commands for each splat object
    Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

    // compose
    m_CommandBuffer.BeginSample(s_ProfCompose);
    m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
    m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
    m_CommandBuffer.EndSample(s_ProfCompose);
    m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
}
```

### `GaussianSplatRenderSystem.GatherSplatsForCamera`

`GatherSplatsForCamera` 从 `m_Splats` 中取出所有需要渲染的3DGS对象放入 `m_ActiveSplats` 并对其按 `m_RenderOrder` 或其距离相机的远近进行排序：

```c#
public bool GatherSplatsForCamera(Camera cam)
{
    if (cam.cameraType == CameraType.Preview)
        return false;
    // gather all active & valid splat objects
    m_ActiveSplats.Clear();
    foreach (var kvp in m_Splats)
    {
        var gs = kvp.Key;
        if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
            continue;
        m_ActiveSplats.Add((kvp.Key, kvp.Value));
    }
    if (m_ActiveSplats.Count == 0)
        return false;

    // sort them by order and depth from camera
    var camTr = cam.transform;
    m_ActiveSplats.Sort((a, b) =>
    {
        var orderA = a.Item1.m_RenderOrder;
        var orderB = b.Item1.m_RenderOrder;
        if (orderA != orderB)
            return orderB.CompareTo(orderA);
        var trA = a.Item1.transform;
        var trB = b.Item1.transform;
        var posA = camTr.InverseTransformPoint(trA.position);
        var posB = camTr.InverseTransformPoint(trB.position);
        return posA.z.CompareTo(posB.z);
    });

    return true;
}
```

这里的 `m_RenderOrder` 是3DGS对象上的一个可调变量，用于手动控制多个3DGS对象的渲染顺序：

```c#
public int m_RenderOrder;
[Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
```

### `GaussianSplatRenderSystem.InitialClearCmdBuffer`

`InitialClearCmdBuffer` 创建了一个 `CommandBuffer` 对象并将其绑定到相机的 `CameraEvent.BeforeForwardAlpha` 事件上，并通过判断条件保证每个相机只绑定一次：

```c#
public CommandBuffer InitialClearCmdBuffer(Camera cam)
{
    m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
    if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
    {
        cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
        m_CameraCommandBuffersDone.Add(cam);
    }

    // get render target for all splats
    m_CommandBuffer.Clear();
    return m_CommandBuffer;
}
```

其中，`CommandBuffer` ‌是Unity提供的GPU渲染命令列表，可以延迟提交渲染指令，减少CPU/GPU负担，批量执行DrawCall，避免CPU过度调用API，它支持插入自定义渲染指令，如深度处理、后处理效果，并且可以脱离GameObject直接控制渲染，如画辅助线、调试网格等；`CameraEvent.BeforeForwardAlpha` 是在Unity中用于指定摄像机渲染事件的一个枚举值，它表示在渲染透明对象之前执行的操作。

>在Unity渲染管线中，前向渲染路径通常包括以下几个主要阶段：‌不透明物体渲染 ‌-> ‌透明物体渲染‌ -> 后期效果处理‌。`CameraEvent.BeforeForwardAlpha`‌ 允许你在透明物体渲染之前插入自定义的渲染命令或效果。这可以用于在透明物体渲染之前执行一些特殊的处理，比如添加额外的光照效果、修改场景的某些部分等‌。

### `GaussianSplatRenderSystem.OnPreCullCamera` 剩余部分

渲染纹理 (Render Texture) 是一种 Unity 在运行时创建和更新的纹理。你可以在其上绘制，然后像使用其它纹理一样使用。渲染纹理的一个典型应用是镜面的渲染，例如汽车的后视镜就可以贴一个Render Texture，它是从这个镜子所对应视角的摄像机渲染而来。
而3DGS对象的渲染过程就是使用 Render Texture 实现的。

#### 设置渲染纹理

```c#
    m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
    m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
    m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);
```

这里调用的 `GetTemporaryRT` 就是分配一段 Render Texture 内存空间用于渲染 Render Texture。
查阅 Unity 文档中关于 `GetTemporaryRT` 的部分可以知道这里输入的几个参数的含义：
1. `nameID` 为这个 Render Texture 的 ID，`GetTemporaryRT`对相同的 `nameID` 会返回同一段 Render Texture 内存空间，避免重复申请和销毁
2. `width` 和 `height` 是这个 Render Texture 的长宽，这里设置为 `-1` 表示使用 `CommandBuffer` 所绑定的相机的长宽；
3. `depthBuffer` 设置为 `0` 表示这个 Render Texture 没有深度信息；
4. `filter` 是纹理采样模式，设置为 `FilterMode.Point` 表示点采样模式，在这种模式下，屏幕上的像素会寻找最近的贴图像素点作为输出，这种采样方式比较生硬，但性能较好，由于直接使用了相机的长宽，这个 Render Texture 的像素和输出图像的像素一一对应，不需要考虑采样问题，所以用最快的采样方法；
5. `format` 设置为 `GraphicsFormat.R16G16B16A16_SFloat` 表示 Render Texture 的像素RGBA均为16位浮点数。

接下来的 `SetRenderTarget` 为 `m_CommandBuffer` 设置了输出位置，表示将 `m_CommandBuffer` 的渲染结果放入上面申请的 Render Texture 中。`ClearRenderTarget` 即清空这个 Render Texture 开始新一轮渲染

#### 执行渲染

```c#
    m_CommandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture, BuiltinRenderTextureType.CameraTarget);

    // add sorting, view calc and drawing commands for each splat object
    Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

    // compose
    m_CommandBuffer.BeginSample(s_ProfCompose);
    m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
    m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
    m_CommandBuffer.EndSample(s_ProfCompose);
    m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
```