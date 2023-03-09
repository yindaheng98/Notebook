# [Torch-Pruning](https://github.com/VainF/Torch-Pruning)解析

## 底层调用

```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
from typing import Sequence

############
# Customize your layer
#
class CustomizedLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))
        self.fc = nn.Linear(self.in_dim, self.in_dim)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return self.fc(x * self.scale + self.bias)

    def __repr__(self):
        return "CustomizedLayer(in_dim=%d)"%(self.in_dim)

class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.customized_layer = CustomizedLayer(HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.customized_layer(x)
        y_hat = self.fc2(x)
        return y_hat

############################
# Implement your pruning function for the customized layer
# You should implement the following class fucntions:
# 1. prune_out_channels
# 2. prune_in_channels
# 3. get_out_channels
# 4. get_in_channels

class MyPruner(tp.pruner.BasePruningFunc):

    def prune_out_channels(self, layer: CustomizedLayer, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_dim)) - set(idxs))
        keep_idxs.sort()
        layer.in_dim = layer.in_dim-len(idxs)
        layer.scale = torch.nn.Parameter(layer.scale.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        tp.prune_linear_in_channels(layer.fc, idxs)
        tp.prune_linear_out_channels(layer.fc, idxs)
        return layer

    def get_out_channels(self, layer):
        return self.in_dim
    
    # identical functions
    prune_in_channels = prune_out_channels
    get_in_channels = get_out_channels
        
model = FullyConnectedNet(128, 10, 256)

DG = tp.DependencyGraph()

# 1. Register your customized layer
my_pruner = MyPruner()
DG.register_customized_layer(
    CustomizedLayer, 
    my_pruner)

# 2. Build dependency graph
DG.build_dependency(model, example_inputs=torch.randn(1,128))

# 3. get a pruning group according to the dependency graph. idxs is the indices of pruned filters.
pruning_group = DG.get_pruning_group( model.fc1, tp.prune_linear_out_channels, idxs=[0, 1, 6] )
print(pruning_group)

# 4. execute this group (prune the model)
pruning_group.exec()
print("The pruned model:\n", model)
print("Output: ", model(torch.randn(1,128)).shape)

assert model.fc1.out_features==253 and model.customized_layer.in_dim==253 and model.fc2.in_features==253
```

`DG = tp.DependencyGraph()`是整个系统的核心模块，看它的调用流程也就看懂了整个`Torch-Pruning`的裁剪过程。

从`DG = tp.DependencyGraph()`开始的调用从上往下看，可以看见`DG.register_customized_layer`、`DG.build_dependency`、`pruning_group = DG.get_pruning_group`和`pruning_group.exec()`

## `DG.register_customized_layer`

`DG.register_customized_layer`用于注册“裁剪方式”。“裁剪方式”与Pytorch中的层一一对应

`DG = tp.DependencyGraph()`内部已经注册了一些层的默认裁剪方式，包括卷积层和线性层等：

```python
PrunerBox = {
    ops.OPTYPE.CONV: ConvPruner(),
    ops.OPTYPE.LINEAR: LinearPruner(),
    ops.OPTYPE.BN: BatchnormPruner(),
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvPruner(),
    ops.OPTYPE.PRELU: PReLUPruner(),
    ops.OPTYPE.LN: LayernormPruner(),
    ops.OPTYPE.EMBED: EmbeddingPruner(),
    ops.OPTYPE.PARAMETER: ParameterPruner(),
    ops.OPTYPE.MHA: MultiheadAttentionPruner(),
    ops.OPTYPE.LSTM: LSTMPruner()
}
```

```python
_dummy_pruners = {
    ops.OPTYPE.CONCAT: ops.ConcatPruner(),
    ops.OPTYPE.SPLIT: ops.SplitPruner(),
    ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
    ops.OPTYPE.CUSTOMIZED: None,
}
```

## `DG.build_dependency`

`DG.build_dependency`用于解析模型中层之间的调用关系，即解析`torch.nn.Module.forward`中的内容。

采取的方式是用一个样例输入执行推断过程，在推断过程进行trace。
具体的trace方案在`DG.build_dependency._trace`函数中。
简言之，就是通过`torch.nn.Module.register_forward_hook`注册hook，从而在每个`forward`函数被调用时记录下调用顺序。

## `pruning_group = DG.get_pruning_group`

在`DG.build_dependency`之后，模型中每个Module之间的调用关系就清楚了，于是输入任意一个要裁的“节点”（输出矩阵中的某个channel，也对应卷积层中的一个卷积核）都能知道该节点在模型中的前后依赖关系（例如该节点被裁剪导致输出少了一个channel，以之作为输入的所有层也需要相应的进行修改）。

`DG.get_pruning_group`就是这样一个根据输入的待裁节点输出裁剪方案的函数。
其输出的修改方案类名为`tp.DependencyGroup`，其由一系列`tp.Dependency`组成，每个`tp.Dependency`都是针对模型中某一层的修改方案。

## `pruning_group.exec()`

最后就是执行这个修改`DG.get_pruning_group`输出的修改方案，很好理解，就是按照`tp.DependencyGroup`里的修改方案`tp.Dependency`一个个执行下去就行了。
