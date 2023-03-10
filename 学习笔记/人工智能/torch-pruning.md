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

从`DG = tp.DependencyGraph()`开始的调用从上往下看，可以看见`DG.register_customized_layer`、`DG.build_dependency`、`pruning_group = DG.get_pruning_group`和`pruning_group.exec()`。

### `DG.register_customized_layer`

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

### `DG.build_dependency`

`DG.build_dependency`用于解析模型中层之间的调用关系，即解析`torch.nn.Module.forward`中的内容。

采取的方式是用一个样例输入执行推断过程，在推断过程进行trace。
具体的trace方案在`DG.build_dependency._trace`函数中。
简言之，就是通过`torch.nn.Module.register_forward_hook`注册hook，从而在每个`forward`函数被调用时记录下调用顺序。

具体来说，`DG.build_dependency._trace`函数trace出的调用顺序包括两方面：输入来自哪些层、输出到哪些层。
得知这些信息后，`DG.build_dependency`会调用`DG._build_dependency`，这个函数将每一个层与层之间的调用顺序（x层的输出到y层的输入）构建为一个`tp.Dependency`，加进相关层的`node.dependencies`中：
```python
    def _build_dependency(self, module2node):

        # There will be a dependency between two pruning operations if they:
        # 1) connects to each other in the computational graph or
        # 2) are equivalent, i.e., applied to the same layer and works in the same way.
        # Note that for some units like BN and PReLU, pruning output channels is equivalent to pruning output_channels
        # Rule 2) is designed for this case.

        for _, node in module2node.items():
            # Rule 1) - Input connections
            for in_node in node.inputs:
                handler = self.REGISTERED_PRUNERS.get(in_node.type)
                if handler is None:
                    handler = self.CUSTOMIZED_PRUNERS[in_node.class_type]
                handler = handler.prune_out_channels

                trigger = self.REGISTERED_PRUNERS.get(node.type)
                if trigger is None:
                    trigger = self.CUSTOMIZED_PRUNERS[node.class_type]
                trigger = trigger.prune_in_channels

                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=in_node
                )
                node.dependencies.append(dep)

            # Rule 1) - Output connections
            for out_node in node.outputs:
                trigger = self.REGISTERED_PRUNERS.get(node.type)
                if trigger is None:
                    trigger = self.CUSTOMIZED_PRUNERS[node.class_type]
                trigger = trigger.prune_out_channels

                handler = self.REGISTERED_PRUNERS.get(out_node.type)
                if handler is None:
                    handler = self.CUSTOMIZED_PRUNERS[out_node.class_type]
                handler = handler.prune_in_channels

                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=out_node
                )
                node.dependencies.append(dep)
......
```
看`tp.Dependency`的输入`trigger=trigger, handler=handler, source=node, target=out_node`，很明显这表示：当`source`层的裁剪过程`trigger`被调用时，需要调用`target`层的`handler`。

此外，进一步看`REGISTERED_PRUNERS`和`CUSTOMIZED_PRUNERS`：
```python
class DependencyGraph(object):

    def __init__(self):
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.ConcatPruner(),
            ops.OPTYPE.SPLIT: ops.SplitPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.CUSTOMIZED: None,
        }
        self.REGISTERED_PRUNERS = function.PrunerBox.copy()  # shallow copy
        self.REGISTERED_PRUNERS.update(_dummy_pruners)
        self.CUSTOMIZED_PRUNERS = {}
        self.IGNORED_LAYERS = []
......
    
    def register_customized_layer(
        self,
        layer_type,
        layer_pruner,
    ):
        """Register a customized layer for pruning.

        Args:
            layer_type (class): the type of layer
            pruner (tp.pruner.BasePruningFunc): a pruner for the given layer type.
        """
        self.CUSTOMIZED_PRUNERS[layer_type] = layer_pruner
......
```
可以发现它们实际上都继承自`tp.pruner.BasePruningFunc`，`PrunerBox`里面的几个是已实现的`_dummy_pruners`里面的几个都未实现。
所以很明显，只有在`DG.register_customized_layer`或者内部自带的继承自`tp.pruner.BasePruningFunc`的类的类方法才能被作为`tp.Dependency`里的`trigger`和`handler`。
再看这个`tp.pruner.BasePruningFunc`：
```python
class BasePruningFunc(ABC):
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, dim=1):
        self.dim = dim

    @abstractclassmethod
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def get_out_channels(self, layer: nn.Module):
        raise NotImplementedError

    @abstractclassmethod
    def get_in_channels(self, layer: nn.Module):
        raise NotImplementedError
        
......
```
这不就是“裁输入”和“裁输出”吗😂

所以**每一层的`node.dependencies`中实际上包含的都是：“trigger=上一层的输出被裁剪, handler=裁剪当前层的输入”或者“trigger=下一层的输入被裁剪, handler=裁剪当前层的输出”**。
看到这就清晰多了，这框架主打的自动解析依赖关系完成裁剪的功能归根结底就是以这种方式组织的。

### `pruning_group = DG.get_pruning_group`

在`DG.build_dependency`之后，模型中每个Module之间的调用关系就清楚了，于是输入任意一个要裁的“节点”（输出矩阵中的某个channel，也对应卷积层中的一个卷积核）都能知道该节点在模型中的前后依赖关系（例如该节点被裁剪导致输出少了一个channel，以之作为输入的所有层也需要相应的进行修改）。

`DG.get_pruning_group`就是这样一个根据输入的某个节点的裁剪方案输出整体裁剪方案的函数。

其输入为要裁的层`module: nn.Module`、裁剪该节点的函数`pruning_fn: typing.Callable`和裁哪些channel`idxs: typing.Union[list, tuple]`。
这里的`pruning_fn`虽然只是`typing.Callable`，但结合`DG.build_dependency`的解析，实际的依赖关系是由一个个`tp.Dependency`所记录函数间的触发关系所描述的。
所以输入的`pruning_fn`要想能触发依赖关系上的相关裁剪函数，其必须是`tp.Dependency`中已经记录过的函数，换言之，它必须要是一个继承于`tp.pruner.BasePruningFunc`子类的类方法，其逻辑上的功能其实是指定是裁这一层的输入还是输出。

其输出的修改方案类名为`tp.DependencyGroup`，其是由`tp.Dependency`组成的数组。
`DG.get_pruning_group`的本质就是从`module`对应的`node.dependencies`（`DG.build_dependency`中构建的依赖关系）中找出`trigger=pruning_fn`的那些`tp.Dependency`组成`tp.DependencyGroup`。

### `pruning_group.exec()`

最后就是执行这个修改`DG.get_pruning_group`输出的修改方案，很好理解，就是按照`tp.DependencyGroup`里的`tp.Dependency`一个个执行它们的`handler`就行了。
