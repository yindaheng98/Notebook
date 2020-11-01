# Python包导入问题

## 问题描述

有一个Python包包含许多互相导入的子包、一个在包外的脚本想要导入这个包，问包内import和包外脚本应该如何书写，使得对任意位置的包外脚本都能导入包内的所有名称？

## 解决方案1：使用相对路径

这是Python建议的解决方案。

需要包是标准格式，即有`__init__.py`文件，否则会报“不可使用相对路径”的错误。

包内：

```python
from . import xxx as x
from . xxx import xxxx as xx
```

包外：

```python
sys.path.append("包的上一级目录")
import xxxxx # xxxxx为包名
```

## 解决方案2：将包路径加入`os.path`中

在`__init__.py`文件中写上：

```python
sys.path.append(os.path.split(os.path.abspath(__file__))[0])
```

包外：

```python
import xxxxx
import xxx as x
from xxx import xxxx as xx
```

### 原理

`__init__.py`文件开头的那就好

Python启动时，`os.path`中会有一系列预置的路径（一般为当前目录+pip安装的包的路径），当需要`import`时，Python会从`os.path`所指的路径中查找所需要`import`的名称，进行导入。换句话说，所有在`os.path`所指目录下的文件都相当于在当前目录下，与当前目录下的Python平起平坐。

因此，将包路径加入`os.path`中就相当于这个包的内容处在当前目录下，这个包是不是一个标准的包都无所谓了。

`__init__.py`文件开头的那句话将`__init__.py`的上级目录，即包的根目录加入到`os.path`，`import`这个包时此句自动执行，之后所有`import`操作就都如同在包内一样了。