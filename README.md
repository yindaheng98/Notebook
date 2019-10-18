# My-docs

这里面都是我的一些学习笔记和一些实验报告和一些作业

## 记录一下使用方法

在TravisCI中进行：

```shell
python meta.py
python process.py
bash deploy.sh $GH_TOKEN
```

其中py脚本会将meta.json中的数据按照Hexo的格式放到md文件开头，shell脚本会把md文件push到我的github.io仓库中

以上操作完成后，在github.io仓库中的TravisCI会进一步执行编译和部署操作。

## 脚本说明

### meta.py

此脚本会从.git的commit记录中读取各文件的标题、所在目录和最后修改时间（重命名和移动不算修改）保存到各个文件夹的`_meta.json`中。此外，该脚本还会读取每个.md文件的第一张图片作为封面数据写入`_meta.json`。

此会覆盖`_meta.json`中的标题、所在目录修改时间和封面数据，但是`_meta.py`的`tags`数据不会被覆盖。

### process.py

此脚本从各文件夹下的`_meta.json`文件中读取数据，然后按照Hexo的格式放到`.md`文件开头。

### deploy.sh

下载另一个仓库的Hexo博客生成器，把博客源文件放入指定位置，并推送。之后生成博客和github pages的过程由另一个仓库从的CI完成。
