# My-docs

这里面都是我的一些学习笔记和一些实验报告和一些作业

## 记录一下使用方法

先在本机上进行：

```shell
python meta.py
```

此脚本会读取各文件的创建时间和标题保存到meta.json中。使用这种方法是因为git在传文件时貌似不会保持文件的创建时间。无法自动收集blog的时间。

此时可以对meta.json中的数据进行修改，meta.py不会覆盖已有的数据。

再在TravisCI中进行：

```shell
python process.py
bash deploy.sh $GH_TOKEN
```

其中py脚本会将meta.json中的数据按照Hexo的格式放到md文件开头，shell脚本会把md文件push到我的github.io仓库中

以上操作完成后，在github.io仓库中的TravisCI会进一步执行编译和部署操作。
