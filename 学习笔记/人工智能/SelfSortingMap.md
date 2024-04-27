# Self-Sorting Map

前置知识：[Self-Organizing Map](./自组织映射.md)

显然，SOM并不能保证每个训练样本都有一个独立的编号，也不能保证最终的图上的像素值与训练样本完全相等。
而Self-Sorting Map(SSM)可以保证每个训练样本都有自己的ID，且出来的图上像素值与训练样本完全相等，因为SSM正如其名，是“排序”，不是SOM那种“训练”和“拟合”；并且因为是“排序”，所以构建SSM也不需要像SOM那种iteration。

典型地，SSM可以做到下面这种效果，即在Self-Organizing的同时还能保证表达数据的精准无误：

![](i/SSM.png)
