# 3DGS转Mesh方法汇总 - 3DGS+SDF/UDF联合优化

3DGS+SDF/UDF联合优化 -> 对SDF/UDF [Marching Cubes](MarchingCube.md) -> Mesh

## NeuSG：让 GS 提供细节，让 SDF 提供连续表面和法线

Gaussian 点云细节丰富，但不连续且有噪声；SDF 连续完整，但容易过平滑。让两种表示互相纠错，比单独使用任何一种都好。

NeuSG是3DGS和一种Implicit Surfaces方法NeuS的结合。

和[PGSR](3D高斯深度渲染.md)一样，NeuSG也设置了一个loss强制拉低最短轴长并以最短轴长作为法线方向。
![](i/NeuGSLoss1.png)
![](i/NeuGSNormal.png)
并且使用SDF的梯度方向约束法线方向。
![](i/NeuGSLoss2.png)
另外，还用3DGS中点位置约束SDF，使其在这些位置处值为0，其实就是把3DGS约束在NeuS的表面上。
![](i/NeuGSLoss3.png)
当然NeuS自己的loss也都保留。
![](i/NeuGSLoss4.png)
![](i/NeuGSLoss5.png)

NeuS和3DGS的RGB loss是分开算了加在一起的。
![](i/NeuGSLoss6.png)

可以看出，作为3DGS+SDF联合优化的早期工作，NeuSG的本质就是在NeuS和3DGS分别优化的基础上加上了用NeuS梯度方向约束3DGS法线方向和把3DGS约束在NeuS的表面上这两个loss。

虽然简单但是已经能体现处联合优化Mesh质量的思想。

## (NeurIPS 2024) GSDF

## (SIGGRAPH Asia 2024, ACM TOG) 3DGSR

## (CVPR 2025 Highlight) GaussianUDF

## (CVPR 2024) SuGaR: 先得到 mesh，再把 Gaussian 粘到三角面上
