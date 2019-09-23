# 概率论提纲

## 随机事件及概率

### 随机事件

* 随机试验
* 随机事件：随机事件的每个可能出现的结果
* 基本事件
* 样本空间
* 必然事件$\Omega$
* 不可能事件$\empty$

### 事件的运算

* 和(与)$A \cap B$
* 积(或)$A \cup B$
* 差$A - B$
* 对立$\overline{A}$

### 事件的关系

* 包含$A \subset B$
* 相等$A = B$
* 互斥$A \cup B = \empty$
* 完备事件组

### 概率

* 统计定义
* 公理化定义
* 古典概型
* 几何概型

### 条件概率

* $P(A|B)=\frac{P(AB)}{P(B)}$
* 条件概率是一种概率
* 乘法公式（上面的式子反过来）
* 全概率公式：$B_i$互斥时有$P(A)=\sum_{i=1}^{n}P(B_i)P(A|B_i)$
* 贝叶斯公式/逆概公式：$P(B_i|A)=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^{n}P(B_j)P(A|B_j)}$
* **意义**：已经观测到一个结果($A$)，且已知结果的一个原因，可求出每个原因($B_i$)导致结果($A$)的程度大小

## 随机变量分布

* 离散型随机变量分布函数$P(X=x)$
* 连续性随机变量概率密度函数$f(x)=\frac{dP(X<=x)}{dx}$

## 随机变量数字特征

* 数学期望
  * 连续$E(X)=\int_{-\infty}^{+\infty}xf(x)dx$
  * 离散$E(X)=\sum_{n=1}^{\infty}x_nP(X=x_n)$
* 方差$D(X)=E((X-E(X))^2)$
* 协方差$COV(X,Y)=E((X-E(X))(Y-E(Y)))$
* 相关系数$\rho_{xy}=\frac{COV(X,Y)}{\sqrt{D(X)}\sqrt{D(Y)}}$
* 标准化$X^*=\frac{X-E(X)}{\sqrt{D(X)}}$
* 用标准化定义的相关系数$\rho_{xy}=COV(X^*,Y^*)$

## 统计分布

* 正态分布$X\sim\mathcal{N}(\mu,\sigma^2)$
* 卡方分布(chi-square)$\chi^2=\sum_{i=1}^{n}X_i^2\sim\chi^2(n)$，其中$X_i(i=1\dots n)\sim\mathcal{N}(0,1)$
* t分布$t=\frac{X}{\sqrt{Y/n}}$，其中$X\sim\mathcal{N}(0,1),Y\sim\chi^2(n)$

## 估计

* 点估计
  * 极大似然：调整参数使试验中观测到的值概率最大
* 区间估计：对给定$\alpha$求区间$[\underline{\theta},\overline{\theta}]$使得$P(\underline{\theta}<\theta<\overline{\theta})=1-\alpha$

## 假设检验

* 提出假设
* 选择统计量
* 构造小概率事件
* 计算拒绝域
* 计算统计量，看是否落在拒绝域

### 单总体T检验

问题：难产儿出生数35，体重均值3.42，S = 0.40，一般婴儿出生体重$\mu_0=3.30$（大规模调查获得），问相同否？

* 提出假设：$H_0:\mu=\mu_0,H_1:\mu\neq\mu_0$
* 选择统计量：t用分布进行检验$t=\frac{X}{\sqrt{Y/n}}$
* 构造小概率事件：$t\in(-\infty,\underline{t})\cup(\overline{t},+\infty)$
* 计算拒绝域：令$\alpha=0.05$，求$P(\underline{t}<t<\overline{t})=1-\alpha$时$\underline{t}$和$\overline{t}$的值
* 计算统计量：算出$t=\frac{X}{\sqrt{Y/n}}=\frac{\overline{x}-\mu_0}{\frac{S}{\sqrt{n}}}=1.77$
* 看是否有$t\in(-\infty,\underline{t})\cup(\overline{t},+\infty)$，有则拒绝原假设，否则接受原假设
  * 算p-value：$p=P(t>1.77)$
