MachineLearning算法原理及python实现
========

# 目录
* [第一部分：机器学习算法框架及基础](#第一部分：机器学习算法框架)
	* [一、框架图](#一框架图)
  	* [二、降维与特征选择](#二降维与特征选择)
	* [三、评估方法](#评估方法)
	* [四、过拟合与欠拟合](#过拟合与欠拟合)
* [第二部分：回归算法](#第二部分：回归算法)
	* [一、线性回归](#一线性回归)
  	* [二、Ridge回归与Lasso回归](#二Ridge回归与Lasso回归)
* [第三部分：分类算法](#第三部分：回归算法)
	* [一、基本概念与性能度量](#一基本概念与性能度量)
	* [二、决策树](#一决策树)
  	* [三、随机森林](#三随机森林)
	* [四、xgboost](#四xgboost)
* [第四部分：聚类算法](#第四部分：聚类算法)
	* [一、K-Means](#一K-Means)
* [第五部分：关联算法](#第五部分：关联算法)
	* [一、Apriori](#一Apriori)
	
# 内容
## 第一部分：机器学习算法框架及基础
### 一、框架图
&ensp;&ensp;&ensp;&ensp;![机器学习框架图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9F%A5%E8%AF%86%E4%BD%93%E7%B3%BB%E5%9B%BE.png) 
### 二、降维与特征选择
&ensp;&ensp;&ensp;&ensp;![降维框架图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/降维框架图.png) 
#### 1、降维
* 降维的作用  
&ensp;&ensp;&ensp;&ensp; 1.降低时间复杂度和空间复杂度；2.节省了提取不必要特征的开销；3.去掉数据集中夹杂的噪音；4.较简单的模型在小数据集上有更强的鲁棒性；5.当数据能有较少的特征进行解释，我们可以更好的解释数据；6.实现数据可视化。
* 几种降维方法：  
（1）主成分分析算法（PCA）  
&ensp;&ensp;&ensp;&ensp;Principal Component Analysis(PCA)是最常用的线性降维方法，它的目标是通过某种线性投影，将高维的数据映射到低维的空间中表示，并期望在所投影的维度上数据的方差最大，以此使用较少的数据维度，同时保留住较多的原数据点的特性。  
&ensp;&ensp;&ensp;&ensp;通俗的理解，如果把所有的点都映射到一起，那么几乎所有的信息（如点和点之间的距离关系）都丢失了，而如果映射后方差尽可能的大，那么数据点则会分散开来，以此来保留更多的信息。可以证明，PCA是丢失原始数据信息最少的一种线性降维方式。（实际上就是最接近原始数据，但是PCA并不试图去探索数据内在结构）
&ensp;&ensp;&ensp;&ensp;设n维向量w为目标子空间的一个坐标轴方向（称为映射向量），最大化数据映射后的方差，有：  
![主成分分析算法方差](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/主成分分析算法方差.png)   
&ensp;&ensp;&ensp;&ensp;其中m是数据实例的个数， xi是数据实例i的向量表达， x拔是所有数据实例的平均向量。定义W为包含所有映射向量为列向量的矩阵，经过线性代数变换，可以得到如下优化目标函数：  
&ensp;&ensp;&ensp;&ensp;![主成分分析算法目标函数](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/主成分分析算法目标函数.png)  
&ensp;&ensp;&ensp;&ensp;其中tr表示矩阵的迹。  
&ensp;&ensp;&ensp;&ensp;![主成分分析算法协方差矩阵](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/主成分分析算法协方差矩阵.png)   
&ensp;&ensp;&ensp;&ensp;A是数据协方差矩阵。  
&ensp;&ensp;&ensp;&ensp;容易得到最优的W是由数据协方差矩阵前k个最大的特征值对应的特征向量作为列向量构成的。这些特征向量形成一组正交基并且最好地保留了数据中的信息。  
&ensp;&ensp;&ensp;&ensp;PCA的输出就是Y = WX，由X的原始维度降低到了k维。  
&ensp;&ensp;&ensp;&ensp;PCA追求的是在降维之后能够最大化保持数据的内在信息，并通过衡量在投影方向上的数据方差的大小来衡量该方向的重要性。但是这样投影以后对数据的区分作用并不大，反而可能使得数据点揉杂在一起无法区分。这也是PCA存在的最大一个问题，这导致使用PCA在很多情况下的分类效果并不好。具体可以看下图所示，若使用PCA将数据点投影至一维空间上时，PCA会选择2轴，这使得原本很容易区分的两簇点被揉杂在一起变得无法区分；而这时若选择1轴将会得到很好的区分结果。  
&ensp;&ensp;&ensp;&ensp;![主成分分析算法降维示意图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/主成分分析算法降维示意图.png)   
&ensp;&ensp;&ensp;&ensp;Discriminant Analysis所追求的目标与PCA不同，不是希望保持数据最多的信息，而是希望数据在降维后能够很容易地被区分开来。后面会介绍LDA的方法，是另一种常见的线性降维方法。另外一些非线性的降维方法利用数据点的局部性质，也可以做到比较好地区分结果，例如LLE，Laplacian Eigenmap等。  
（2）LDA    
&ensp;&ensp;&ensp;&ensp;Linear Discriminant Analysis (也有叫做Fisher Linear Discriminant)是一种有监督的（supervised）线性降维算法。与PCA保持数据信息不同，LDA是为了使得降维后的数据点尽可能地容易被区分。  
&ensp;&ensp;&ensp;&ensp;假设原始数据表示为X,（m\*n矩阵，m是维度，n是sample的数量）  
&ensp;&ensp;&ensp;&ensp;既然是线性的，那么就是希望找到映射向量a，使得aX后的数据点能够保持以下两种性质：  
&ensp;&ensp;&ensp;&ensp;1、同类的数据点尽可能的接近（within class）  
&ensp;&ensp;&ensp;&ensp;2、不同类的数据点尽可能的分开（between class）  
&ensp;&ensp;&ensp;&ensp;所以呢还是上次PCA用的这张图，如果图中两堆点是两类的话，那么我们就希望他们能够投影到轴1去（PCA结果为轴2），这样在一维空间中也是很容易区分的。  
&ensp;&ensp;&ensp;&ensp;![LDA算法降维示意图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/LDA算法降维示意图.png)   
（3）局部线性嵌入（LLE）  
&ensp;&ensp;&ensp;&ensp;Locally linear embedding（LLE）是一种非线性降维算法，它能够使降维后的数据较好地保持原有流形结构。LLE可以说是流形学习方法最经典的工作之一。很多后续的流形学习、降维方法都与LLE有密切联系。   
&ensp;&ensp;&ensp;&ensp;![LLE算法降维示意图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/LLE算法降维示意图.png)  
&ensp;&ensp;&ensp;&ensp;见图，使用LLE将三维数据（b）映射到二维（c）之后，映射后的数据仍能保持原有的数据流形（红色的点互相接近，蓝色的也互相接近），说明LLE有效地保持了数据原有的流行结构。  
&ensp;&ensp;&ensp;&ensp;但是LLE在有些情况下也并不适用，如果数据分布在整个封闭的球面上，LLE则不能将它映射到二维空间，且不能保持原有的数据流形。那么我们在处理数据中，首先假设数据不是分布在闭合的球面或者椭球面上。如图，LLE降维算法使用实例：  
&ensp;&ensp;&ensp;&ensp;![LLE算法降维流程示意图](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/LLE算法降维流程示意图.png)   
&ensp;&ensp;&ensp;&ensp;LLE算法认为每一个数据点都可以由其近邻点的线性加权组合构造得到。算法的主要步骤分为三步：(1)寻找每个样本点的k个近邻点；（2）由每个样本点的近邻点计算出该样本点的局部重建权值矩阵；（3）由该样本点的局部重建权值矩阵和其近邻点计算出该样本点的输出值。具体的算法流程如下图所示：  
（4）Laplacian Eigenmaps 拉普拉斯特征映射  
&ensp;&ensp;&ensp;&ensp;Laplacian Eigenmaps 看问题的角度和LLE有些相似，也是用局部的角度去构建数据之间的关系。它的直观思想是希望相互间有关系的点在降维后的空间中尽可能的靠近。Laplacian Eigenmaps可以反映出数据内在的流形结构。

#### 2、特征选择 
- 特征选择的目标  
&ensp;&ensp;&ensp;&ensp;引用自吴军《数学之美》上的一句话：一个正确的数学模型应当在形式上是简单的。构造机器学习的模型的目的是希望能够从原始的特征数据集中学习出问题的结构与问题的本质，当然此时的挑选出的特征就应该能够对问题有更好的解释，所以特征选择的目标大致如下：  
&ensp;&ensp;&ensp;&ensp;<1>提高预测的准确性  
&ensp;&ensp;&ensp;&ensp;<2>构造更快，消耗更低的预测模型  
&ensp;&ensp;&ensp;&ensp;<3>能够对模型有更好的理解和解释
- 特征选择的方法  
&ensp;&ensp;&ensp;&ensp;主要有三种方法：  
（1）Filter方法  
&ensp;&ensp;&ensp;&ensp;其主要思想是：对每一维的特征“打分”，即给每一维的特征赋予权重，这样的权重就代表着该维特征的重要性，然后依据权重排序。  
&ensp;&ensp;&ensp;&ensp;主要的方法有:  
&ensp;&ensp;&ensp;&ensp;Chi-squared test(卡方检验)、information gain(信息增益)、correlation coefficient scores(相关系数)  
（2）Wrapper方法  
&ensp;&ensp;&ensp;&ensp;其主要思想是：将子集的选择看作是一个搜索寻优问题，生成不同的组合，对组合进行评价，再与其他的组合进行比较。这样就将子集的选择看作是一个优化问题，这里有很多的优化算法可以解决，尤其是一些启发式的优化算法，如GA，PSO，DE，ABC等。  
&ensp;&ensp;&ensp;&ensp;主要方法有：recursive feature elimination algorithm(递归特征消除算法)  
（3）Embedded方法  
&ensp;&ensp;&ensp;&ensp;其主要思想是：在模型既定的情况下学习出对提高模型准确性最好的属性。这句话并不是很好理解，其实是讲在确定模型的过程中，挑选出那些对模型的训练有重要意义的属性。  
&ensp;&ensp;&ensp;&ensp;主要方法：正则化，岭回归就是在基本线性回归的过程中加入了正则项。
- 特征选择和降维的相同点和不同点  
&ensp;&ensp;&ensp;&ensp;特征选择和降维有着些许的相似点，这两者达到的效果是一样的，就是试图去减少特征数据集中的属性(或者称为特征)的数目；但是两者所采用的方式方法却不同：降维的方法主要是通过属性间的关系，如组合不同的属性得新的属性，这样就改变了原来的特征空间；而特征选择的方法是从原始特征数据集中选择出子集，是一种包含的关系，没有更改原始的特征空间。

### 三、评估方法 
&ensp;&ensp;&ensp;&ensp;学习器的学习能力和泛化能力是由学习器的算法和数据的内在结构共同决定的。对于学习器的算法，不同的参数设置能导致算法生成不同的模型，如何评估一个模型的好坏，我们通常从其泛化误差来进行评估。  
&ensp;&ensp;&ensp;&ensp;泛化误差的定义是学习器在新样本上的误差，而新样本我们通常是不知道其真实输出的，那么如何评估呢？为此，通常在训练集中分出一部分数据，这些并不参与学习器的学习，而是使用其他的剩余样本来训练样本，用这些选出来的样本进行泛化误差的计算。这些被选出来的样本我们称之为验证集（testing set）因为我们假设验证样本的是从真实样本中独立同分布的采样而来，从而可以将测试误差当作泛化误差的近似。  
&ensp;&ensp;&ensp;&ensp;给定一个数据集D={(x1,y1),(x2,y2),⋅⋅⋅,(xm,ym)}，如何从中选出我们所需的训练集S和验证集T呢？常用的方法有以下几种。
- 留出法  
&ensp;&ensp;&ensp;&ensp;留出法是直接将数据集D划分为两个互斥的集合。其中一个作为验证集，一个作为训练集。  
&ensp;&ensp;&ensp;&ensp;<1>注意，两个样本子集的划分尽量保证数据分布的一致性，即子集中正反样本的数目尽量平衡。通常用分层采样来保留类别比例。  
&ensp;&ensp;&ensp;&ensp;<2>一次划分往往容易导致结果不稳定，通常采用多次划分重复评估取平均值的方法来提高可靠性。  
&ensp;&ensp;&ensp;&ensp;<3>通常使用数据的2/3∼4/5作为训练，剩余的用作验证。
- 交叉验证法  
&ensp;&ensp;&ensp;&ensp;将数据集划为k个大小近似的互斥子集，每个子集都是通过分层采样得到。每次都将k−1个子集作为训练集，剩余的一个子集作为验证集。这样就能有k次试验。因此，此种方法也称为“k折交叉验证”(k-fold cross validation)。
- 自助法  
&ensp;&ensp;&ensp;&ensp;使用自助采样法：每次有放回的从数据集中采样一个数据，重复m次获得一个子样本集。这样的子样本集中可能含有同样的样本。用没有出现在该子样本集中的样本数据作为验证样本集。一个样本不被采样到的概率为：  
  &ensp;&ensp;&ensp;&ensp;![一个样本不被采样到的概率](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/一个样本不被采样到的概率.png)  
&ensp;&ensp;&ensp;&ensp;自助法适用于数据集小、难以有效划分训练集/验证集的场合。

### 四、过拟合与欠拟合
- 1、欠拟合：模型在训练和预测时表现都不好的情况，型没有很好地捕捉到数据特征，不能够很好地拟合数据。  
解决方法：  
&ensp;&ensp;&ensp;&ensp添加其他特征项，有时候我们模型出现欠拟合的时候是因为特征项不够导致的，可以添加其他特征项来很好地解决。例如，“组合”、“泛化”、“相关性”三类特征是特征添加的重要手段，无论在什么场景，都可以照葫芦画瓢，总会得到意想不到的效果。除上面的特征之外，“上下文特征”、“平台特征”等等，都可以作为特征添加的首选项。  
（1）添加多项式特征，这个在机器学习算法里面用的很普遍，例如将线性模型通过添加二次项或者三次项使模型泛化能力更强。例如上面的图片的例子。  
（2）减少正则化参数，正则化的目的是用来防止过拟合的，但是现在模型出现了欠拟合，则需要减少正则化参数。
- 2、过拟合：在训练数据上表现良好，在未知数据上表现差。  
解决方法：  
（1）特征筛选，通过相关性、方差大小等筛选特征。  
（2）重新清洗数据，导致过拟合的一个原因也有可能是数据不纯导致的，如果出现了过拟合就需要我们重新清洗数据。  
（3）增大数据的训练量，还有一个原因就是我们用于训练的数据量太小导致的，训练数据占总数据的比例过小。  
（4）采用正则化方法。正则化方法包括L0正则、L1正则和L2正则，而正则一般是在目标函数之后加上对于的范数。但是在机器学习中一般使用L2正则，下面看具体的原因。  
&ensp;&ensp;&ensp;&ensp;L0范数是指向量中非0的元素的个数。L1范数是指向量中各个元素绝对值之和，也叫“稀疏规则算子”（Lasso regularization）。两者都可以实现稀疏性，既然L0可以实现稀疏，为什么不用L0，而要用L1呢？个人理解一是因为L0范数很难优化求解（NP难问题），二是L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解。所以大家才把目光和万千宠爱转于L1范数。  
&ensp;&ensp;&ensp;&ensp;L2范数是指向量各元素的平方和然后求平方根。可以使得W的每个元素都很小，都接近于0，但与L1范数不同，它不会让它等于0，而是接近于0。L2正则项起到使得参数w变小加剧的效果，但是为什么可以防止过拟合呢？一个通俗的理解便是：更小的参数值w意味着模型的复杂度更低，对训练数据的拟合刚刚好（奥卡姆剃刀），不会过分拟合训练数据，从而使得不会过拟合，以提高模型的泛化能力。还有就是看到有人说L2范数有助于处理 condition number不好的情况下矩阵求逆很困难的问题。  
（5）采用dropout方法。这个方法在神经网络里面很常用。dropout方法是ImageNet中提出的一种方法，通俗一点讲就是dropout方法在训练的时候让神经元以一定的概率不工作。具体看下图：  
&ensp;&ensp;&ensp;&ensp;![dropout](https://github.com/Lg-AiLearn/ML/blob/master/images/part_one_frame/dropout.jpeg) 

## 第二部分：回归算法
### 一、线性回归
#### 代码实现（自定义）
* [代码实现（自定义）](/LinearRegression/LogisticRegression.py)
#### 代码实现（scikit-learn库）
* [代码实现（scikit-learn库）](/LinearRegression/LinearRegression_scikit-learn.py)
#### 1、线性回归问题
&ensp;&ensp;&ensp;&ensp;线性回归，是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w’x+e，e为误差服从均值为0的正态分布。它适用于有监督学习的预测。  
&ensp;&ensp;&ensp;&ensp;一元线性回归分析：hθ(x)=ax+b，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示。  
&ensp;&ensp;&ensp;&ensp;多元线性回归分析：hθ(x)=θ0+θ1x1+...+θnxn，包括两个或两个以上的自变量，并且因变量和自变量是线性关系。
&ensp;&ensp;&ensp;&ensp;综上线性回归模型可统一为：  
&ensp;&ensp;&ensp;&ensp;![线性回归模型](https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.png)  
&ensp;&ensp;&ensp;&ensp;公式中θ和x是向量，n是样本数。  
&ensp;&ensp;&ensp;&ensp;假如我们依据这个公式来预测h(x)，公式中的x是我们已知的，然而θ的取值却不知道，只要我们把θ的取值求解出来，我们就可以依据这个  公式来做预测了。  
&ensp;&ensp;&ensp;&ensp;那么如何依据训练数据求解θ的最优取值呢？这就牵扯到另外一个概念：**损失函数（Loss Function）**。
#### 2、损失函数
- ![损失函数](https://github.com/Lg-AiLearn/ML/blob/master/images/LineRecost.png)
- 损失函数代码实现：
```python
# 计算代价函数
def computerCost(X,y,theta):
    m = len(y)
    J = 0
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m) #计算代价J
    return J
```
&ensp;&ensp;&ensp;&ensp;要选择最优的θ，使得h(x)最近进真实值。这个问题就转化为求解最优的θ，使损失函数J(θ)取最小值。那么如何解决这个转化后的问题呢？  这又牵扯到一个概念：**梯度下降（Radient Descent）**。
#### 3、梯度下降
&ensp;&ensp;&ensp;&ensp;怎样最小化损失函数?损失函数的定义是一个凸函数，就可以使用凸优化的一些方法：  
* 梯度下降法：
&ensp;&ensp;&ensp;&ensp;梯度下降法是最早最简单，也是最为常用的最优化方法。梯度下降法实现简单，当目标函数是凸函数时，梯度下降法的解是全局解。一般情况下，其解不保证是全局最优解，梯度下降法的速度也未必是最快的。梯度下降法的优化思想是用当前位置负梯度方向作为搜索方向，因为该方向为当前位置的最快下降方向，所以也被称为是”最速下降法“。最速下降法越接近目标值，步长越小，前进越慢。梯度下降法的搜索迭代示意图如下图所示：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E5%9B%BE%E7%A4%BA.png)  
&ensp;&ensp;&ensp;&ensp;损失函数对θ求偏导：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E6%B1%82%E5%81%8F%E5%AF%BC.png)  
&ensp;&ensp;&ensp;&ensp;所以，θ更新方式如下：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/theta.png)  
&ensp;&ensp;&ensp;&ensp;其中![\alpha ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Calpha%20)为学习速率，控制梯度下降的速度，一般取**0.01,0.03,0.1,0.3.....**

- 为什么梯度下降可以逐步减小代价函数：  
&ensp;&ensp;&ensp;&ensp;假设函数`f(x)`  
&ensp;&ensp;&ensp;&ensp;泰勒展开：`f(x+△x)=f(x)+f'(x)*△x+o(△x)`  
&ensp;&ensp;&ensp;&ensp;令：`△x=-α*f'(x)`   ,即负梯度方向乘以一个很小的步长`α`  
&ensp;&ensp;&ensp;&ensp;将`△x`代入泰勒展开式中：`f(x+△x)=f(x)-α*[f'(x)]²+o(△x)`  
&ensp;&ensp;&ensp;&ensp;可以看出，`α`是取得很小的正数，`[f'(x)]²`也是正数，所以可以得出：`f(x+△x)<=f(x)`  
&ensp;&ensp;&ensp;&ensp;所以沿着**负梯度**放下，函数在减小，多维情况一样。

- 学习率：  
&ensp;&ensp;&ensp;&ensp;上段公式中的α就是学习率。它决定了下降的节奏快慢，就像一个人下山时候步伐的快慢。α过小会导致收敛很慢，α太大有可能会导致震荡。如何选择学习率呢，目前也有好多关于学习率自适应算法的研究。工程上，一般会调用一些开源包，包含有一些自适应方法。自己做的话会选择相对较小的α，比如0.01。下图展示了梯度下降的过程。   
&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E7%9A%84%E9%98%BF%E5%B0%94%E6%B3%95%E9%97%AE%E9%A2%98.png)  
- 梯度下降法的缺点：  
（1）靠近极小值时收敛速度减慢，如下图所示；  
（2）直线搜索时可能会产生一些问题；  
（3）可能会“之字形”地下降。  
&ensp;&ensp;&ensp;&ensp;<img src="https://github.com/Lg-AiLearn/ML/blob/master/images/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E5%AD%98%E5%9C%A8%E7%9A%84%E9%97%AE%E9%A2%98.png" height="330" width="380" >
- 梯度下降法的具体形式  
（1）批量梯度下降法（Batch Gradient Descent，简称BGD）是梯度下降法最原始的形式，它的具体思路是在更新每一参数时都使用所有的样本来进行更新。  
&ensp;&ensp;&ensp;&ensp;优点：全局最优解；易于并行实现；  
&ensp;&ensp;&ensp;&ensp;缺点：当样本数目很多时，训练过程会很慢。  
（2）随机梯度下降法：它的具体思路是在更新每一参数时都使用一个样本来进行更新。每一次更新参数都用一个样本，更新很多次。如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次，这种跟新方式计算复杂度太高。
&ensp;&ensp;&ensp;&ensp;但是，SGD伴随的一个问题是噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。  
&ensp;&ensp;&ensp;&ensp;优点：训练速度快；  
&ensp;&ensp;&ensp;&ensp;缺点：准确度下降，并不是全局最优；不易于并行实现。  
（3）小批量梯度下降法（Mini-batch Gradient Descent，简称MBGD）：它的具体思路是在更新每一参数时都使用一部分样本来进行更新，也就是方程中的m的值大于1小于所有样本的数量。为了克服上面两种方法的缺点，又同时兼顾两种方法的有点。
如果样本量比较小，采用批量梯度下降算法。如果样本太大，或者在线算法，使用随机梯度下降算法。在实际的一般情况下，采用小批量梯度下降算法。

- 实现代码
```python
# 梯度下降算法
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)   
    temp = np.matrix(np.zeros((n,num_iters)))   # 暂存每次迭代计算的theta，转化为矩阵形式
    J_history = np.zeros((num_iters,1)) #记录每次迭代计算的代价值
    for i in range(num_iters):  # 遍历迭代次数    
        h = np.dot(X,theta)     # 计算内积，matrix可以直接乘
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))   #梯度的计算
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)      #调用计算代价函数
        print '.',      
    return theta,J_history  
```
* 补充其它凸函数求解最优点的方法：  
（1）牛顿法：  
&ensp;&ensp;&ensp;&ensp;是一种在实数域和复数域上近似求解方程的方法。方法使用函数f (x)的泰勒级数的前面几项来寻找方程f (x) = 0的根。牛顿法最大的特点就在于它的收敛速度很快。  
&ensp;&ensp;&ensp;&ensp;从几何上说，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径。  
&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/%E7%89%9B%E9%A1%BF%E6%B3%95.png)  
**牛顿法的优缺点总结：**  
&ensp;&ensp;&ensp;&ensp;优点：二阶收敛，收敛速度快；  
&ensp;&ensp;&ensp;&ensp;缺点：牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂。  
**关于牛顿法和梯度下降法的效率对比：**  
&ensp;&ensp;&ensp;&ensp;从本质上去看，牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快。如果更通俗地说的话，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以，可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。（牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，没有全局思想。）    
（2）拟牛顿法（Quasi-Newton Methods）  
&ensp;&ensp;&ensp;&ensp;拟牛顿法的本质思想是改善牛顿法每次需要求解复杂的Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂度。拟牛顿法和最速下降法一样只要求每一步迭代时知道目标函数的梯度。通过测量梯度的变化，构造一个目标函数的模型使之足以产生超线性收敛性。这类方法大大优于最速下降法，尤其对于困难的问题。另外，因为拟牛顿法不需要二阶导数的信息，所以有时比牛顿法更为有效。如今，优化软件中包含了大量的拟牛顿算法用来解决无约束，约束，和大规模的优化问题。  
（3）共轭梯度法（Conjugate Gradient）  
&ensp;&ensp;&ensp;&ensp;共轭梯度法是介于最速下降法与牛顿法之间的一个方法，它仅需利用一阶导数信息，但克服了最速下降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点，共轭梯度法不仅是解决大型线性方程组最有用的方法之一，也是解大型非线性最优化最有效的算法之一。 在各种优化算法中，共轭梯度法是非常重要的一种。其优点是所需存储量小，具有步收敛性，稳定性高，而且不需要任何外来参数。  
&ensp;&ensp;&ensp;&ensp;下图为共轭梯度法和梯度下降法搜索最优解的路径对比示意图：  
&ensp;&ensp;&ensp;&ensp;![](https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%85%B1%E8%BD%AD%E6%A2%AF%E5%BA%A6%E6%B3%95.png)    
&ensp;&ensp;&ensp;&ensp;注：绿色为梯度下降法，红色代表共轭梯度法  
（4）解决约束优化问题——拉格朗日乘数法
#### 4、原理总结
&ensp;&ensp;&ensp;&ensp;线性回归是回归问题中的一种，线性回归假设目标值与特征之间线性相关，即满足一个多元一次方程。使用最小二乘法构建损失函数，用梯度下降来求解损失函数最小时的θ值。

### 二、Ridge回归与Lasso回归  
&ensp;&ensp;&ensp;&ensp;当样本特征很多，样本数相对较少时，模型容易陷入过拟合。为了缓解过拟合问题，有两种方法：  
&ensp;&ensp;&ensp;&ensp;方法一：减少特征数量（人工选择重要特征来保留，会丢弃部分信息）。  
&ensp;&ensp;&ensp;&ensp;方法二：正则化（减少特征参数W的数量级）。
#### 1、Ridge回归 
- 定义：  
&ensp;&ensp;&ensp;&ensp;由于直接套用线性回归可能产生过拟合，我们需要加入正则化项，如果加入的是L2正则化项，就是Ridge回归，有时也翻译为脊回归。它和一般线性回归的区别是在损失函数上增加了一个L2正则化的项，和一个调节线性回归项和正则化项权重的系数α。  
- Ridge回归L2范数损失函数定义：
&ensp;&ensp;&ensp;&ensp;![岭回归L2损失函数](https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%B2%AD%E5%9B%9E%E5%BD%92-Lasso%E5%9B%9E%E5%BD%92/%E5%B2%AD%E5%9B%9E%E5%BD%92L2%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.gif)   
&ensp;&ensp;&ensp;&ensp;岭回归的代价函数仍然是一个凸函数，因此可以利用梯度等于0的方式求得全局最优解（正规方程）：
&ensp;&ensp;&ensp;&ensp;![岭回归L2正规方程](https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%B2%AD%E5%9B%9E%E5%BD%92-Lasso%E5%9B%9E%E5%BD%92/%E5%B2%AD%E5%9B%9E%E5%BD%92L2%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B1.png)  
&ensp;&ensp;&ensp;&ensp;上述正规方程与一般线性回归的正规方程相比，多了一项λI，其中II表示单位矩阵。假如XTX是一个奇异矩阵（不满秩），添加这一项后可以保证该项可逆。由于单位矩阵的形状是对角线上为1其他地方都为0，看起来像一条山岭，因此而得名。

#### 2、Lasso回归
#### 1、定义：  
&ensp;&ensp;&ensp;&ensp;Lasso回归有时也叫做线性回归的L1正则化，和Ridge回归的主要区别就是在正则化项，Ridge回归用的是L2正则化，而Lasso回归用的是L1正则化。
#### 2、lasso回归L1范数损失函数:
&ensp;&ensp;&ensp;&ensp;![Lasso回归L1损失函数](https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%B2%AD%E5%9B%9E%E5%BD%92-Lasso%E5%9B%9E%E5%BD%92/lasso%E5%9B%9E%E5%BD%92L1%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.gif)  
#### 3、lasso回归损失函数求最优解：  
&ensp;&ensp;&ensp;&ensp;Lasso回归使得一些系数变小，甚至还是一些绝对值较小的系数直接变为0，因此特别适用于参数数目缩减与参数的选择，因而用来估计稀疏参数的线性模型。  
&ensp;&ensp;&ensp;&ensp;但是Lasso回归有一个很大的问题，导致我们需要把它单独拎出来讲，就是它的损失函数不是连续可导的，由于L1范数用的是绝对值之和，导致损失函数有不可导的点。也就是说，我们 的最小二乘法，梯度下降法，牛顿法与拟牛顿法对它统统失效了。那我们怎么才能求有这个L1范数的损失函数极小值呢？  
两种全新的求极值解法：坐标轴下降法（coordinate descent）和最小角回归法（ Least Angle Regression， LARS）。　　　　　　　　　　
#### 4、为什么L1正则化更易获得“稀疏”解呢？  
&ensp;&ensp;&ensp;&ensp;假设仅有两个属性，W只有两个参数W1,W2，绘制不带正则项的目标函数-平方误差项等值线，再绘制L1,L2，范数等值线，如图正则化后优化目标的解要在平方误差项和正则化项之间折中，即出现在图中等值线相交处采用。L1范数时，交点常出现在坐标轴上，即或为0;而采用范数时，交点常出现在某个象限中，即w1,w2均非0。也就是说，L1范数比L2范数更易获得“稀疏”解。  
&ensp;&ensp;&ensp;&ensp;<img src="https://github.com/Lg-AiLearn/ML/blob/master/images/%E5%B2%AD%E5%9B%9E%E5%BD%92-Lasso%E5%9B%9E%E5%BD%92/L1-L2.png" height="330" width="380" >
#### 5、原理总结  
&ensp;&ensp;&ensp;&ensp;Lasso回归是在ridge回归的基础上发展起来的，如果模型的特征非常多，需要压缩，那么Lasso回归是很好的选择。一般的情况下，普通的线性回归模型就够了。

## 第三部分：分类算法
### 一、基本概念与性能度量
- 基本概念：  
&ensp;&ensp;&ensp;&ensp;**错误率(error rate)** ：分类错误的样本占样本总数的比例。或者说是预测错误(包括将正例预测为反例和将反例预测为正例)的比例。  
&ensp;&ensp;&ensp;&ensp;**精度(accuracy)** ：分类正确的样本占样本的总数。   
&ensp;&ensp;&ensp;&ensp;**误差(error)** ：学习器实际预测输出与样本的真实输出之间的差异。   
&ensp;&ensp;&ensp;&ensp;**训练误差(training error)** ：学习器在训练集上的误差。   
&ensp;&ensp;&ensp;&ensp;**泛化误差(generalization error)** ：学习器在新样本上的误差。   
&ensp;&ensp;&ensp;&ensp;泛化误差可以分解为偏差、方差与噪声之和，偏差-方差分解（bias-variance decomposition）是解释泛化性能的重要工具。  
&ensp;&ensp;&ensp;&ensp;yD：测试样本x在数据集中的标记；   
&ensp;&ensp;&ensp;&ensp;y：测试样本x的真实标记；   
&ensp;&ensp;&ensp;&ensp;f(x;D)：训练集D上学得模型f在x上的预测输出；   
&ensp;&ensp;&ensp;&ensp;那么我们可以知道模型的**期望预测值为** ：  
&ensp;&ensp;&ensp;&ensp;![模型的期望预测值](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/模型的期望预测值.png)  
&ensp;&ensp;&ensp;&ensp;**偏差**度量了算法的期望预测与真实结果的偏离程度。偏差描述的是平均预测结果与真实结果之间的差距，即其描述了模型对于数据的拟合程度，偏差越小，说明模型对数据拟合的越好 。偏差的度量：  
&ensp;&ensp;&ensp;&ensp;![偏差](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/偏差.png)  
&ensp;&ensp;&ensp;&ensp;**方差**度量了同样大小训练集变动导致的性能变化。方差描述的是预测结果的稳定性，即数据集的变化对于预测结果的影响，同样也度量了数据集变化对于模型学习性能的变化，方差越小，说明我们的模型对于数据集的变化越不敏感，也就是对于新数据集的学习越稳定。方差的度量：  
&ensp;&ensp;&ensp;&ensp;![方差](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/方差.png)  
&ensp;&ensp;&ensp;&ensp;**噪声**涉及问题本身的难度。噪声描述的是数据集中样本的标记与这些样本真实标记的距离，只要是数据，都一定拥有噪声，因此我们可以认为噪声误差为模型训练的下限，因为这部分误差无论模型怎么学习都不可能消除，这是数据带来的本源误差。噪音的度量：  
&ensp;&ensp;&ensp;&ensp;![噪音](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/噪音.png)  
&ensp;&ensp;&ensp;&ensp;综上，由此可见：由方差、偏差、噪声、泛化误差的公式可以看出，偏差度量了模型预测的期望值与真实值之间的偏离程度，刻画了模型本身的拟合能力；方差度量了训练集的变动对预测结果的影响；噪声表达了能达到的期望误差的下界，刻画了学习问题本身的难度。但偏差与方差是有冲突的，即**偏差-方差窘境（bias-variance dilemma）**。如图：  
&ensp;&ensp;&ensp;&ensp;![偏差-方差窘境](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/偏差-方差窘境.png)  
&ensp;&ensp;&ensp;&ensp;在训练程度不足时，学习器拟合程度不强，训练数据的扰动不足以产生显著变化，此时偏差主导泛化错误率。  
&ensp;&ensp;&ensp;&ensp;随着训练程度加深，学习器拟合能力增强，训练数据的扰动逐渐可以被学习器学到，方差逐渐主导泛化错误率。  
&ensp;&ensp;&ensp;&ensp;如果继续加深训练，则有可能发生过拟合。  
&ensp;&ensp;&ensp;&ensp;**过拟合(over fitting)** :学习器在训练集上表现很好却泛化能力表现差。一般是学习器把训练样本的一些特殊性质当作了总体样本的一般性质进行了学习。   
&ensp;&ensp;&ensp;&ensp;**欠拟合(under fitting)** : 学习器未能学的训练集样本的一般性质，对训练集的表现不好。  
&ensp;&ensp;&ensp;&ensp;**查准率(precision)** ：预测结果中预测为正例的结果中真正为正例的比例。  
&ensp;&ensp;&ensp;&ensp;**查全率(recall)** ：预测结果中预测为正例中真正正例占全部正例的比例。

- 性能度量  
&ensp;&ensp;&ensp;&ensp;!用实验的方法可以评估模型的性能，但还是需要一个直接的评价标准，这就是性能度量。性能度量往往多方面有关：算法、数据、任务。回归任务中常用的是均方误差；分类任务中常用的是错误率和精度。  
&ensp;&ensp;&ensp;&ensp;针对预测值和真实值之间的关系，我们可以将样本分为四个部分，分别是：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**真正例（True Positive，TP）** ：预测值和真实值都为1  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**假正例（False Positive，FP）** ：预测值为1，真实值为0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**真负例（True Negative，TN）** :预测值与真实值都为0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**假负例（False Negative，FN）** ：预测值为0，真实值为1  
&ensp;&ensp;&ensp;&ensp;分类结果混淆矩阵（confusion matrix）如图所示：  
&ensp;&ensp;&ensp;&ensp;![混淆矩阵](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/混淆矩阵.png)  
&ensp;&ensp;&ensp;&ensp;注：降低阀值，提高Recall；提高阀值，提高Precision  
1、**均方误差**  
&ensp;&ensp;&ensp;&ensp;![均方误差](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/均方误差.png)  
&ensp;&ensp;&ensp;&ensp;将其进行推广为：  
&ensp;&ensp;&ensp;&ensp;![均方误差推广](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/均方误差推广.png)  
2、**错误率**  
&ensp;&ensp;&ensp;&ensp;![错误率](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/错误率.png)  
3、**精度**  
&ensp;&ensp;&ensp;&ensp;![精度](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/精度.png)  
4、**查准率**  
&ensp;&ensp;&ensp;&ensp;![查准率](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/查准率.png)  
5、**查全率**   
&ensp;&ensp;&ensp;&ensp;![查全率](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/查全率.png)  
&ensp;&ensp;&ensp;&ensp;注：下图为P-R图  
&ensp;&ensp;&ensp;&ensp;![P-R图](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/P-R图.png)  
6、**F1**  
&ensp;&ensp;&ensp;&ensp;![F1](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/F1.png)  
7、**ROC**  
&ensp;&ensp;&ensp;&ensp;受试者工作特征曲线 （receiver operating characteristic curve，简称ROC曲线），又称为感受性曲线（sensitivity curve）。  
&ensp;&ensp;&ensp;&ensp;横坐标：Sensitivity，伪正类率(False positive rate， FPR)，预测为正但实际为负的样本占所有负例样本的比例；  
&ensp;&ensp;&ensp;&ensp;纵坐标：1-Specificity，真正类率(True positive rate， TPR)，预测为正且实际为正的样本占所有正例样本 的比例。（相当于召回率recall）  
&ensp;&ensp;&ensp;&ensp;![ROC](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/ROC.png) 
8、**AUC**  
&ensp;&ensp;&ensp;&ensp;AUC (Area Under Curve) 被定义为 ROC 曲线下的面积，显然这个面积的数值不会大于 1。又由于 ROC 曲线一般都处于 y=x 这条直线的上方，所以 AUC 的取值范围一般在 0.5 和 1 之间。使用 AUC 值作为评价标准是因为很多时候 ROC 曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应 AUC 更大的分类器效果更好。


### 二、决策树
#### 代码实现（自定义）
* [代码实现（自定义）](/DecisionTree/DecisionTree.py)
#### 代码实现（scikit-learn库）
* [代码实现（scikit-learn库）](/DecisionTree/DecisionTree_scikit-learn.py)
- &ensp;&ensp;&ensp;&ensp;决策树算法是一类常见的分类和回归算法，顾名思义，决策树是基于树的结构来进行决策的。决策树又称为判定树，是运用于分类的一种树结构，其中的每个内部节点代表对某一属性的一次测试，每条边代表一个测试结果，叶节点代表某个类或类的分布。决策树的决策过程需要从决策树的根节点开始，待测数据与决策树中的特征节点进行比较，并按照比较结果选择选择下一比较分支，直到叶子节点作为最终的决策结果。  
&ensp;&ensp;&ensp;&ensp;决策树学习通常包括 3 个步骤：特征选择、决策树的生成和决策树的修剪。  
&ensp;&ensp;&ensp;&ensp;目前建立决策树有三种主要算法：ID3、C4.5以及CART。
#### 1、信息论  
- **信息熵**  
&ensp;&ensp;&ensp;&ensp;在决策树算法中，熵是一个非常重要的概念。一件事发生的概率越小，我们说它所蕴含的信息量越大。  
&ensp;&ensp;&ensp;&ensp;所以我们这样衡量信息量：  
&ensp;&ensp;&ensp;&ensp;![衡量信息量](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/衡量信息量.png)  
&ensp;&ensp;&ensp;&ensp;其中，P(y)是事件发生的概率。  
&ensp;&ensp;&ensp;&ensp;信息熵：是所有可能发生的事件的信息量的期望：  
&ensp;&ensp;&ensp;&ensp;![信息熵](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/信息熵.png)  
&ensp;&ensp;&ensp;&ensp;表达了Y事件发生的不确定度。  
- **条件熵**  
&ensp;&ensp;&ensp;&ensp;条件熵：表示在X给定条件下，Y的条件概率分布的熵对X的数学期望。其数学推导如下：  
&ensp;&ensp;&ensp;&ensp;![条件熵](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/条件熵.png)  
&ensp;&ensp;&ensp;&ensp;条件熵H（Y|X）表示在已知随机变量X的条件下随机变量Y的不确定性。注意一下，条件熵中X也是一个变量，意思是在一个变量X的条件下（变量X的每个值都会取到），另一个变量Y的熵对X的期望。  
举个例子：  
&ensp;&ensp;&ensp;&ensp;例：女生决定主不主动追一个男生的标准有两个：颜值和身高，如下表所示：  
&ensp;&ensp;&ensp;&ensp;![举个例子](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/举个例子.png)  
&ensp;&ensp;&ensp;&ensp;上表中随机变量Y=｛追，不追｝，P(Y=追)=2/3，P(Y=不追)=1/3，得到Y的熵：  
&ensp;&ensp;&ensp;&ensp;![Y的熵](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/Y的熵.png)  
&ensp;&ensp;&ensp;&ensp;这里还有一个特征变量X,X=｛高，不高｝。当X=高时，追的个数为1，占1/2，不追的个数为1，占1/2，此时：  
&ensp;&ensp;&ensp;&ensp;![Y的熵-1](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/Y的熵-1.png)  
&ensp;&ensp;&ensp;&ensp;同理  
&ensp;&ensp;&ensp;&ensp;![Y的熵-同理](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/Y的熵-同理.png)  
&ensp;&ensp;&ensp;&ensp;(注意：我们一般约定，当p=0时，plogp=0)  
&ensp;&ensp;&ensp;&ensp;所以我们得到条件熵的计算公式：  
&ensp;&ensp;&ensp;&ensp;![案例条件熵](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/案例条件熵.png)  
- **信息增益**  
&ensp;&ensp;&ensp;&ensp;当我们用另一个变量X对原变量Y分类后，原变量Y的不确定性就会减小了(即熵值减小)。而熵就是不确定性，不确定程度减少了多少其实就是信息增益。这就是信息增益的由来，所以信息增益定义如下：  
&ensp;&ensp;&ensp;&ensp;![信息增益](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/信息增益.png)  
&ensp;&ensp;&ensp;&ensp;此外，信息论中还有互信息、交叉熵等概念，它们与本算法关系不大，这里不展开。

#### 2、决策树算法简介  
&ensp;&ensp;&ensp;&ensp;以二分类为例，我们希望从给定训练集中学得一个模型来对新的样例进行分类。  
&ensp;&ensp;&ensp;&ensp;举个例子,有一个划分是不是鸟类的数据集合，如下：  
&ensp;&ensp;&ensp;&ensp;![鸟类的数据集合](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/鸟类的数据集合.png)  
&ensp;&ensp;&ensp;&ensp;这时候我们建立这样一颗决策树：  
&ensp;&ensp;&ensp;&ensp;![一颗决策树](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/一颗决策树.png)  
&ensp;&ensp;&ensp;&ensp;当我们有了一组新的数据时，我们就可以根据这个决策树判断出是不是鸟类。创建决策树的伪代码如下：  
&ensp;&ensp;&ensp;&ensp;![决策树的伪代码](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/决策树的伪代码.png)  
&ensp;&ensp;&ensp;&ensp;生成决策树是一个递归的过程，在决策树算法中，出现下列三种情况时，导致递归返回：  
&ensp;&ensp;&ensp;&ensp;(1)当前节点包含的样本属于同一种类，无需划分；  
&ensp;&ensp;&ensp;&ensp;(2)当前属性集合为空，或者所有样本在所有属性上取值相同，无法划分；  
&ensp;&ensp;&ensp;&ensp;(3)当前节点包含的样本集合为空，无法划分。
#### 3、属性选择  
&ensp;&ensp;&ensp;&ensp;在决策树算法中，最重要的就是划分属性的选择，即我们选择哪一个属性来进行划分。三种划分属性的主要算法是：ID3、C4.5以及CART。  
- **ID3算法**  
&ensp;&ensp;&ensp;&ensp;ID3算法所采用的度量标准就是我们前面所提到的“信息增益”。当属性a的信息增益最大时，则意味着用a属性划分，其所获得的“纯度”提升最大。我们所要做的，就是找到信息增益最大的属性。由于前面已经强调了信息增益的概念，这里不再赘述。  
- **C4.5算法**  
&ensp;&ensp;&ensp;&ensp;实际上，信息增益准则对于可取值数目较多的属性会有所偏好，为了减少这种偏好可能带来的不利影响，C4.5决策树算法不直接使用信息增益，而是使用“信息增益率”来选择最优划分属性，信息增益率定义为:  
&ensp;&ensp;&ensp;&ensp;![四五信息增益率](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/四五信息增益率.png)  
&ensp;&ensp;&ensp;&ensp;其中，分子为信息增益，分母为属性X的熵。  
&ensp;&ensp;&ensp;&ensp;需要注意的是，增益率准则对可取值数目较少的属性有所偏好。  
&ensp;&ensp;&ensp;&ensp;所以一般这样选取划分属性：先从候选属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的。  
- **CART算法**  
&ensp;&ensp;&ensp;&ensp;ID3算法和C4.5算法主要存在三个问题：  
&ensp;&ensp;&ensp;&ensp;(1)每次选取最佳特征来分割数据，并按照该特征的所有取值来进行划分。也就是说，如果一个特征有4种取值，那么数据就将被切成4份，一旦特征被切分后，该特征就不&ensp;&ensp;&ensp;&ensp;会再起作用，有观点认为这种切分方式过于迅速。  
&ensp;&ensp;&ensp;&ensp;(2)它们不能处理连续型特征。只有事先将连续型特征转换为离散型，才能在上述算法中使用。  
&ensp;&ensp;&ensp;&ensp;(3)会产生过拟合问题。  
&ensp;&ensp;&ensp;&ensp;为了解决上述(1)、(2)问题，产生了CART算法，它主要的衡量指标是基尼系数。为了解决问题(3)，主要采用剪枝技术和随机森林算法。  
&ensp;&ensp;&ensp;&ensp;为什么同样作为建立决策树的三种算法之一，我们要将CART算法单独拿出来讲。因为ID3算法和C4.5算法采用了较为复杂的熵来度量，所以它们只能处理分类问题。  
&ensp;&ensp;&ensp;&ensp;CART算法既能处理分类问题，又能处理回归问题。对于分类树，CART采用基尼指数最小化准则；对于回归树，CART采用平方误差最小化准则。  
<3.1>**CART分类树**  
&ensp;&ensp;&ensp;&ensp;CART分类树与上一节讲述的ID3算法和C4.5算法在原理部分差别不大，唯一的区别在于划分属性的原则。CART选择“基尼指数”作为划分属性的选择。  
&ensp;&ensp;&ensp;&ensp;Gini指数作为一种做特征选择的方式，其表征了特征的不纯度。  
&ensp;&ensp;&ensp;&ensp;在具体的分类问题中，对于数据集D，我们假设有K个类别，每个类别出现的概率为Pk，则数据集D的基尼指数的表达式为：  
&ensp;&ensp;&ensp;&ensp;![数据集D的基尼指数的表达式](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/数据集D的基尼指数的表达式.png)    
&ensp;&ensp;&ensp;&ensp;我们取一个极端情况，如果数据集合中的类别只有一类，那么：  
&ensp;&ensp;&ensp;&ensp;![数据集合中的类别只有一类](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/数据集合中的类别只有一类.png)    
&ensp;&ensp;&ensp;&ensp;我们发现，当只有一类时，数据的不纯度是最低的，所以Gini指数等于零。Gini(D)越小，则数据集D的纯度越高。  
&ensp;&ensp;&ensp;&ensp;特别地，对于样本D，如果我们选择特征A的某个值a，把D分成了D1和D2两部分，则此时，Gini指数为：  
&ensp;&ensp;&ensp;&ensp;![Gini指数](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/Gini指数.png)  
&ensp;&ensp;&ensp;&ensp;与信息增益类似，我们可以计算如下表达式：  
&ensp;&ensp;&ensp;&ensp;![变更的Gini指数](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/变更的Gini指数.png)  
&ensp;&ensp;&ensp;&ensp;即以特征A划分后，数据不纯度减小的程度。显然，我们在做特征选取时，应该选择最大的一个。  
<3.2>**CART回归树**  
&ensp;&ensp;&ensp;&ensp;首先要明白回归树与分类树的区别。如果决策树最终的输出是离散值，那么就是分类树；如果最终的输出是连续值，那么就是回归树。  
&ensp;&ensp;&ensp;&ensp;在之前做分类树的时候，用了熵、Gini指数等指标衡量了离散数据的混乱度。那我们用什么来衡量连续数据的混乱度呢？很简单，采用平方误差准则。首先我们计算所有&ensp;&ensp;&ensp;&ensp;数据的均值，然后计算每条数据的值到均值的差值，平方后求和：  
&ensp;&ensp;&ensp;&ensp;![数据的均值平方后求和](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/数据的均值平方后求和.png)  
&ensp;&ensp;&ensp;&ensp;其中，一共有K条数据，均值为c，第i条数据的值为yi。  
&ensp;&ensp;&ensp;&ensp;那我们怎么对输入空间进行划分呢？  
&ensp;&ensp;&ensp;&ensp;选择第j个特征和它的取值s作为切分变量和切分点，并划分为两个区域：  
&ensp;&ensp;&ensp;&ensp;![对输入空间进行划分两区域](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/对输入空间进行划分两区域.png)  
&ensp;&ensp;&ensp;&ensp;然后寻找最优切分变量j和最优切分点s，使得：  
&ensp;&ensp;&ensp;&ensp;![对输入空间进行划分最优切分点](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/对输入空间进行划分最优切分点.png)  
&ensp;&ensp;&ensp;&ensp;其中，yi为R1区域中第i条数据的目标值，c1为R1区域中目标的均值。  
&ensp;&ensp;&ensp;&ensp;伪代码如下：  
&ensp;&ensp;&ensp;&ensp;![切分伪代码](https://github.com/Lg-AiLearn/ML/blob/master/images/decisiontree/切分伪代码.png)  
&ensp;&ensp;&ensp;&ensp;进行递归之后，我们的回归树就建好了。回归树建好后，我们采用的是用最终叶子节点的均值或者中位数作为预测输出结果。  
&ensp;&ensp;&ensp;&ensp;注：用树来对数据进行建模，除了把叶子节点简单设为常数值之外，我们也可以把叶节点设为分段线性函数，这样就建立了模型树。由于其原理和回归树相差不大。
#### 4、停止条件  
&ensp;&ensp;&ensp;&ensp;决策树的构建过程是一个递归的过程，所以需要确定停止条件，否则过程将不会结束。一种最直观的方式是当每个子节点只有一种类型的记录时停止，但是这样往往会使得树的节点过多，导致过拟合问题（Overfitting）。另一种可行的方法是当前节点中的记录数低于一个最小的阀值，那么就停止分割，将max(P(i))对应的分类作为当前叶节点的分类。
#### 5、剪枝与随机森林  
&ensp;&ensp;&ensp;&ensp;决策树算法中很容易发生“过拟合”现象，导致算法的泛化能力不强。目前，解决决策树“过拟合”现象的主要方法有两种：1.剪枝技术，2.随机森林算法。  
- **剪枝**  
&ensp;&ensp;&ensp;&ensp;剪枝技术分为“预剪枝”和“后剪枝”。预剪枝是指在决策树生成过程中，对每个节点在划分前先进行估计，若当前节点的划分不能带来决策树泛化能力的提升，则停止划分并将当前节点标记为叶子节点；后剪枝是先从训练集中生成一颗完整的决策树，然后自底向上地对非叶节点进行考察，若将该节点对应的子树替换为叶子节点能够带来决策树泛化能力的提升，则将该子树替换为叶子节点。  
&ensp;&ensp;&ensp;&ensp;以分类问题为例，上一节讲过，我们将叶子节点的类别标记为叶子节点中训练样例数最多的类别。  
&ensp;&ensp;&ensp;&ensp;那我们怎么判断决策树泛化能力是否提升呢？我们通常情况可预留一部分数据用作“验证集”，我们判断的依据就是剪枝前后这个节点在验证集上的分类正确率。如果剪枝后正确率上升，那么这个节点可以进行剪枝；反之，则不对该节点进行剪枝。注意一下，如果剪枝前后正确率不变，根据奥卡姆剃刀准则，我们一般情况下是选择进行剪枝操作，因为“如无必要，勿增实体”。  
&ensp;&ensp;&ensp;&ensp;剪枝技术小结：  
&ensp;&ensp;&ensp;&ensp;(1) 对于预剪枝来讲，它使得决策树的很多分支都没有展开，这不仅降低了过拟合的风险，还显著减少了决策树的训练和预测时间开销。但是另一方面，有些分支的当前划分虽然不能提升泛化性能、甚至可能导致泛化性能暂时下降，但是在其基础上的后续划分却有可能导致性能显著提高，这给预剪枝带来了欠拟合的风险；  
&ensp;&ensp;&ensp;&ensp;(2) 对于后剪枝来讲，通常情况下，它比预剪枝决策树保留了更多的分支。后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树，但是其训练时间开销比未剪枝决策树和预剪枝决策树大得多。  
- **随机森林**  
&ensp;&ensp;&ensp;&ensp;随机森林算法属于Bagging学习中的一种方法，可以解决决策树过拟合问题。它们的主要思想就是建立多个分类器(如决策树)，以降低过拟合的风险。  
<1>Bagging  
&ensp;&ensp;&ensp;&ensp;Bagging算法是并行式集成学习中最著名的代表。Bagging算法的基本流程如下：  
&ensp;&ensp;&ensp;&ensp;(1)从训练样本集中有重复地选中n个样本  
&ensp;&ensp;&ensp;&ensp;(2)在所有属性上，对n个样本建立一个分类器  
&ensp;&ensp;&ensp;&ensp;(3)重复以上两步m次，即可获得m个分类器  
&ensp;&ensp;&ensp;&ensp;(4)将测试数据放到m个分类器上，最后根据这m个分类器的投票结果，决定数据属于哪一类  
<2>随机森林算法  
&ensp;&ensp;&ensp;&ensp;随机森林算法是在Bagging基础上的扩展，进一步在决策树的训练过程中引入了随机属性选择。基本流程如下：  
&ensp;&ensp;&ensp;&ensp;(1)从训练样本集中有重复地选中n个样本  
&ensp;&ensp;&ensp;&ensp;(2)在所有属性中随机选取K个属性，选择最优划分属性，对n个样本建立一个分类器  
&ensp;&ensp;&ensp;&ensp;(3)重复以上两步m次，即可获得m个分类器  
&ensp;&ensp;&ensp;&ensp;(4)将测试数据放到m个分类器上，最后根据这m个分类器的投票结果，决定数据属于哪一类  
&ensp;&ensp;&ensp;&ensp;一般取k=logd，d是所有属性的个数，如果k=d，则Bagging算法和随机森林算法就是一样的。  
&ensp;&ensp;&ensp;&ensp;其实随机森林算法中的“随机”有两层含义：1.从训练样本集中有重复随机选中n个样本；2.每次在所有属性中随机选取k个属性。
#### 6、决策树优缺点  
&ensp;&ensp;&ensp;&ensp;决策树的优点：  
&ensp;&ensp;&ensp;&ensp;相对于其他数据挖掘算法，决策树在以下几个方面拥有优势：  
&ensp;&ensp;&ensp;&ensp;1.决策树易于理解和实现，人们在通过解释后都有能力去理解决策树所表达的意义;  
&ensp;&ensp;&ensp;&ensp;2.对于决策树，数据的准备往往是简单或者是不必要的，其他的技术往往要求先把数据一般化，比如去掉多余的或者空白的属性;  
&ensp;&ensp;&ensp;&ensp;3.能够同时处理数据型和常规型属性。其他的技术往往要求数据属性的单一;  
&ensp;&ensp;&ensp;&ensp;4.在相对短的时间内能够对大型数据源做出可行且效果良好的结果;  
&ensp;&ensp;&ensp;&ensp;5.对缺失值不敏感;  
&ensp;&ensp;&ensp;&ensp;6.可以处理不相关特征数据;  
&ensp;&ensp;&ensp;&ensp;7.效率高，决策树只需要一次构建，反复使用，每一次预测的最大计算次数不超过决策树的深度。  
&ensp;&ensp;&ensp;&ensp;决策树的缺点：  
&ensp;&ensp;&ensp;&ensp;1.对连续性的字段比较难预测。  
&ensp;&ensp;&ensp;&ensp;2.对有时间顺序的数据，需要很多预处理的工作。  
&ensp;&ensp;&ensp;&ensp;3.当类别太多时，错误可能就会增加的比较快。  
&ensp;&ensp;&ensp;&ensp;4.一般的算法分类的时候，只是根据一个字段来分类。  
&ensp;&ensp;&ensp;&ensp;5.在处理特征关联性比较强的数据时表现得不是太好


### 三、随机森林
#### 代码实现（自定义）
* [代码实现（自定义）](/RandomRorest/RandomRorest.py)
#### 代码实现（scikit-learn库）
* [代码实现（scikit-learn库）](/RandomRorest/RandomRorest_scikit-learn.py)  
- &ensp;&ensp;&ensp;&ensp;集成学习有两个流派，一个是boosting派系，它的特点是各个弱学习器之间有依赖关系。另一种是bagging流派，它的特点是各个弱学习器之间没有依赖关系，可以并行拟合。  
&ensp;&ensp;&ensp;&ensp;神经网络预测精确，但是计算量很大。随机森林在运算量没有显著提高的前提下提高了预测精度。  
&ensp;&ensp;&ensp;&ensp;随机森林顾名思义，是用随机的方式建立一个森林，森林里面有很多的决策树组成，随机森林的每一棵决策树之间是没有关联的。在得到森林之后，当有一个新的输入样本进入的时候，就让森林中的每一棵决策树分别进行一下判断，看看这个样本应该属于哪一类（对于分类算法），然后看看哪一类被选择最多，就预测这个样本为那一类。
#### 1、随机森林算法原理：  
&ensp;&ensp;&ensp;&ensp;随机森林是从原始训练样本集N中有放回地重复随机抽取k个样本生成新的训练样本集合，然后根据自助样本集生成k个分类树组成随机森林，新数据的分类结果按分类树投票多少形成的分数而定。其实质是对决策树算法的一种改进，将多个决策树合并在一起，每棵树的建立依赖于一个独立抽取的样品，森林中的每棵树具有相同的分布，分类误差取决于每一棵树的分类能力和它们之间的相关性。特征选择采用随机的方法去分裂每一个节点，然后比较不同情况下产生的误差。能够检测到的内在估计误差、分类能力和相关性决定选择特征的数目。单棵树的分类能力可能很小，但在随机产生大量的决策树后，一个测试样品可以通过每一棵树的分类结果经统计后选择最可能的分类。  
&ensp;&ensp;&ensp;&ensp;在建立每一棵决策树的过程中，有两点需要注意采样与完全分裂。首先是两个随机采样的过程，random forest对输入的数据要进行行、列的采样。对于行采样，采用有放回的方式，也就是在采样得到的样本集合中，可能有重复的样本。假设输入样本为N个，那么采样的样本也为N个。这样使得在训练的时候，每一棵树的输入样本都不是全部的样本，使得相对不容易出现over-fitting。然后进行列采样，从M个feature中，选择m个（m < M）。之后就是对采样之后的数据使用完全分裂的方式建立出决策树，这样决策树的某一个叶子节点要么是无法继续分裂的，要么里面的所有样本的都是指向的同一个分类。一般很多的决策树算法都有一个重要的步骤——剪枝，但是这里不这样干，由于之前的两个随机采样的过程保证了随机性，所以就算不剪枝，也不会出现over-fitting。
#### 2、关于随机  
&ensp;&ensp;&ensp;&ensp;（1）训练每棵树时，从全部训练样本中选取一个子集进行训练（即bootstrap取样）。用剩余的数据进行评测，评估其误差；  
&ensp;&ensp;&ensp;&ensp;（2）在每个节点，随机选取所有特征的一个子集，用来计算最佳分割方式。
#### 3、算法流程：  
&ensp;&ensp;&ensp;&ensp;（1）训练总样本的个数为N，则单棵决策树从N个训练集中有放回的随机抽取n个作为此单颗树的训练样本（bootstrap有放回取样）。  
&ensp;&ensp;&ensp;&ensp;（2）令训练样例的输入特征的个数为M，m远远小于M，则我们在每颗决策树的每个节点上进行分裂时，从M个输入特征里随机选择m个输入特征，然后从这m个输入特征里选择一个最好的进行分裂。m在构建决策树的过程中不会改变。  
&ensp;&ensp;&ensp;&ensp;注意：要为每个节点随机选出m个特征，然后选择最好的那个特征来分裂。  
&ensp;&ensp;&ensp;&ensp;注意：决策树中分裂属性的两个选择度量：信息增益和基尼指数。  
&ensp;&ensp;&ensp;&ensp;（3）每棵树都一直这样分裂下去，直到该节点的所有训练样例都属于同一类，不需要剪枝。由于之前的两个随机采样的过程保证了随机性，所以就算不剪枝，也不会出现over-fitting。
#### 4、结果判定：  
&ensp;&ensp;&ensp;&ensp;（1）目标特征为数字类型：取t个决策树的平均值作为分类结果。  
&ensp;&ensp;&ensp;&ensp;（2）目标特征为类别类型：少数服从多数，取单棵树分类结果最多的那个类别作为整个随机森林的分类结果。
#### 5、预测：  
&ensp;&ensp;&ensp;&ensp;随机森林是用随机的方式建立一个森林，森林里面有很多的决策树组成，随机森林的每一棵决策树之间是没有关联的。在得到森林之后，当有一个新的输入样本进入的时候，就让森林中的每一棵决策树分别进行一下判断，看看这个样本应该属于哪一类，然后看看哪一类被选择最多，就预测这个样本为那一类。  
&ensp;&ensp;&ensp;&ensp;说明：通过bagging有放回取样后，大约36.8%的没有被采样到的数据，我们常常称之为袋外数据。这些数据没有参与训练集模型的拟合，因此可以用来检测模型的泛化能力。
#### 6、RF优缺点  
&ensp;&ensp;&ensp;&ensp;RF的主要优点有：  
&ensp;&ensp;&ensp;&ensp;（1） 训练可以高度并行化，对于大数据时代的大样本训练速度有优势。  
&ensp;&ensp;&ensp;&ensp;（2） 由于可以随机选择决策树节点划分特征，这样在样本特征维度很高的时候，仍然能高效的训练模型。  
&ensp;&ensp;&ensp;&ensp;（3） 在训练后，可以给出各个特征对于输出的重要性。  
&ensp;&ensp;&ensp;&ensp;（4） 由于采用了随机采样，训练出的模型的方差小，泛化能力强。  
&ensp;&ensp;&ensp;&ensp;（5） 相对于Boosting系列的Adaboost和GBDT，RF实现比较简单。  
&ensp;&ensp;&ensp;&ensp;（6） 对部分特征缺失不敏感。  
&ensp;&ensp;&ensp;&ensp;RF的主要缺点有：  
&ensp;&ensp;&ensp;&ensp;（1）在某些噪音比较大的样本集上，RF模型容易陷入过拟合。  
&ensp;&ensp;&ensp;&ensp;（2) 取值划分比较多的特征容易对RF的决策产生更大的影响，从而影响拟合的模型的效果。


### 四、xgboost
- &ensp;&ensp;&ensp;&ensp;XGBoost是boosting算法的其中一种。Boosting算法的思想是将许多弱分类器集成在一起形成一个强分类器。因为XGBoost是一种提升树模型，所以它是将许多树模型集成在一起，形成一个很强的分类器。而所用到的树模型则是CART回归树模型。讲解其原理前，先讲解一下CART回归树。
#### 1、CART回归树  
&ensp;&ensp;&ensp;&ensp;CART回归树是假设树为二叉树，通过不断将特征进行分裂。比如当前树结点是基于第j个特征值进行分裂的，设该特征值小于s的样本划分为左子树，大于s的样本划分为右子树。  
&ensp;&ensp;&ensp;&ensp;![CART回归树分裂](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/CART回归树分裂.png)  
&ensp;&ensp;&ensp;&ensp;而CART回归树实质上就是在该特征维度对样本空间进行划分，而这种空间划分的优化是一种NP难问题，因此，在决策树模型中是使用启发式方法解决。典型CART回归树产生的目标函数为：  
&ensp;&ensp;&ensp;&ensp;![CART回归树产生的目标函数](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/CART回归树产生的目标函数.png)  
&ensp;&ensp;&ensp;&ensp;因此，当我们为了求解最优的切分特征j和最优的切分点s，就转化为求解这么一个目标函数：  
&ensp;&ensp;&ensp;&ensp;![CART回归树产生的目标函数求解](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/CART回归树产生的目标函数求解.png)  
所以我们只要遍历所有特征的的所有切分点，就能找到最优的切分特征和切分点。最终得到一棵回归树。
#### 2、XGBoost算法思想  
&ensp;&ensp;&ensp;&ensp;该算法思想就是不断地添加树，不断地进行特征分裂来生长一棵树，每次添加一个树，其实是学习一个新函数，去拟合上次预测的残差。当我们训练完成得到k棵树，我们要预测一个样本的分数，其实就是根据这个样本的特征，在每棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数，最后只需要将每棵树对应的分数加起来就是该样本的预测值。  
&ensp;&ensp;&ensp;&ensp;![样本预测值](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/样本预测值.png)  
&ensp;&ensp;&ensp;&ensp;注：w_q(x)为叶子节点q的分数，f(x)为其中一棵回归树。  
&ensp;&ensp;&ensp;&ensp;如下图例子，训练出了2棵决策树，小孩的预测分数就是两棵树中小孩所落到的结点的分数相加。爷爷的预测分数同理。  
&ensp;&ensp;&ensp;&ensp;![案例图](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/案例图.png)
#### 3、XGBoost原理  
&ensp;&ensp;&ensp;&ensp;XGBoost目标函数定义为：  
&ensp;&ensp;&ensp;&ensp;![XGBoost目标函数](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/XGBoost目标函数.png)  
&ensp;&ensp;&ensp;&ensp;![XGBoost目标函数-2](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/XGBoost目标函数-2.png)  
&ensp;&ensp;&ensp;&ensp;目标函数由两部分构成，第一部分用来衡量预测分数和真实分数的差距，另一部分则是正则化项。正则化项同样包含两部分，T表示叶子结点的个数，w表示叶子节点的分数。γ可以控制叶子结点的个数，λ可以控制叶子节点的分数不会过大，防止过拟合。  
&ensp;&ensp;&ensp;&ensp;正如上文所说，新生成的树是要拟合上次预测的残差的，即当生成t棵树后，预测分数可以写成：  
&ensp;&ensp;&ensp;&ensp;![预测分数](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/预测分数.png)  
&ensp;&ensp;&ensp;&ensp;同时，可以将目标函数改写成：  
&ensp;&ensp;&ensp;&ensp;![目标函数改写成](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/目标函数改写成.png)  
&ensp;&ensp;&ensp;&ensp;很明显，我们接下来就是要去找到一个f_t能够最小化目标函数。XGBoost的想法是利用其在f_t=0处的泰勒二阶展开近似它。所以，目标函数近似为：  
&ensp;&ensp;&ensp;&ensp;![目标函数近似为](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/目标函数近似为.png)  
&ensp;&ensp;&ensp;&ensp;其中g_i为一阶导数，h_i为二阶导数：  
&ensp;&ensp;&ensp;&ensp;![二阶导数](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/二阶导数.png)  
&ensp;&ensp;&ensp;&ensp;由于前t-1棵树的预测分数与y的残差对目标函数优化不影响，可以直接去掉。简化目标函数为：  
&ensp;&ensp;&ensp;&ensp;![简化目标函数为](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/简化目标函数为.png)  
&ensp;&ensp;&ensp;&ensp;上式是将每个样本的损失函数值加起来，我们知道，每个样本都最终会落到一个叶子结点中，所以我们可以将所以同一个叶子结点的样本重组起来，过程如下图：  
&ensp;&ensp;&ensp;&ensp;![损失函数值加起来](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/损失函数值加起来.png)  
&ensp;&ensp;&ensp;&ensp;因此通过上式的改写，我们可以将目标函数改写成关于叶子结点分数w的一个一元二次函数，求解最优的w和目标函数值就变得很简单了，直接使用顶点公式即可。因此，最优的w和目标函数公式为：  
&ensp;&ensp;&ensp;&ensp;![最优的w和目标函数公式](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/最优的w和目标函数公式.png)

#### 4、分裂结点算法  
&ensp;&ensp;&ensp;&ensp;在上面的推导中，我们知道了如果我们一棵树的结构确定了，如何求得每个叶子结点的分数。但我们还没介绍如何确定树结构，即每次特征分裂怎么寻找最佳特征，怎么寻找最佳分裂点。  
&ensp;&ensp;&ensp;&ensp;正如上文说到，基于空间切分去构造一颗决策树是一个NP难问题，我们不可能去遍历所有树结构，因此，XGBoost使用了和CART回归树一样的想法，利用贪婪算法，遍历所有特征的所有特征划分点，不同的是使用上式目标函数值作为评价函数。具体做法就是分裂后的目标函数值比单子叶子节点的目标函数的增益，同时为了限制树生长过深，还加了个阈值，只有当增益大于该阈值才进行分裂。  
&ensp;&ensp;&ensp;&ensp;同时可以设置树的最大深度、当样本权重和小于设定阈值时停止生长去防止过拟合。
#### 5、Shrinkage and Column Subsampling  
&ensp;&ensp;&ensp;&ensp;XGBoost还提出了两种防止过拟合的方法：Shrinkage and Column Subsampling。Shrinkage方法就是在每次迭代中对树的每个叶子结点的分数乘上一个缩减权重η，这可以使得每一棵树的影响力不会太大，留下更大的空间给后面生成的树去优化模型。Column Subsampling类似于随机森林中的选取部分特征进行建树。其可分为两种，一种是按层随机采样，在对同一层内每个结点分裂之前，先随机选择一部分特征，然后只需要遍历这部分的特征，来确定最优的分割点。另一种是随机选择特征，则建树前随机选择一部分特征然后分裂就只遍历这些特征。一般情况下前者效果更好。
#### 6、近似算法  
&ensp;&ensp;&ensp;&ensp;对于连续型特征值，当样本数量非常大，该特征取值过多时，遍历所有取值会花费很多时间，且容易过拟合。因此XGBoost思想是对特征进行分桶，即找到l个划分点，将位于相邻分位点之间的样本分在一个桶中。在遍历该特征的时候，只需要遍历各个分位点，从而计算最优划分。从算法伪代码中该流程还可以分为两种，全局的近似是在新生成一棵树之前就对各个特征计算分位点并划分样本，之后在每次分裂过程中都采用近似划分，而局部近似就是在具体的某一次分裂节点的过程中采用近似算法。  
&ensp;&ensp;&ensp;&ensp;![近似算法](https://github.com/Lg-AiLearn/ML/blob/master/images/xgboost/近似算法.png)
#### 7、针对稀疏数据的算法(缺失值处理)  
&ensp;&ensp;&ensp;&ensp;当样本的第i个特征值缺失时，无法利用该特征进行划分时，XGBoost的想法是将该样本分别划分到左结点和右结点，然后计算其增益，哪个大就划分到哪边。
#### 8、XGBoost的优点  
&ensp;&ensp;&ensp;&ensp;之所以XGBoost可以成为机器学习的大杀器，广泛用于数据科学竞赛和工业界，是因为它有许多优点：  
&ensp;&ensp;&ensp;&ensp;1.使用许多策略去防止过拟合，如：正则化项、Shrinkage and Column Subsampling等。  
&ensp;&ensp;&ensp;&ensp;2. 目标函数优化利用了损失函数关于待求函数的二阶导数  
&ensp;&ensp;&ensp;&ensp;3.支持并行化，这是XGBoost的闪光点，虽然树与树之间是串行关系，但是同层级节点可并行。具体的对于某个节点，节点内选择最佳分裂点，候选分裂点计算增益用多线程并行。训练速度快。  
&ensp;&ensp;&ensp;&ensp;4.添加了对稀疏数据的处理。  
&ensp;&ensp;&ensp;&ensp;5.交叉验证，early stop，当预测结果已经很好的时候可以提前停止建树，加快训练速度。  
&ensp;&ensp;&ensp;&ensp;6.支持设置样本权重，该权重体现在一阶导数g和二阶导数h，通过调整权重可以去更加关注一些样本。


## 第四部分：聚类算法
### 一、K-Means算法
#### 代码实现（自定义）
* [代码实现（自定义）](/K-Means/K-Menas.py)
#### 代码实现（scikit-learn库）
* [代码实现（scikit-learn库）](/K-Means/K-Means_scikit-learn.py)  
- &ensp;&ensp;&ensp;&ensp;定义：K-means算法是很典型的基于距离的聚类算法。从N个文档随机选取K个点作为质心；对剩余的每个点测量其到每个质心的距离，并把它归到最近的质心的类；重新计算已经得到的各个类的质心迭代2～3步直至新的质心与原质心相等或小于指定阈值，算法结束。
- 基本的聚类分析算法：  
（1）K均值:  
基于原型的、划分的距离技术，它试图发现用户指定个数(K)的簇。  
（2）凝聚的层次距离:  
思想:开始时，每个点都作为一个单点簇，然后重复的合并两个最靠近的簇，直到尝试单个、包含所有点的簇。  
（3）DBSCAN:   
一种基于密度的划分距离的算法，簇的个数有算法自动的确定，低密度中的点被视为噪声而忽略，因此其不产生完全聚类。
- 距离量度:不同的距离量度会对距离的结果产生影响，常见的距离量度如下所示：
欧氏距离：定义在两个向量（两个点）上：点xx和点yy的欧氏距离为：  
&ensp;&ensp;&ensp;&ensp;![欧氏距离](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/欧氏距离.png)  
闵可夫斯基距离:Minkowski distance， 两个向量（点）的p阶距离：  
&ensp;&ensp;&ensp;&ensp;![闵可夫斯基距离](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/闵可夫斯基距离.png)  
当p=1p=1时就是曼哈顿距离，当p=2p=2时就是欧氏距离。  
马氏距离:定义在两个向量（两个点）上，这两个点在同一个分布里。点xx和点yy的马氏距离为：  
&ensp;&ensp;&ensp;&ensp;![马氏距离](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/马氏距离.png)  
其中，Σ是这个分布的协方差。  
当Σ=1时，马氏距离退化为欧氏距离。
#### 1、基础Kmeans算法.  
Kmeans算法的属于基础的聚类算法，它的核心思想是：从初始的数据点集合，不断纳入新的点，然后再从新计算集合的“中心”，再以改点为初始点重新纳入新的点到集合，再计算“中心”，依次往复，直到“中心”不再改变，这些集合不再都不能再纳入新的数据为止.
- 图解:  
假如我们在坐标轴中存在如下Ａ,B,C,D,E一共五个点，然后我们初始化（或者更贴切的说指定）两个特征点（意思就是将五个点分成两个类），采用欧式距离计算距离.  
&ensp;&ensp;&ensp;&ensp;![图解](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/图解.png)
- 注意的点:  
１．中心计算方式不固定，常用的有使用距离（欧式距离，马式距离，曼哈顿距离，明考斯距离）的中点，还有重量的质心，还有属性值的均值等等，虽然计算方式不同，但是整体上Kmeans求解的思路相同.  
２．初始化的特征点（选取的Ｋ个特征数据）会对整个收据聚类产生影响.所以为了得到需要的结果，需要预设指定的凸显的特征点，然后再用Kmeans进行聚类.
#### 2、K-Means算法的优化目标  
之前学习线性回归、逻辑回归的时候，都有一个优化目标，就是其代价函数。对于K均值（K-means）算法，同样也有这样的优化目标：那就是使得各个数据点距离聚类中心的距离总和最小，也就是下图中所有红线和蓝线加起来的总长度最小。  
&ensp;&ensp;&ensp;&ensp;![优化目标](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/优化目标.png)  
K均值（K-means）算法的代价函数是:  
&ensp;&ensp;&ensp;&ensp;![代价函数](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/代价函数.png)  
我们的优化目标就是 min J( c(1) , … , c(m) , μ1 , … , μK) 。
- 随机初始化  
对于这样的训练数据，我们是希望把它们分成三类的：  
&ensp;&ensp;&ensp;&ensp;![结果2](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/结果1.png)  
但有时候，因为初始化的选择，我们得到的不是全局最优解，而是局部最优解。得到的结果可能是这样的：  
&ensp;&ensp;&ensp;&ensp;![结果1](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/结果2.png)  
也可能是这样的：  
&ensp;&ensp;&ensp;&ensp;![结果2](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/结果3.png)  
面对这样的问题，我们应该怎么办呢？：多次进行随机初始化，以及运行 k-means 算法，最终选择代价函数最小的一个结果。这个次数，可以在 50 到 1000 次左右。随着运行次数的增加，经常都能找到较好的局部最优解。当然如果聚类中心数量 K 很大的话，得到的结果可能不会好太多。
- 聚类中心数量 K 的选择  
首先，K 应该小于训练样本数量 m。其次，K 的大小，常常是模凌两可的，例如下面这个图，我们既可以认为 K = 4 ，也可以认为 K = 2 ，  
&ensp;&ensp;&ensp;&ensp;![类别1](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/类别1.png)&ensp;&ensp;&ensp;&ensp;![类别2](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/类别2.png)&ensp;&ensp;&ensp;&ensp;![类别3](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/类别3.png)  
有一个方法叫做肘部法则（Elbow Method），可以给我们提供一个参考。例如下图，随着聚类中心个数 K 的增加，其代价函数计算的结果会下降：  
&ensp;&ensp;&ensp;&ensp;![肘部法则1](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/肘部法则1.png)  
我们说图中这个位置（之后的下降都不太明显），就是应该设置的聚类中心个数 K 。但是有时候，我们绘出的图形是这样的：  
&ensp;&ensp;&ensp;&ensp;![肘部法则2](https://github.com/Lg-AiLearn/ML/blob/master/images/kmeans/肘部法则2.png)  
这样的情况就找不到比较明显的肘部。肘部法则值得尝试，但是对其结果不要抱有太大的期待。对于聚类中心个数 K 的选择，更多是人工进行的。这主要看我们运行K均值（K-means）算法来达到什么目的（根据实务和专业知识确定）。
#### 3、K-Means算法优缺点
- K-Means算法优点  
（1）是解决聚类问题的一种经典算法，简单、快速  
（2）对处理大数据集，该算法保持可伸缩性和高效性  
（3）当簇接近高斯分布时，它的效果较好。
- K-Means算法缺点  
（1）在簇的平均值可被定义的情况下才能使用，可能不适用于某些应用；  
（2）在 K-means 算法中 K 是事先给定的，这个 K 值的选定是非常难以估计的。很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适；  
（3）在 K-means 算法中，首先需要根据初始聚类中心来确定一个初始划分，然后对初始划分进行优化。这个初始聚类中心的选择对聚类结果有较大的影响，一旦初始值选择的不好，可能无法得到有效的聚类结果；  
（4）该算法需要不断地进行样本分类调整，不断地计算调整后的新的聚类中心，因此当数据量非常大时，算法的时间开销是非常大的；  
（5）若簇中含有异常点，将导致均值偏离严重（即:对噪声和孤立点数据敏感）；  
（6）不适用于发现非凸形状的簇或者大小差别很大的簇。
- K-means算法缺点的改进  
（1）很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适。通过类的自动合并和分裂，得到较为合理的类型数目 K，例如 ISODATA 算法。  
（2）针对上述(3)，可选用二分K-均值聚类；或者多设置一些不同的初值，对比最后的运算结果，一直到结果趋于稳定结束。  
（3）针对上述第(5)点，改成求点的中位数，这种聚类方式即K-Mediods聚类（K中值）


## 第五部分：关联算法
### 一、Apriori算法
#### 代码实现（自定义）
* [代码实现（自定义）](/Apriori/Apriori.py)

#### 1、前提要点  
&ensp;&ensp;&ensp;&ensp;Apriori关联分析是一种在大规模数据集中寻找关系的算法，这些关系有两种形式频繁项集或关联规则，暗示两个物品间有很强的关系。  
关联分析是在大规模数据集中寻找有趣关系的任务。这些关系可以有两种形式：1)频繁项集，2)关联规则  
&ensp;&ensp;&ensp;&ensp;1.**频繁项集(frequency item sets)** ：经常同时出现的一起的一些元素的集合--利用支持度度量，满足最小支持度阈值的所有项集。  
&ensp;&ensp;&ensp;&ensp;2.**关联规则(association rules)** : 意味着两种元素之间存在很强的关系--利用置信度度量。  
&ensp;&ensp;&ensp;&ensp;下面用一个例子来说明这两种概念：图给出了某个杂货店的交易清单。  
&ensp;&ensp;&ensp;&ensp;![交易清单](https://github.com/Lg-AiLearn/ML/blob/master/images/Apriori/交易清单.png)  
&ensp;&ensp;&ensp;&ensp;我们用支持度和可信度来度量这些有趣的关系。一个项集的**支持度（support）** 被定义数据集中包含该项集的记录所占的比例。如上图中，{豆奶}的支持度为3/5，{豆奶,尿布}的支持度为2/5。支持度是针对项集来说的，因此可以定义一个最小支持度，保留满足最小值尺度的项集。  
&ensp;&ensp;&ensp;&ensp;**可信度** 或**置信度（confidence）** 是针对关联规则来定义的。规则{尿布}➞{啤酒}的可信度被定义为"支持度({尿布,啤酒})/支持度({尿布})"，由于{尿布,啤酒}的支持度为3/5，尿布的支持度为4/5，所以"尿布➞啤酒"的可信度为3/4。这意味着对于包含"尿布"的所有记录，我们的规则对其中75%的记录都适用。
#### 2、Apriori原理  
&ensp;&ensp;&ensp;&ensp;假设我们有一家经营着4种商品（商品0，商品1，商品2和商品3）的杂货店，2图显示了所有商品之间所有的可能组合：  
&ensp;&ensp;&ensp;&ensp;![可能组合](https://github.com/Lg-AiLearn/ML/blob/master/images/Apriori/可能组合.png)  
&ensp;&ensp;&ensp;&ensp;研究人员发现一种所谓的Apriori原理，可以帮助我们减少计算量。**Apriori原理是说如果某个项集是频繁的，那么它的所有子集也是频繁的。更常用的是它的逆否命题，即如果一个项集是非频繁的，那么它的所有超集也是非频繁的**。  
&ensp;&ensp;&ensp;&ensp;![非频繁](https://github.com/Lg-AiLearn/ML/blob/master/images/Apriori/非频繁.png)  
#### 3、使用Apriori算法来发现频繁集  
&ensp;&ensp;&ensp;&ensp;前面提到，关联分析的目标包括两项：发现频繁项集和发现关联规则。首先需要找到频繁项集，然后才能获得关联规则（正如前文所讲，计算关联规则的可信度需要用到频繁项集的支持度）。  
&ensp;&ensp;&ensp;&ensp;Apriori算法是发现频繁项集的一种方法。Apriori算法的两个输入参数分别是最小支持度和数据集。该算法首先会生成所有单个元素的项集列表。接着扫描数据集来查看哪些项集满足最小支持度要求，那些不满足最小支持度的集合会被去掉。然后，对剩下来的集合进行组合以生成包含两个元素的项集。接下来，再重新扫描交易记录，去掉不满足最小支持度的项集。该过程重复进行直到所有项集都被去掉。
- 生成候选项集  
&ensp;&ensp;&ensp;&ensp;数据集扫描的伪代码大致如下：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;对数据集中的每条交易记录tran：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;对每个候选项集can：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;检查can是否是tran的子集  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;如果是，则增加can的计数  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;对每个候选项集：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;如果其支持度不低于最小值，则保留该项集  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;返回所有频繁项集列表
-  完整的Apriori算法-发现频繁集  
&ensp;&ensp;&ensp;&ensp;整个Apriori算法的伪代码如下：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;当集合中项的个数大于0时：  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;构建一个由k个项组成的候选项集的列表（k从1开始）  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;计算候选项集的支持度，删除非频繁项集  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;构建由k+1项组成的候选项集的列表
- 从频繁集中挖掘相关规则  
&ensp;&ensp;&ensp;&ensp;解决了频繁项集问题，下一步就可以解决相关规则问题。  
&ensp;&ensp;&ensp;&ensp;要找到关联规则，我们首先从一个频繁项集开始。从杂货店的例子可以得到，如果有一个频繁项集{豆奶, 莴苣}，那么就可能有一条关联规则“豆奶➞莴苣”。这意味着如果有人购买了豆奶，那么在统计上他会购买莴苣的概率较大。注意这一条反过来并不总是成立，也就是说，可信度(“豆奶➞莴苣”)并不等于可信度(“莴苣➞豆奶”)。  
&ensp;&ensp;&ensp;&ensp;前文也提到过，一条规则P➞H的可信度定义为support(P|H)/support(P)，其中“|”表示P和H的并集。可见可信度的计算是基于项集的支持度的。  
&ensp;&ensp;&ensp;&ensp;图给出了从项集{0,1,2,3}产生的所有关联规则，其中阴影区域给出的是低可信度的规则。可以发现如果{0,1,2}➞{3}是一条低可信度规则，那么所有其他以3作为后件（箭头右部包含3）的规则均为低可信度的。  
&ensp;&ensp;&ensp;&ensp;![所有关联规则](https://github.com/Lg-AiLearn/ML/blob/master/images/Apriori/所有关联规则.png)    
&ensp;&ensp;&ensp;&ensp;可以观察到，如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求。以图为例，假设规则{0,1,2} ➞ {3}并不满足最小可信度要求，那么就知道任何左部为{0,1,2}子集的规则也不会满足最小可信度要求。可以利用关联规则的上述性质属性来减少需要测试的规则数目，类似于Apriori算法求解频繁项集。
#### 4、使用FP-growth算法来高效发现频繁项集  
&ensp;&ensp;&ensp;&ensp; FP-growth算法基于Apriori构建，但采用了高级的数据结构减少扫描次数，大大加快了算法速度。FP-growth算法只需要对数据库进行两次扫描，而Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，因此**FP-growth算法的速度要比Apriori算法快**。  
&ensp;&ensp;&ensp;&ensp;FP-growth算法发现频繁项集的基本过程如下：  
&ensp;&ensp;&ensp;&ensp;（1）构建FP树；（2）从FP树中挖掘频繁项集；（3）FP-growth算法  
&ensp;&ensp;&ensp;&ensp;FP-growth优点：一般要快于Apriori。  
&ensp;&ensp;&ensp;&ensp;FP-growth缺点：实现比较困难，在某些数据集上性能会下降。  
&ensp;&ensp;&ensp;&ensp;FP-growth适用数据类型：离散型数据。











