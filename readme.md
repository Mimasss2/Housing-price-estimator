# 房屋价格预测

hitsz大数据导论课程项目

本项目旨在通过房屋交易市场交易记录，挖掘数据间潜在的关联关系，设计高效、解释性强、鲁棒的算法对房屋交易价格进行预测。

### 可视化工具

- sweetviz: Python开源库，快速可视化目标值和比较数据集。
- shap: SHapley Additive exPlanation，解释模型输出。

### 基本内容

#### 数据分析

房屋交易市场交易记录中，包括数值型属性，文字属性，时间属性等。其中数值型属性包括连续值、离散值、缺失值等。文字型属性包括级别、地址、街区等。

先使用pandas、matplotlib、seaborn，sweetviz等库函数可视化数据，挖掘数据之间的潜在的关系，并去除相关性较小的特征。

数据清洗前，各特征之间的相关性较弱，相关系数比较低，如下图所示。其中，出售价格是我们的预测目标，和此特征最相关的特征为总平方英尺，但其相关系数仅为**0.43**

#### 数据预处理

对选取的特征中的值进行预处理：填充缺失值，将文字属性数值化，识别并处理异常值等。

用众数填充总单元、居住单元、邮编、修建年份、公寓号中的缺失值。按居住单元分组，用均值填充总平方英尺中的缺失值。用0填充出售价格中的缺失值。

去除训练集中出售价格为0的数据。根据图像显示的数据间的关系，以及数据间的相关系数，发现商业单元、土地平方英尺、地役权、出售日期这四个特征与目标出售价格之间的相关性很低，遂将其删除。用均值填充所有含nan的数值列。

对所有特征进行归一化，并使用主成分分析方法，分析每个特征对总体的贡献。

经过多轮缺失值填充，归一化等操作，数据清洗后特征之间的相关关系如下图所示。地役权和所属区域、建筑类型、税收级别关系较强，其他指标间的相关性也略有提升。但出售价格和这些特征之间的相关性还是不太强，这是受到了原始数据复杂性的约束所致。

#### 模型选择

首先在普通的LinearRegression线性回归模型上进行预测，再引入带惩罚项函数的模型，如Ridge,Lasso等，进一步采用RandomForest，比较不同模型的回归结果，选择最优模型。

#### 模型验证

随机选取原训练集的20%作为验证集，剩下的80%作为训练集。使用训练集对模型进行训练，预测验证集的结果，与实际结果比较，计算损失(均方误差)，评判模型的效果。

采用shap包可视化各特征对最终结果的影响。其中，总平方英尺和土地平方英尺这两个指标贡献最大。这个结果可以比较直观的理解为，房屋的出售价格和房屋的面积之间的正相关关系较强。在区域相同的情况下，此现象与生活中的场景较为吻合。

### 结果分析

初始时，将所有特征按照常识进行预处理，所得预测结果较差。进一步分析发现处理后的数据与最终结果之间的相关性均较小，最大不超过0.5，而普遍小于0.1。此后利用图像分析各个特征之间的相关性，发现总平方英寸这一特征与目标之间的相关性最高，于是先从此特征入手。利用此单一特征进行回归预测，在测试集上的均方误差为7749025.804。

此后，加入其他特征，并分组填充缺失值，提高数据质量，使用主成分分析等方法分析特征的重要性，并选取几个比较显著的特征，在测试集上的均方误差为7224550.11

使用主成分分析对结果进行分析，不同特征对结果均有一定贡献，第一主成分的贡献不到30%。累计解释占比随着特征数的增加而升高，趋势从陡峭趋向平缓。


由于此数据集较为复杂，特征之间的相关性较低，难以在任务上得到比较理想的结果，预测价格与实际价格的相关系数最高不超过50%。如果需要更为准测的预测，可能需要更多高质量的相关性较高的数据。