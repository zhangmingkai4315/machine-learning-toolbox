### K-NN算法

K近邻算法是一种基本分类方法，给定训练数据集合，对于输入的数据查找与训练数据集合中最接近的K个实例，如果这K个实例多数属于某一个实例则分类该数据。

#### 1. K值确认
K-NN算法中比较关键的是如何确定K的值大小，k太小容易导致噪声干扰（过拟合),k值过大又会导致判断的过于简单。因此具体使用中需要人为的调整参数获得最佳的结果。

#### 2. 维度距离
关于维度中距离的计算，一般设计采用p=2的欧式距离来计算彼此之间的邻近关系。

![维度计算公式](https://pic4.zhimg.com/80/v2-60bb382b0d22ec0ce296ed0e024f31bc_hd.jpg "维度计算公式")

#### 3. 维度归一化

维度归一化完成所有度量标准的影响是均衡的，不会导致单独属性的数据对于其他属性的影响。

```python
def autoNorm(dataSet):
  minVals = dataSet.min(0)
  maxVals = dataSet.max(0)
  ranges = maxVals - minVals
  normDataset = zeros(shape(dataSet))
  m = dataSet.shape[0]
  normDataset = dataSet - tile(minVals, (m, 1))
  normDataset = normDataset * 1.0 / tile(ranges, (m, 1))
  return normDataset, ranges, minVals
```

#### 4. 手写识别实例

实例来自于《Machine Learning In Action》