# 写在前面
> 理解都来自于《Spark The Definitive Guide》

ML是高级的数据分析，Spark作为一个数据分析的集大成者，自然不会缺席。不过目前看比Python的生态要弱一些。ML一般的分类如下：

- 有监督学习（需要人工标注，**本次重点**）
- 推荐引擎（根据用户行为推荐商品）
- 无监督学习（难以判断效果）
- [Deep Learning](https://www.deeplearningbook.org/)（个人背景知识不足，还不会），
- Featrue engineering（数据特性提取，不属于ML算法的内容，但这活才真需要人工干）
## 有监督学习
有监督的意思就是输入训练的数据都需要人工标注，例如标注一张图片`是否`是色情图片。一般场景是分类模型和回归模型，根据多种特性，来预测一个值。

- 分类（classification），处理离散型数据，如二元分类（0、1），其他元（多个有限的类别）
- 回归（regression），处理连续型数据，机票的价格预测
## Spark训练流程
Spark在ML方面主要优势是集群计算，向`scikit-learn`适用于单机训练。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/452860/1655975524417-c00905f5-bceb-4948-8a13-2d683eb81fd1.png#clientId=u341f9611-1f44-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=612&id=u38e6dc6a&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1224&originWidth=1886&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1122220&status=done&style=none&taskId=u8b327e37-dca7-427a-9fb7-ed4c1361ea6&title=&width=943)
# 本次实践
选取的实例是机票的价格预测，因为价格属于连续型数据，所以选取回归（regression）模型。
## 内容
包含两个数据集，三种训练模型（LinearRegression、RandomForestTrees、GradientBoostedTrees）
### 数据集1，mh
重点是数据的特性提取。取自，[kaggle](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh)，原始内容摘要如下
```scala
Airline,Date_of_Journey,Source,Destination,Route,Dep_Time,Arrival_Time,Duration,Total_Stops,Additional_Info,Price
IndiGo,24/03/2019,Banglore,New Delhi,BLR → DEL,22:20,01:10 22 Mar,2h 50m,non-stop,No info,3897
Air India,1/05/2019,Kolkata,Banglore,CCU → IXR → BBI → BLR,05:50,13:15,7h 25m,2 stops,No info,7662
Jet Airways,9/06/2019,Delhi,Cochin,DEL → LKO → BOM → COK,09:25,04:25 10 Jun,19h,2 stops,No info,13882
```
### 数据集2，easemytrip
重点是模型预测的结果分析。取自，[kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)，原始内容摘要如下
```scala
,airline,flight,source_city,departure_time,stops,arrival_time,destination_city,class,duration,days_left,price
0,SpiceJet,SG-8709,Delhi,Evening,zero,Night,Mumbai,Economy,2.17,1,5953
1,SpiceJet,SG-8157,Delhi,Early_Morning,zero,Morning,Mumbai,Economy,2.33,1,5953
2,AirAsia,I5-764,Delhi,Early_Morning,zero,Early_Morning,Mumbai,Economy,2.17,1,5956
```
## How To Run
### 环境

- Spark，3.2.1
- Java，1.8.0_191
- Scala，2.13.8
### 步骤
```scala
// Step 1, Compile jar
$> cd flight-price
$> sbt package


// Step 2, Submit to Spark
$> SPARK_HOME/spark-3.2.1-bin-hadoop3.2-scala2.13/bin/spark-submit \
--class zhiwin.spark.practice.ml.entry.MainApp \
--master "local[*]"   \
--packages com.typesafe.scala-logging:scala-logging_2.13:3.9.4 \
target/scala-2.13/main-scala-ch24_2.13-1.0.jar [EASE | MH]

// Step 3, just waiting
```
## 实践解析
标准的三个步骤：数据清洗、训练模型、模型预测结果分析。一般而言数据工程师关心第一步，数据科学家关心第二和第三步。
### 特性提取
因为`regression`训练算法的输入只能处理数字型数据，拿到原始数据后不可避免需要做很多转换（注意：没有标准来规定某个特性，需要怎么处理，需要By场景调整）。其实这一步往往是最复杂的，会遇到千奇百怪的数据源，处理各式各样的数据格式，然后喂给模型。这里我们聚焦数据集1（因为数据集2是已经处理好的）。
```scala
root
|-- Airline: string (nullable = true)
|-- Date_of_Journey: date (nullable = true)
|-- Source: string (nullable = true)
|-- Destination: string (nullable = true)
|-- Route: string (nullable = true)
|-- Dep_Time: string (nullable = true)
|-- Arrival_Time: string (nullable = true)
|-- Duration: string (nullable = true)
|-- Total_Stops: string (nullable = true)
|-- Additional_Info: string (nullable = true)
|-- Price: integer (nullable = true)
```
从原始数据来看，11个特性，除了`Price`外（目标特性），没有数值型特性，全部都需要转换。

- 日期类特性：Date_of_Journey、Dep_Time、Arrival_Time、Duration
- 字符串类特性：Airline、Source、Destination、Additional_Info
- 特殊处理的特性：Total_Stops
- 无用特性：Route
#### 无用、无效数据处理
第一步需要先把对训练没有贡献的数据进行处理，最常见的就是null。
```scala
scala> rawDF.filter(isnull($"Route")).show()
+---------+---------------+------+-----------+-----+--------+------------+--------+-----------+---------------+-----+
|  Airline|Date_of_Journey|Source|Destination|Route|Dep_Time|Arrival_Time|Duration|Total_Stops|Additional_Info|Price|
+---------+---------------+------+-----------+-----+--------+------------+--------+-----------+---------------+-----+
|Air India|     2019-05-06| Delhi|     Cochin| null|   09:45|09:25 07 May| 23h 40m|       null|        No info| 7480|
+---------+---------------+------+-----------+-----+--------+------------+--------+-----------+---------------+-----+
```
这里的处理方式就是直接忽略，因为只有一条数据，不影响训练（如果是很多数据都有null，需要额外的策略）。<br />另外，Route这个特性的意义其实和Total_Stops是重复的，可以将其删掉
```scala
rawDF.filter(!isnull($"Route")).drop("Route")
```
#### 日期数据处理
在航班这个场景下，某个时间节点对于预测价格不太合适（值空间太大），这里采用清洗的策略是取月、日、时、分的数值，年这个特性意义不大（不可重复），所以忽略掉。

- Date_of_Journey  -> Journey_Day/Journey_Month
- Dep_Time               -> Departure_Hour/Departure_Minute
- Arrival_Time           -> Arrival_Hour/Arrival_Minute
- Duration                  -> Duration_hours/Duration_minutes

由于Duration的数据格式有点不规则（如你所想，现实数据是很残酷的），使用了UDF来处理：
```scala
spark.udf.register("hhmmUDF", (hhmm: String) => hhmm match {
  case s"${h}h ${m}m"  => (h.toInt, m.toInt)
  case s"${h}h"         => (h.toInt, 0)
  case s"${m}m"         => (0, m.toInt)
  case _               => (0, 0)
})
```
#### 字符串数据处理
这里出现的字符串类型都属于有穷分类的集合（普通文本属于NLP范畴，这里不涉及），Spark针对这类数据有专门的处理方式：One-Hot Encoding，转换0|1的向量空间，避免数据带有大小关系的特性（例如 '红色' > '绿色'，会误导模型）。
#### 特殊处理的特性
Total_Stops表示航班中转了多少次，对预测机票价格是有意义的，原始数据中采用字符串来表示的，转换成一个有大小关系的数字特性。
```scala
scala> rawDF.select("Total_Stops").distinct().show()
+-----------+
|Total_Stops|
+-----------+
|    4 stops|
|   non-stop|
|    2 stops|
|     1 stop|
|    3 stops|
+-----------+
```
从日常经验推理：0中转的可定比中转4次的机票受欢迎，具有大小关系更合理
```scala
spark.udf.register("stops2numUDF", (stops: String) => stops match {
  case "non-stop"  => 0
  case "1 stop"    => 1
  case "2 stops"   => 2
  case "3 stops"   => 3
  case "4 stops"   => 4
})
```
### 算法选择
到了这一步，可以算是进入标准流水线作业的工作了。因为使用的模型训练库都是Spark提供好统一接口的（关心模型训练算法如何实现？还是先学会使用吧），开箱即用。也很方便切换不同的模型，也可以挨个训练看效果。本次实践选取了三个模型：

- LinearRegression
- RandomForestTrees
- GradientBoostedTrees

这里选择模型的标准是：多试几个，对比看看效果。可以多试，主要是因为从代码实现的角度成本很低；但需要等待的时间会比较长，而且费电。
### 结果
根据模型预测的值和实际的值，进行比较来看模型的好坏。本实践是根据历史机票数据，训练模型；训练之前是把整数据集分成两个集合的：训练集、测试集。测试集不能参与训练过程，否则直接给答案的考试，没有意义。
#### 评估指标
这里参考比较常见、易理解的两个指标，具体指标的理论先放一放，只需知道如何判断就行。

- RMSE，值域0到正无穷，越接近0越好
- R的平方（R-squared），值域负无穷到1，越接近1越好
#### 数据集1，mh
```scala
LinearRegression:
MSE = 3.341123914769855E7
RMSE = 5780.245595794227
R-squared = 0.23850286422372546
MAE = 4118.344172583785
Explained variance = 9871303.999823103

GradientBoostedTrees:
MSE = 3.2176045580276933E7
RMSE = 5672.3932850497
R-squared = 0.2666549587797764
MAE = 3907.6932681379812
Explained variance = 1.388243529298386E7

RandomForestRegressor:
MSE = 3.2462901053517725E7
RMSE = 5697.62240355727
R-squared = 0.26011705037449484
MAE = 3905.251135993035
Explained variance = 1.1194450671137968E7
```
从RMSE和R-squared的标准来看，三个模型的预测结果都不好。主要的原因还是数据集，总共只有13354条数据，如前面所述，这个数据集目的是用来观察Feature提取。
#### 数据集2，easemytrip
```scala
GradientBoostedTrees:
MSE = 2.217834088577796E7
RMSE = 4709.388589379513
R-squared = 0.957221585978728
MAE = 2813.8764660294496
Explained variance = 4.897021489636351E8

LinearRegression:
MSE = 3.836675511264046E7
RMSE = 6194.090337784916
R-squared = 0.9259967666962065
MAE = 4277.772269744498
Explained variance = 4.78840610974997E8

RandomForestRegressor:
MSE = 1.2166650487375194E7
RMSE = 3488.072603512604
R-squared = 0.9765325091501861
MAE = 1925.5305063516325
Explained variance = 4.9820222641830015E8
```
根据R-squared这个指标来看，模型的效果很不错了（数据集大约30w条），其中RandomForestRegressor的效果最好，R-squared 达到了0.9765（RMSE值也是相对最小）。<br />来看一下实际预测的数据：
```scala
prediction,label
5105.415274137838,4028.0
5302.9008673630415,4028.0
5502.68678219195,4028.0
4010.5946348205657,4071.0
4045.503947937312,4071.0
4010.5946348205657,4071.0
4490.293321964762,4502.0
4490.293321964762,4502.0
4483.428497772996,4502.0
5105.434487065396,4294.0
5128.192801228469,4456.0
4480.085423175375,4498.0
4476.906022585791,4498.0
4507.057720705781,4500.0
...
```
测试集输出也很大，只取样了1000个预测结果进行可视化，从下图可以看出效果对比。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/452860/1656061857283-3a8f373a-02b2-4a19-9107-af3fcf01bf47.png#clientId=u4bd4d2d8-d8cf-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=627&id=ubbbe0938&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1254&originWidth=1666&originalType=binary&ratio=1&rotation=0&showTitle=false&size=189927&status=done&style=none&taskId=u4dcc6e56-23fb-4f91-ab2d-c482bfc77b3&title=&width=833)<br />下图是实验所使用的硬件情况，使用了两个Mac笔记本作为工作节点，花了19分钟。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/452860/1656057271933-86b16473-3f05-40a1-9be6-6eb01f775d27.png#clientId=u4bd4d2d8-d8cf-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=444&id=u86004744&margin=%5Bobject%20Object%5D&name=image.png&originHeight=888&originWidth=1656&originalType=binary&ratio=1&rotation=0&showTitle=false&size=199763&status=done&style=none&taskId=u671ec9d7-1a72-4606-b8cf-7910452a048&title=&width=828)
# 最后
1，本次实验的源码：[https://github.com/changzhiwin/flight-price](https://github.com/changzhiwin/flight-price)<br />2，Spark: The Definitive Guide，好书，建议直接看英文版，[https://zh.u1lib.org/category-list](https://zh.u1lib.org/category-list)<br />3，Spark 官网：[https://spark.apache.org/](https://spark.apache.org/)<br />4，语雀文档，[《Spark ML 入门实践&理解》](https://www.yuque.com/docs/share/b980be21-fc83-4767-a56f-69f48a48baa3)
