
# How to run
```
/Users/changzhi/Spark/spark-3.2.1-bin-hadoop3.2-scala2.13/bin/spark-submit \
--class zhiwin.spark.practice.ml.entry.MainApp \
--master "local[*]"   \
--packages com.typesafe.scala-logging:scala-logging_2.13:3.9.4 \
target/scala-2.13/main-scala-ch24_2.13-1.0.jar
```

# DataSet source
```

easemytrip:
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

mh:
https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh
```

# Data
```
-----------------------------------------------
LinearRegression:
MSE = 3.2286587421273742E7
RMSE = 5682.128775491959
R-squared = 0.24542978331312348
MAE = 4090.722103774412
Explained variance = 1.1655098011449557E7
-----------------------------------------------

-----------------------------------------------
GradientBoostedTrees:
MSE = 3.1204036829702444E7
RMSE = 5586.057360044063
R-squared = 0.27073008599913073
MAE = 3878.7305198150593
Explained variance = 1.3352882926898304E7
-----------------------------------------------

-----------------------------------------------
RandomForestRegressor:
MSE = 3.1337899431295894E7
RMSE = 5598.026387156093
R-squared = 0.2676015815532232
MAE = 3971.3318518508418
Explained variance = 9353933.325238349
-----------------------------------------------


-----------------------------------------------
GradientBoostedTrees:
MSE = 2.221802826057642E7
RMSE = 4713.600350112048
R-squared = 0.9570282159561934
MAE = 2785.8476640787544
Explained variance = 4.9095901242378914E8
-----------------------------------------------

```