package zhiwin.spark.practice.ml.exercise

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ TrainValidationSplit, ParamGridBuilder }
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.evaluation.RegressionMetrics

object RandomForestTreesExercise extends RegressionExercise {

  val name = "random-forest-trees"

  def trainModel(df: DataFrame, cateCols: Array[String], doubleCols: Array[String], testDF: DataFrame, outDir: String = ""): Unit = {

    val stages = earlyEncodeStates(cateCols, doubleCols)
    val rf = new RandomForestRegressor()
    stages += rf
    val pipeline = new Pipeline().setStages(stages.toArray)

    val params = new ParamGridBuilder().
      addGrid(rf.maxDepth, Array(10, 15)).
      addGrid(rf.numTrees, Array(10, 15)).
      build()

    val evaluator = new RegressionEvaluator()
      //.setMetricName("r2")
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(params)
      .setTrainRatio(0.8)

    val model = tvs.fit(df)
    //model.write.overwrite().save("/tmp/modelLocationRandomForestRegressor")

    val outDF = model.transform(testDF).
      select("prediction", "label")

    if (outDir.length > 5){
      outDF.write.format("csv").
        option("header", true).
        save(s"${outDir}/${name}")
    }

    val out = outDF.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))  

    val rm = new RegressionMetrics(out)
    println("-----------------------------------------------")
    println("RandomForestRegressor:")
    println(s"MSE = ${rm.meanSquaredError}")
    println(s"RMSE = ${rm.rootMeanSquaredError}")
    println(s"R-squared = ${rm.r2}")
    println(s"MAE = ${rm.meanAbsoluteError}")
    println(s"Explained variance = ${rm.explainedVariance}")
    
  }
}