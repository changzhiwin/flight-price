package zhiwin.spark.practice.ml.exercise

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ TrainValidationSplit, ParamGridBuilder }
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.tuning.TrainValidationSplitModel

object GradientBoostedTreesExercise extends RegressionExercise {

  val name = "gradient-boosted-trees"

  def trainModel(df: DataFrame, cateCols: Array[String], doubleCols: Array[String], testDF: DataFrame, outDir: String = ""): Unit = {

    val stages = earlyEncodeStates(cateCols, doubleCols)
    val gbt = new GBTRegressor()
    stages += gbt
    val pipeline = new Pipeline().setStages(stages.toArray)

    val params = new ParamGridBuilder().
      addGrid(gbt.maxIter, Array(10, 20)).
      build()

    val evaluator = new RegressionEvaluator()
      //.setMetricName("r2")
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(params)
      .setTrainRatio(0.7)

    val model = tvs.fit(df)
    //model.write.overwrite().save("/tmp/modelLocationGradientBoostedTrees")

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
    println("GradientBoostedTrees:")
    println(s"MSE = ${rm.meanSquaredError}")
    println(s"RMSE = ${rm.rootMeanSquaredError}")
    println(s"R-squared = ${rm.r2}")
    println(s"MAE = ${rm.meanAbsoluteError}")
    println(s"Explained variance = ${rm.explainedVariance}")
  }

  // Can load also, not used in this case
  def metrics(df: DataFrame, predictFile: String = ""): Unit = {

    val model = TrainValidationSplitModel.load("/tmp/modelLocationGradientBoostedTrees")

    val outDF = model.transform(df).
      select("prediction", "label")

    outDF.show(5, false)
  }
}