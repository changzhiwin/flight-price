package zhiwin.spark.practice.ml.exercise

import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.tuning.CrossValidatorModel

object LinearRegressionExercise extends RegressionExercise {

  val name = "linear-regression"

  def trainModel(df: DataFrame, cateCols: Array[String], doubleCols: Array[String], testDF: DataFrame, outDir: String = ""): Unit = {

    val stages = earlyEncodeStates(cateCols, doubleCols)
    val lr = new LinearRegression()
    stages += lr
    val pipeline = new Pipeline().setStages(stages.toArray)

    val params = new ParamGridBuilder().
      addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).
      addGrid(lr.fitIntercept).
      addGrid(lr.regParam, Array(0.1, 0.05, 0.01)).
      build()

    val evaluator = new RegressionEvaluator()
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val cv = new CrossValidator().
      setEstimator(pipeline).
      setEvaluator(evaluator).
      setEstimatorParamMaps(params).
      setNumFolds(3)

    val model = cv.fit(df)
    //model.write.overwrite().save("/tmp/modelLocationLinearRegression")

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
    println("LinearRegression:")
    println(s"MSE = ${rm.meanSquaredError}")
    println(s"RMSE = ${rm.rootMeanSquaredError}")
    println(s"R-squared = ${rm.r2}")
    println(s"MAE = ${rm.meanAbsoluteError}")
    println(s"Explained variance = ${rm.explainedVariance}")
  }

}