package zhiwin.spark.practice.ml.entry

import org.apache.spark.sql.{ DataFrame, SparkSession }

import org.apache.spark.sql.functions.{ col }
import zhiwin.spark.practice.ml.basic.Basic
import zhiwin.spark.practice.ml.features.ExtractFeatureMH
import zhiwin.spark.practice.ml.exercise.{ RegressionExercise, LinearRegressionExercise, GradientBoostedTreesExercise, RandomForestTreesExercise }

object MainApp extends Basic {
  def main(args: Array[String]) = {

    val lib = args(0)
    val spark = getSession(s"Flight Price Model - ${lib}")

    lib match {
      case "EASE"   => doEaseMyTripLib(spark)
      case _        => doMhLib(spark)
    }

    spark.close()
  }

  def doMhLib(spark: SparkSession): Unit = {

    val featrueDF =  ExtractFeatureMH.getFeatures(spark, "data/mh/DataSet.csv")
    featrueDF.cache()

    val Array(train, test) = featrueDF.randomSplit(Array(0.90, 0.10), 42)
    train.sample(0.01).show(2)
    test.sample(0.1).show(2)

    val cateCols = Array("Airline", "Source", "Destination", "Additional_Info")
    val doubleCols = Array("Total_Stops", "Journey_Day", "Journey_Month", "Departure_Hour", 
      "Departure_Minute", "Arrival_Hour", "Arrival_Minute", "Duration_hours", "Duration_minutes")

    val exercise: Seq[RegressionExercise] = Seq(LinearRegressionExercise, RandomForestTreesExercise, GradientBoostedTreesExercise)
    exercise.foreach(e => {
      e.trainModel(train, cateCols, doubleCols, test)
    })
  }

  def doEaseMyTripLib(spark: SparkSession): Unit = {
    // is clean, no need extract
    val featrueDF = spark.read.format("csv").
      option("header", true).
      option("inferSchema", true).
      load("data/easemytrip/Clean_Dataset.csv").
      withColumn("label", col("price").cast("double")).drop("Price")
    featrueDF.cache()

    val Array(train, test) = featrueDF.randomSplit(Array(0.80, 0.20), 42)
    train.sample(0.01).show(2)
    test.sample(0.1).show(2)

    val cateCols = Array("airline", "flight", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class")
    val doubleCols = Array("duration", "days_left")

    val exercise: Seq[RegressionExercise] = Seq(/*GradientBoostedTreesExercise, LinearRegressionExercise, */RandomForestTreesExercise)
    exercise.foreach(e => {
      e.trainModel(train, cateCols, doubleCols, test, "/tmp/flight-price")
    })
  }
}