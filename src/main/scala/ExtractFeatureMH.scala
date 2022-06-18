package zhiwin.spark.practice.ml.features

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{ DataFrame, SparkSession }

import zhiwin.spark.practice.ml.basic.Basic

object ExtractFeatureMH extends Basic {

  def getFeatures(spark: SparkSession, dataset: String): DataFrame = {

    spark.udf.register("hhmmUDF", (hhmm: String) => hhmm match {
      case s"${h}h ${m}m"  => (h.toInt, m.toInt)
      case s"${h}h"         => (h.toInt, 0)
      case s"${m}m"         => (0, m.toInt)
      case _               => (0, 0)
    })

    spark.udf.register("stops2numUDF", (stops: String) => stops match {
      case "non-stop"  => 0
      case "1 stop"    => 1
      case "2 stops"   => 2
      case "3 stops"   => 3
      case "4 stops"   => 4
    })

    val schema = """
      Airline STRING,Date_of_Journey DATE,
      Source STRING,Destination STRING,Route STRING,
      Dep_Time STRING,Arrival_Time STRING,Duration STRING,
      Total_Stops STRING,Additional_Info STRING,Price INT
    """

    val rawDF = spark.read.format("csv").
      schema(schema).
      option("header", true).
      option("dateFormat", "d/MM/yyyy").
      load(dataset)

    import spark.implicits._
    val noNullDF = rawDF.filter(!isnull($"Route"))

    val formatTimeDF = noNullDF.
      //withColumn("Journey_WeekDay", dayofweek($"Date_of_Journey")).
      withColumn("Journey_Day", dayofmonth($"Date_of_Journey")).
      withColumn("Journey_Month", month($"Date_of_Journey")).
      drop("Date_of_Journey").

      withColumn("Dep_Time_Arr", split($"Dep_Time", ":")).
      withColumn("Departure_Hour", $"Dep_Time_Arr".getItem(0).cast("int")).
      withColumn("Departure_Minute", $"Dep_Time_Arr".getItem(1).cast("int")).
      drop("Dep_Time_Arr", "Dep_Time").

      withColumn("Arrival_Hour_Arr_F", split($"Arrival_Time", " ")).
      withColumn("Arrival_Hour_Arr", split($"Arrival_Hour_Arr_F".getItem(0), ":")).
      withColumn("Arrival_Hour", $"Arrival_Hour_Arr".getItem(0).cast("int")).
      withColumn("Arrival_Minute", $"Arrival_Hour_Arr".getItem(1).cast("int")).
      drop("Arrival_Hour_Arr", "Arrival_Hour_Arr_F", "Arrival_Time").

      withColumn("HhMm_Tup", call_udf("hhmmUDF", $"Duration")).
      withColumn("Duration_hours", $"HhMm_Tup._1").
      withColumn("Duration_minutes", $"HhMm_Tup._2").
      drop("HhMm_Tup", "Duration").

      withColumn("Additional_Info", regexp_replace($"Additional_Info", "No info", "No Info"))

    val finalDF = formatTimeDF.withColumn("Total_Stops", call_udf("stops2numUDF", $"Total_Stops")).
      drop("Route").
      withColumn("label", col("Price").cast("double")).drop("Price")

    finalDF
  }
}