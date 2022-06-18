package zhiwin.spark.practice.ml.basic

import org.apache.spark.sql.SparkSession

trait Basic {
  def getSession(appName: String): SparkSession = {
    SparkSession.builder().
      appName(appName).
      config("spark.executor.memory", "6g").
      getOrCreate()
  }
}