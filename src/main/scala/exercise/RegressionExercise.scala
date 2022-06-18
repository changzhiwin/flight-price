package zhiwin.spark.practice.ml.exercise

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{ StringIndexer, OneHotEncoder }

import scala.collection.mutable.ArrayBuffer

trait RegressionExercise {

  def trainModel(df: DataFrame, cateCols: Array[String], doubleCols: Array[String]): Unit

  def metrics(df: DataFrame, predictFile: String = ""): Unit

  final def earlyEncodeStates(cateCols: Array[String], doubleCols: Array[String]): ArrayBuffer[PipelineStage] = {

    val stages = new ArrayBuffer[PipelineStage]()

    val cates = cateCols //Array("Airline", "Source", "Destination", "Additional_Info")

    val sis = cates.map(c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}Ind").setHandleInvalid("skip"))
    stages ++= sis

    val ohes = cates.map(c => new OneHotEncoder().setInputCol(s"${c}Ind").setOutputCol(s"${c}Vec").setDropLast(false))
    stages ++= ohes

    val featrueCols = doubleCols ++ cateCols.map(c => s"${c}Vec")
    val assembler = new VectorAssembler().
      setInputCols(featrueCols).
      /*
      // "Journey_WeekDay",
      setInputCols(Array("Total_Stops", "Journey_Day", "Journey_Month", "Departure_Hour", 
        "Departure_Minute", "Arrival_Hour", "Arrival_Minute", "Duration_hours", "Duration_minutes",
        "AirlineVec", "SourceVec", "DestinationVec", "Additional_InfoVec")).
      */
      setOutputCol("features").
      setHandleInvalid("skip")
    stages += assembler
  }
}