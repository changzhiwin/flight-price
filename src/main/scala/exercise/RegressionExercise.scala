package zhiwin.spark.practice.ml.exercise

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{ StringIndexer, OneHotEncoder }

import scala.collection.mutable.ArrayBuffer

trait RegressionExercise {

  def trainModel(df: DataFrame, cateCols: Array[String], doubleCols: Array[String], testDF: DataFrame, outDir: String = ""): Unit

  def name: String

  final def earlyEncodeStates(cateCols: Array[String], doubleCols: Array[String]): ArrayBuffer[PipelineStage] = {

    val stages = new ArrayBuffer[PipelineStage]()

    val cates = cateCols

    val sis = cates.map(c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}Ind").setHandleInvalid("skip"))
    stages ++= sis

    val ohes = cates.map(c => new OneHotEncoder().setInputCol(s"${c}Ind").setOutputCol(s"${c}Vec").setDropLast(false))
    stages ++= ohes

    val featrueCols = doubleCols ++ cateCols.map(c => s"${c}Vec")
    val assembler = new VectorAssembler().
      setInputCols(featrueCols).
      setOutputCol("features").
      setHandleInvalid("skip")
    stages += assembler
  }
}