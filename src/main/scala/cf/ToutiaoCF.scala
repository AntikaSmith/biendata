package cf

/**
  * Created by ZhangHan on 2016/9/14.
  */

import java.io.{BufferedReader, InputStreamReader}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import util.ParseUtil

object ToutiaoCF {


  case class Rating(questionId: Int, userId: Int, answer : Float)
  case class Prediction(questionId: Int, userId: Int, answer : Float, prediction: Float)



  def pasreRating(str: String): Rating = {
    import ParseUtil.{questionIdMap, userIdMap}

    val fields = str.split("\t")

    assert(fields.size == 3 && fields(0) != "qid")
    Rating(questionIdMap(fields(0)), userIdMap(fields(1)), fields(2).toFloat)
  }

  def convertProb(x: Row) = {
//    val prediction = x(3).asInstanceOf[Float]
//    if (prediction.isNaN){
//      //println(s"unkown num:$x")
//      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], 0)
//    }
//    else
      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], math.min(x(3).asInstanceOf[Float], 1))
  }

  def main(args: Array[String]) = {
    val spark = ParseUtil.spark

    import spark.implicits._

    def constructData(fileName: String) = spark.read.textFile(fileName).map(pasreRating(_)).toDF()

    val training = constructData("src/main/resources/invited_info_train.txt")

    val validating = constructData("src/main/resources/validate_nolabel.txt")

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(200)
      .setUserCol("userId")
      .setItemCol("questionId")
      .setRatingCol("answer")
      .setNonnegative(true)

    val model = als.fit(training)

    val predictions: Dataset[Prediction] = model.transform(validating).map(convertProb)
    predictions
      // place all data in a single partition
      .coalesce(1)
      .write.format("com.databricks.spark.csv").mode(SaveMode.Overwrite)
      .option("header", "true")
      .save("data")
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("answer")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
    // $example off$

    spark.stop()
  }
}
