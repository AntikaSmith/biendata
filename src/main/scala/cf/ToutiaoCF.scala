package cf

/**
  * Created by ZhangHan on 2016/9/14.
  */

import java.io.{BufferedReader, InputStreamReader}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Row, SparkSession}
import util.ParseUtil

object ToutiaoCF {


  case class Rating(questionId: Int, userId: Int, answer : Float)
  case class Prediction(questionId: Int, userId: Int, answer : Float, prediction: Float)



  def pasreRating(str: String): Rating = {
    import ParseUtil.{questionIdMap, userIdMap}

    val fields = str.split("\t")
    assert(fields.size == 3 && !fields(2).toFloat.isNaN)
    Rating(questionIdMap(fields(0)), userIdMap(fields(1)), fields(2).toFloat)
  }

  def convertProb(x: Row) = {
    val prediction = x(3).asInstanceOf[Float]
    if (prediction.isNaN){
      println(s"unkown num:$x")
      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], 0)
    }
    else Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], math.min(x(3).asInstanceOf[Float], 1))
  }

  def main(args: Array[String]) = {
    val spark = ParseUtil.spark

    import spark.implicits._

    val ratings = spark.read.textFile("src/main/resources/invited_info_train.txt").map(pasreRating(_)).toDF()

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(200)
      .setUserCol("userId")
      .setItemCol("questionId")
      .setRatingCol("answer")
      .setNonnegative(true)

    val model = als.fit(training)

    val predictions = model.transform(test).map(convertProb)
    predictions.show()
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
