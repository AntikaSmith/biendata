package cf

/**
  * Created by ZhangHan on 2016/9/14.
  */

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object ToutiaoCF {


  case class Rating(questionId: String, userId: String, answer_? : Float)

  def pasreRating(str: String): Rating = {
    val fields = str.split("\t")
    assert(fields.size == 3)
    Rating(fields(0), fields(1), fields(2).toFloat)
  }

  def main(args: Array[String]) = {
    val spark = SparkSession.builder().appName("toutiaocf").getOrCreate()

    import spark.implicits._

    val ratings = spark.read.textFile("invited_info_train.txt").map(pasreRating(_)).toDF()

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("questionId")
      .setRatingCol("answer_?")

    val model = als.fit(training)

    val predictions = model.transform(test)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
    // $example off$

    spark.stop()
  }
}
