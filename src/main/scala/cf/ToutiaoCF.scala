package cf

/**
  * Created by ZhangHan on 2016/9/14.
  */

import java.io.{BufferedReader, InputStreamReader}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Row, SparkSession}

object ToutiaoCF {


  case class Rating(questionId: Int, userId: Int, answer : Float)
  case class Prediction(questionId: Int, userId: Int, answer : Float, prediction: Float)

  def getStringFromFile(filePath: String) = {
    val in = getClass.getClassLoader.getResourceAsStream(filePath)
    val inReader = new InputStreamReader(in, "utf-8")
    val bf = new BufferedReader(inReader)
    val sb = new StringBuilder()
    var line = ""
    do {
      line = bf.readLine()
      if (line != null) {
        if (sb.length != 0) {
          sb.append("\n")
        }
        sb.append(line)
      }
    } while (line != null)

    in.close()
    inReader.close()
    bf.close()

    sb.toString()
  }

  def extractMap(file: String) = getStringFromFile(file).split("\n").map(_.split("\t").apply(0)).zipWithIndex.toMap

  val questionIdMap = extractMap("question_info.txt")
  val userIdMap = extractMap("user_info.txt")

  def pasreRating(str: String): Rating = {
    val fields = str.split("\t")
    assert(fields.size == 3 && !fields(2).toFloat.isNaN)
    Rating(questionIdMap(fields(0)), userIdMap(fields(1)), fields(2).toFloat)
  }

  def convertProb(x: Row) = {
    val prediction = x(3).asInstanceOf[Float]
    if (prediction.isNaN){
      println(s"unkown num:$x")
      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], 1)
    }
    else Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], math.min(x(3).asInstanceOf[Float], 1))
  }

  def main(args: Array[String]) = {
    val spark = SparkSession.builder().appName("toutiaocf").config("spark.sql.warehouse.dir", "file:///C/workspace/biendata/warehouse").getOrCreate()

    import spark.implicits._

    val ratings = spark.read.textFile("src/main/resources/invited_info_train.txt").map(pasreRating(_)).toDF()

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(10)
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
