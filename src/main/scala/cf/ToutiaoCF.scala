package cf

/**
  * Created by ZhangHan on 2016/9/14.
  */

import java.io.{BufferedReader, File, InputStreamReader, PrintWriter}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql._
import _root_.util.ParseUtil

object ToutiaoCF {


  case class Rating(questionId: Int, userId: Int, answer : Float)
  case class Prediction(questionId: Int, userId: Int, answer : Float, prediction: Float)
  case class Result(qid: String, uid: String, label: Float)

  val defualtValueLossMap1 = scala.collection.mutable.Map[Float, Double](0.01f -> 1)
  val defualtValueLossMap2 = scala.collection.mutable.Map[Float, Double](0.01f -> 1)

  def pasreRating(str: String): Rating = {
    import ParseUtil.{qid2numberIdMap, uid2numberIdMap}

    val fields = str.split("\t")

    assert(fields.size == 3 && fields(0) != "qid")
    Rating(qid2numberIdMap(fields(0)), uid2numberIdMap(fields(1)), fields(2).toFloat)
  }

  def convertProb(x: Row)(defaultValue: Float) = {
    val prediction = x(3).asInstanceOf[Float]
    if (prediction.isNaN){
      //println(s"unkown num:$x")
      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], defaultValue)
    }
    else
      Prediction(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int], x(2).asInstanceOf[Float], math.min(x(3).asInstanceOf[Float], 1))
  }

  def modelAndTest(training: DataFrame, test: DataFrame, sparkSession: SparkSession, userStr: String, questionStr: String, map: scala.collection.mutable.Map[Float, Double]) = {
    import sparkSession.implicits._

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(200)
      .setUserCol(userStr)
      .setItemCol(questionStr)
      .setRatingCol("answer")
      .setNonnegative(true)

    val model = als.fit(training)
    var loss = 1.0
    var tmp:Dataset[Prediction] = null
    for (defaultValue <- map.keys){
      val predictions: Dataset[Prediction] = model.transform(test).map(convertProb(_)(defaultValue))
      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("answer")
        .setPredictionCol("prediction")
      val rmse = evaluator.evaluate(predictions)
      map(defaultValue) = rmse
      if (rmse < loss) {
        tmp = predictions
        loss = rmse
      }
    }

    tmp
  }

  def constructData(fileName: String, spark: SparkSession) = {
    import spark.implicits._
    spark.read.textFile(fileName).map(pasreRating(_)).toDF()
  }

  def main(args: Array[String]) = {
    val spark: SparkSession = ParseUtil.spark

    import spark.implicits._

    val training: DataFrame = constructData("src/main/resources/invited_info_train.txt", spark)
    val validating = constructData("src/main/resources/validate_nolabel.txt", spark)

//    val ratings = spark.read.textFile("src/main/resources/invited_info_train.txt").map(pasreRating(_)).toDF()
//    val Array(training, validating) = ratings.randomSplit(Array(0.9, 0.1))



    val predictionBaseUser = modelAndTest(training, validating, spark, "userId", "questionId", defualtValueLossMap1)
    val predictionBaseQuestion = modelAndTest(training, validating, spark, "questionId", "userId", defualtValueLossMap2)

    predictionBaseQuestion.createOrReplaceTempView("Question")
    predictionBaseUser.createOrReplaceTempView("User")
    val predictions = spark.sql("select Question.questionId, Question.userId, Question.prediction+User.prediction " +
      "from Question join User where Question.questionId=User.QuestionId and Question.userId=User.UserId").map(row =>
      Prediction(row.getInt(0), row.getInt(1), 0.0f, row.getFloat(2))
    )

    val result = predictions.map(prediction =>
      Result(ParseUtil.numberId2qidMap(prediction.questionId), ParseUtil.numberId2uidMap(prediction.userId), prediction.prediction)
    )

    result
      // place all data in a single partition
      .coalesce(1)
      .write.format("com.databricks.spark.csv").mode(SaveMode.Append)
      .option("header", "true")
      .save("data")
//    val nn_output = scala.io.Source.fromFile("nn_output.txt").mkString.split("\n").map(_.toFloat)
//    val cf_output = scala.io.Source.fromFile("data/final.csv").mkString.split("\n").map(_.split(",")).zip(nn_output).map{
//      case (arr, nn) if arr(0) != "qid"=>
//        Array(arr(0), arr(1), (arr(2).toFloat * 0.8 + nn * 0.2).toFloat).mkString(",")
//      case x =>
//        x._1.mkString(",")
//    }.mkString("", "\n", "\n")
//
//    val finalFile = new File("final.csv")
//    val writer = new PrintWriter(finalFile)
//    writer.write(cf_output)
//    writer.close()



    spark.stop()
  }
}
