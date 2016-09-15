package util

/**
  * Created by ZhangHan on 2016/9/14.
  */

import java.io.PrintWriter

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.ScalaReflection.Schema
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

import scala.reflect.runtime.universe.TypeTag

object ParseUtil {

  val spark = SparkSession.builder().appName("toutiaocf").config("spark.sql.warehouse.dir", "file:///C/workspace/biendata/warehouse").getOrCreate()


  def readFile(path: String) = {
    val stream = getClass.getClassLoader.getResourceAsStream(path)
    println(stream)
    scala.io.Source.fromInputStream(stream).getLines()
  }

  def userLines = readFile("user_info.txt").toList

  def questionLines = readFile("question_info.txt").toList

  def extractMap(file: String) = readFile(file).map(_.split("\t").apply(0)).zipWithIndex.toMap

  lazy val questionIdMap = extractMap("question_info.txt")
  lazy val userIdMap = extractMap("user_info.txt")

  def doc2vec(docs: Seq[Row], tpe: StructType, vectorSize:Int = 100) = {
    import spark.implicits._
    val docRdd = spark.sparkContext.parallelize(docs)
    val df: DataFrame = spark.createDataFrame(docRdd, tpe)
    val word2vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(vectorSize)
      .setMinCount(0)

    val model = word2vec.fit(df)
    model.transform(df)
  }

  def getWordVecs(path: String, wordsIndex: Int, vectorSize:Int = 100) = doc2vec(readFile(path).toSeq.map(_.split('\t')).map { question =>
    Row(question(0), question(wordsIndex).split("/"))
  }, StructType(Seq(StructField("id", StringType),StructField("text", ArrayType(StringType, true)))), vectorSize)

  lazy val questionDesc = getWordVecs("question_info.txt", 2)
  lazy val userDesc = getWordVecs("user_info.txt", 2)
  lazy val userTag = getWordVecs("user_info.txt", 1, 10)

  def df2map(df: DataFrame) = df.collect().map(x =>
    x.get(0).asInstanceOf[String] -> x.get(2).asInstanceOf[SparkVector]
  ).toMap

  lazy val questionVecMap = df2map(questionDesc)

  lazy val userVecMap = df2map(userDesc)

  lazy val userTagVec = df2map(userTag)

  lazy val userMap = userLines.map(x => x.split("\t").head -> x.split("\t")).toMap
  lazy val questionMap = questionLines.map(x => x.split("\t").head -> x.split("\t")).toMap

  def computeMeanSigma(arr: Array[Int]) = {
    val sum = arr.sum
    val mean = sum.toDouble / arr.size
    val sigma = arr.map(_ - mean).map(math.pow(_,2)).sum / arr.size
    (mean, math.sqrt(sigma))
  }

  def normalize(arr: Array[Int]) = {
    val (agreeMean, agreeSigma) = (972.1509573810995,  5016.309790789912)
    val (answerMean, answerSigma) =  (40.78060531192094,190.6041421861907)
    val (boutiqueMean, boutiqueSigma) = (9.621371216800494,29.48870975159823)
    Array(arr(0) - agreeMean/ agreeSigma, arr(1) - answerMean / answerSigma, arr(2) - boutiqueMean/ boutiqueSigma)
  }

  def convertInput(path:String) = {
    val outputName = path.split("\\.").head + "outpu" +  ".txt"
    val file = new PrintWriter(outputName)
    for (line <- readFile(path)){
      val Array(qid, uid, label) = line.split("\t")
      if (qid.size == 32){
        file.println(List(questionMap(qid)(1),
          questionVecMap(qid).toArray.map(_.toFloat).mkString(","),
          normalize(questionMap(qid).takeRight(3).map(_.toInt)).mkString(","),
          userTagVec(uid).toArray.map(_.toFloat).mkString(","),
          userVecMap(uid).toArray.map(_.toFloat).mkString(",")
        ).mkString(",") + "\t" + label
        )
      }
    }
    file.flush()
    file.close()
  }
}
