package util

/**
  * Created by ZhangHan on 2016/9/14.
  */

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{Row, SparkSession}
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

  def userLines = readFile("user_info.txt")

  def questionLines = readFile("question_info.txt")

  def extractMap(file: String) = readFile(file).map(_.split("\t").apply(0)).zipWithIndex.toMap

  val questionIdMap = extractMap("question_info.txt")
  val userIdMap = extractMap("user_info.txt")

  def doc2vec(docs: Seq[Row], tpe: StructType) = {
    import spark.implicits._
    val docRdd = spark.sparkContext.parallelize(docs)
    val df = spark.createDataFrame(docRdd, tpe)
    val word2vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(100)
      .setMinCount(0)

    val model = word2vec.fit(df)
    model.transform(df)
  }

  def getWordVecs(path: String) = doc2vec(readFile(path).toSeq.map(_.split('\t')).map { question =>
    Row(question(0), question(2).split("/"))
  }, StructType(Seq(StructField("id", StringType),StructField("text", ArrayType(StringType, true)))))

  lazy val questionDesc = getWordVecs("question_info.txt")
  lazy val userDesc = getWordVecs("user_info.txt")
}
