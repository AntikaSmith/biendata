import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}
import util.ParseUtil

/**
  * Created by ZhangHan on 2016/9/14.
  */
class UtilTest extends FlatSpec with BeforeAndAfterAll with Matchers{
  "doc2vec" should  "be ok" in {
    import ParseUtil.spark.implicits._
    val vecDF = ParseUtil.doc2vec(ParseUtil.readFile("user_info.txt").toSeq.map(_.split("\t").apply(2)).map(_.split("/")))
//    val vecDF = ParseUtil.spark.createDataFrame(Seq(
//      "Hi I heard about Spark".split(" "),
//      "I wish Java could use case classes".split(" "),
//      "Logistic regression models are neat".split(" ")
//    ).map(Tuple1.apply)).toDF("text")
    vecDF.show()
  }
}
