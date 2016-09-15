import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}
import util.ParseUtil

/**
  * Created by ZhangHan on 2016/9/14.
  */
class UtilTest extends FlatSpec with BeforeAndAfterAll with Matchers{
  "doc2vec" should  "be ok" in {
    import ParseUtil.spark.implicits._
    println(ParseUtil.userTagVec.last)
  }

  "convert train" should "be ok" in {
    ParseUtil.convertInput("invited_info_train.txt")
  }

  "convert validate" should "be ok" in {
    ParseUtil.convertInput("validate_nolabel.txt")
  }
}
