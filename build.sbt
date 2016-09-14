name := "biendata"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= {
  val SparkVersion = "2.0.0"
  Seq(
    "org.apache.spark" %% "spark-core" % SparkVersion,
    "org.apache.spark" %% "spark-mllib" % SparkVersion
  )
}