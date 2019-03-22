name := "ScalaSpark"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++={
  val sparkVer = "2.2.0"
  Seq(
    //% "provided" ony when assembly
    /*
    "org.apache.spark" %% "spark-core" % sparkVer % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVer % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
    */
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-sql" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer
  )
}

libraryDependencies += "au.com.bytecode" % "opencsv" % "2.4"
