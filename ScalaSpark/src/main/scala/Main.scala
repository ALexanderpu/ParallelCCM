/**
  * @author Bo Pu
  * achieve all pipelines in CCM
  *
  */

import org.apache.spark.FutureAction
import org.apache.spark.sql.DataFrame
import java.io.File
import scala.collection.JavaConversions._
import scala.io.Source
import scala.collection.mutable.Set
import scala.collection.mutable.Map
import java.util.StringTokenizer

import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.log4j.{LogManager, Level}

import scala.util.Random.nextInt
import scala.collection.mutable.ArrayBuffer
import Utils.{ccmCore, shadowManifold, sortNeighbors}

// write csv file
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.mutable.ListBuffer
import java.io.{BufferedWriter, FileWriter}


object Main {

  //val SPARK_MASTER = "local[*]"
  val SPARK_MASTER = "yarn"

  def isEnvSection (text :String):Boolean = {
    if(text.startsWith("[") && text.endsWith("]")) {
      true
    }else{
      false
    }
  }

  def keyValParse(text:String):Array[String] = {
    text.replaceAll("\\s", "").split("=")
  }

  def sectionParse(text:String):String = {
    text.replaceAll("\\s", "").slice(1, text.length-1)
  }

  // for each l in Ls return samples rhos
  def ccm(x:Array[Double], y:Array[Double], e:Int, tau:Int, Ls:Array[Int], samples:Int):Array[Tuple2[Int, Array[Double]]] = {

    // for local run
    val spark = SparkSession.builder().master(SPARK_MASTER).appName("CCM Job").getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    // 1. build Mx and sort accordingly
    val Mx = shadowManifold(x, e, tau)
    val n = Mx.length
    val MxRDD = spark.sparkContext.parallelize(Mx) // query points q
    val MxBroadcast = spark.sparkContext.broadcast(Mx) // reference points R
    val sortedDistanceTable = MxRDD.map(t=>(t._1, sortNeighbors(t._2, MxBroadcast.value))).collect()

    // 2. broadcast global info
    val sortedDistanceTableBroadcast = spark.sparkContext.broadcast(sortedDistanceTable)
    val XBroadcast = spark.sparkContext.broadcast(x)
    val YBroadcast = spark.sparkContext.broadcast(y)

    // 3. (window size l do samples)
    val results = ArrayBuffer[FutureAction[Seq[Double]]]()
    var l = 0
    for(l <- Ls){
      // build realizations rdd with replacement
      val Realizations = (1 to samples toArray).map(_ => {
        // draw l indices for countSamples times
        val indices = (1 to l toArray).map(_ =>{ (e-1)*tau + nextInt(n - 1)})
        Sample(indices, l, tau, e)
      })
      val RealizationsRDD = spark.sparkContext.parallelize(Realizations)
      // for each sample
      results += RealizationsRDD.map(sample => ccmCore(sample, sortedDistanceTableBroadcast.value, XBroadcast.value, YBroadcast.value)).map(i => Math.max(0.0, i)).collectAsync()
    }

    var ind = 0
    // collect result
    val ans = ArrayBuffer[Tuple2[Int, Array[Double]]]()
    for(ind <- Ls.indices){
      ans += Tuple2(Ls(ind), results(ind).get().toArray)
    }

    ans.toArray
  }

  //main to accept anything after the .jar line as an argument.
  // e.g. spark-submit --class "SimpleApp" --master localtarget/scala-2.10/simpleapp_2.10-1.0.jar "/Users/username/Spark/README.md"
  def main(args:Array[String]): Unit ={
    // check config file path
    if(args.length == 0){
      println("Config file path is required.")
      System.exit(1)
    }
    // setting spark logger level  to show info time and results
    val logger = LogManager.getLogger("scalaspark")
    logger.setLevel(Level.WARN)

    // step 1: parse config file given the config path passed by arguments
    println("Parsing config file.")
    val configFile = args(0)
    val cache = Set[String]()
    var curEnv = ""
    var key = ""
    var value = ""
    var config = Map[String, String]()
    for(line <- Source.fromFile(configFile).getLines){
      if(!line.isEmpty()){
        if(!isEnvSection(line)){
          val kv = keyValParse(line)
          key = kv(0)
          value = kv(1)
          config += (curEnv + "-" + key -> value)
          // println(key + " " + value)
        }else{
          curEnv = sectionParse(line)
          // println(curEnv)
        }
      }
    }

    println("Read inputs and parameters.")
    // step 2: read inputs and parameters and introduce spark environment
    val spark = SparkSession.builder().master(SPARK_MASTER).appName("Reading csv input for ccm").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    var begin_time = System.nanoTime()
    // 1. read csv file here to get x, y and generate lag-vectors
    val input = spark.read.format("csv").option("header", "true").option("inferschema", "true").load(config("paths-input"))
    // 2. read parameters
    val num_samples = config("parameters-num_samples").toInt
    val l1 = config("parameters-LStart").toInt
    val l2 = config("parameters-LEnd").toInt
    val ldelta = config("parameters-LInterval").toInt

    val ls = Array.range(l1, l2+1, ldelta)
    val taus = config("parameters-tau").split(",").map(_.toInt)
    val Es = config("parameters-E").split(",").map(_.toInt)
    val IsWriteFile = config("options-GenerateOutputCSV").toInt

    val xsName = config("inputs-x")
    val ysName = config("inputs-y")

    val xsRaw = input.select(xsName).collect().map(_(0).asInstanceOf[Double])
    val ysRaw = input.select(ysName).collect().map(_(0).asInstanceOf[Double])
    //logger.warn(s"**** read inputs xs: $xsName ****")
    //logger.warn(s"**** read inputs ys: $ysName ****")
    var end_time = System.nanoTime()
    var duration = (end_time - begin_time) / 1e9d
    logger.warn(s"**** read inputs and parameters time: $duration second ****")

    spark.stop()

    println("Execute CCM tasks")
    // step 3: ccm main part call ccm functions
    begin_time = System.nanoTime()
    var e = 0
    var tau = 0
    for(e <- Es){
      for(tau <- taus){
        // run task with e and tau and collect data
        val result = ccm(xsRaw, ysRaw, e, tau, ls, num_samples)

        if(IsWriteFile == 1){
          // time-consuming 80% time on writing? try spark write but need to convert to dataframe
          val outputpath = config("paths-output")
          val outputFile = new BufferedWriter(new FileWriter(outputpath + "/e_" + e.toString + "_tau_" + tau.toString + "_scalaspark.csv"))
          val csvWriter = new CSVWriter(outputFile)
          val csvFields = Array("E", "tau", "L", "rho")
          val totalRecords = result.length * num_samples
          val Ecol = List.fill(totalRecords)(e)
          val taucol = List.fill(totalRecords)(tau)
          val lcol = result.unzip._1.flatMap(List.fill(num_samples)(_))
          val rhocol = result.unzip._2.flatten
          var listOfRecords = new ListBuffer[Array[String]]()
          listOfRecords += csvFields
          var i = 0
          for(i <- rhocol.indices){
            listOfRecords += Array(Ecol(i).toString, taucol(i).toString, lcol(i).toString, rhocol(i).toString)
          }
          // have to import scala.collection.JavaConversions_
          csvWriter.writeAll(listOfRecords.toList)
          outputFile.close()
        }
      }
    }
    end_time = System.nanoTime()
    duration = (end_time - begin_time) / 1e9d
    logger.warn(s"**** ccm tasks time: $duration second ****")
  }
}
