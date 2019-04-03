# spark scala yarn cluster example: work directory: ScalaSpark
cd ./ScalaSpark

# before assembly: change two things in the ScalaSpark folder: 1. source code: SPARK_MASTER -> "yarn"; 2. build.sbt spark package to "provided"
# if assembly successfully, the fat-jar will be stored at ./target/scala-2.11/scala-spark-ccm-assembly-0.1.jar
sbt assembly


# put local file into hdfs
hadoop fs -put ~/cloud/CCM-Parralization/TestInputCSVData/test_float_1000.csv

# change [paths|inputs] to the hdfs directory like: /user/bo/test_float_1000.csv
spark-submit --master yarn ./target/scala-2.11/scala-spark-ccm-assembly-0.1.jar ~/cloud/CCM-Parralization/ccm-scala.cfg
