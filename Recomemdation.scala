// Import the basic recommender libraries from Spark's MLlib package
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.rdd._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Row, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
Logger.getLogger("org").setLevel(Level.ERROR)
Logger.getLogger("akka").setLevel(Level.ERROR)
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.io.{BufferedWriter, FileWriter, File}
import org.apache.spark.{SparkConf, SparkContext}
import scala.language.implicitConversions

///////////////////Importing the textFiles/////////////////

val rawArtistAlias = sc.textFile("/home/users/fosei/audioscrobbler/artist_alias.txt")
val rawUserArtistData = sc.textFile("/home/users/fosei/audioscrobbler/user_artist_data.txt")
val rawArtistData = sc.textFile("/home/users/fosei/audioscrobbler/artist_data.txt")
////////////////////////////////////////////////////////////
val rawArtistAlias = sc.textFile("/home/users/fosei/audioscrobbler/artist_alias.txt")
val rawUserArtistData = sc.textFile("/home/users/fosei/audioscrobbler/user_artist_data.txt")
val rawArtistData = sc.textFile("/home/users/fosei/audioscrobbler/artist_data.txt")

val artistByID = rawArtistData.flatMap { 
  line => val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) { 
    None 
  } else { 
    try { 
      Some((id.toInt, name.trim)) } 
    catch { case e: NumberFormatException => None } 
  } 
} 

val artistAlias = rawArtistAlias.flatMap { line => 
  val tokens = line.split('\t') 
  if (tokens(0).isEmpty) { 
    None 
  } else {
    Some((tokens(0).toInt, tokens(1).toInt)) 
  } 
}.collectAsMap()

val bArtistAlias = sc.broadcast(artistAlias)

val trainData = rawUserArtistData.map { 
  line => val Array(userID, artistID, count) = 
    line.split(' ').map(_.toInt) 
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID) 
    (userID, finalArtistID, count) 
}

////////////problem 3a(i) ///////////////////
val ch = trainData.toDF("user", "artist", "listn_times")
val dfff =  ch.groupBy("user").count().where(col("count")>=100)
val ff = dfff.select("user").rdd.map(r => r(0)).collect()
val dataDF1 = ch.filter($"user".isin(ff:_*))
val dataDF = dataDF1.toDF("user", "artist", "listn_times")
////////////////////////////////////////////



//////////////////Question 3a(ii)///////////////////////////////

val ws1 = Window.partitionBy("user").orderBy("artist")
val ws2 = Window.partitionBy("user")

val dataDF2 = dataDF.withColumn("row_number", row_number.over(ws1)).
	withColumn("count", count("user").over(ws2)).
	withColumn("percent", (col("row_number")/col("count")))

val data_rdd = dataDF2.rdd.keyBy(r => r.getInt(0))
val frac = data_rdd.map(r => r._1).distinct.map(x => (x, 0.9)).collectAsMap()
val trainData100 = data_rdd.sampleByKeyExact(withReplacement = false, frac, 2L).values
////////////////////////////////////////////////
var trainD = spark.createDataFrame(trainData100, dataDF2.schema)
var testD = dataDF2.except(trainD) 
trainD.drop(dataDF2.col("count")).drop(dataDF2.col("percent")).drop(dataDF2.col("row_number"))
testD.drop(dataDF2.col("count")).drop(dataDF2.col("percent")).drop(dataDF2.col("row_number"))
val trainData90 = trainD.rdd.map(row => Rating(row.getInt(0), row.getInt(1), row.getInt(2)))
val testData10 = testD.rdd.map(row => Rating(row.getInt(0), row.getInt(1), row.getInt(2)))

trainData90.cache()
testData10.cache()

val model = ALS.trainImplicit(trainData90, 10, 5, 0.01, 1.0)
/////////////////////////////////////////////////////////////////



//////////////////Question 3a(iii)///////////////////////////////

val someUser = testData10.map(x => x.user).distinct().takeSample(false, 100, 100)  

implicit def bool2int(b:Boolean) = if (b) 1 else 0
val auROCs = someUser.map(r => {
  val recommendations = model.recommendProducts(r, 25)
  val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).filter { case Array(user,_,_) => user.toInt == r }
  val actualArtistsForUser = testData10.filter(s => s.user == r).map(s => s.product).collect().toSet
  val tuples = recommendations.map(r => {
    val recommendedArtistID = r.product
    val result = actualArtistsForUser.contains(recommendedArtistID)
    (r.rating, bool2int(result).toDouble)
  })

  val metrics = new BinaryClassificationMetrics(sc.parallelize(tuples))
  val auROC = metrics.areaUnderROC
  println("Area under ROC = " + auROC)
  (auROC)
})
 
//Average AUC over all 100 selected users
val auROCsList = auROCs.toList
val sumAUC = auROCsList.sum
val avgAUC = sumAUC / 100
println("The Average AUC is: " + avgAUC)



/////Average AUC over 100 previously selected users///////
val artistsTotalCount = trainData90.map(r => (r.product, r.rating)).reduceByKey(_ + _).collect().sortBy(-_._2)
val topArtists = artistsTotalCount.take(25)


val auROCs_pre = someUser.map(r => {
  val recommendations2 = topArtists.map{case (artist, rating) => Rating(r, artist, rating)}

  val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).filter { case Array(user,_,_) => user.toInt == r }
  val actualArtistsForUser = testData10.filter(s => s.user == r).map(s => s.product).collect().toSet
  val tuples = recommendations2.map(r => {
    val recommendedArtistID = r.product
    val result = actualArtistsForUser.contains(recommendedArtistID)
    (r.rating, bool2int(result).toDouble)
  })

  val metrics = new BinaryClassificationMetrics(sc.parallelize(tuples))
  val auROC = metrics.areaUnderROC
  println("Area under ROC = " + auROC)
  (auROC)
})

//Average AUC over all 100 selected users
val auROCsList_pre = auROCs_pre.toList
val sumAUC_pre = auROCsList_pre.sum
val avgAUC_pre = sumAUC_pre / 100
println("The Average AUC is: " + avgAUC_pre)

//////////////////Question 3b///////////////////////////////
val evaluations = for (rank <- Array(10, 25, 100);
lambda <- Array(1.0, 0.01, 0.0001); 
alpha <- Array(1.0, 10.0, 100.0))
yield {
  val model = ALS.trainImplicit(trainData90, rank, 10, lambda, alpha)
  implicit def bool2int(b:Boolean) = if (b) 1 else 0
  val auROCs = someUser.map(r => {
    val recommendations = model.recommendProducts(r, 25)
    val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).filter { case Array(user,_,_) => user.toInt == r }
    val actualArtistsForUser = testData10.filter(s => s.user == r).map(s => s.product).collect().toSet
    val tuples = recommendations.map(r => {
      val recommendedArtistID = r.product
      val result = actualArtistsForUser.contains(recommendedArtistID)
      // rdd with (y^, y)
      (r.rating, bool2int(result).toDouble)
    })
    val metrics = new BinaryClassificationMetrics(sc.parallelize(tuples))
    val auROC = metrics.areaUnderROC
    (auROC)
  })
  val auROCsList = auROCs.toList
  val sumAUC = auROCsList.sum
  val auc = sumAUC / 100.0
  println("(" + rank + ": " + lambda + ": " + alpha + ")  average auc: " + auc)
  ((rank, lambda, alpha), auc)
}

evaluations.sortBy(_._2).reverse.foreach(x =>  println(x + "\n"))


//////////////////Question 3c///////////////////////////////
val someUsers = trainData.map(x => x.user).distinct().take(10)  
val someRecommendations = someUsers.map(userID =>   
    model.recommendProducts(userID, 5)) 
  someRecommendations.map(recs => recs.head.user + " -> " +   
    recs.map(_.product).mkString(", ")).foreach(println)

var AUC1 = 0.0
var AUC2 = 0.0

for (someUser <- someUsers) {
  
  val actualArtists = actualArtistsForUser(someUser)

  val recommendations1 = model.recommendProducts(someUser, 100)
  val predictionsAndLabels1 = recommendations1.map { 
    case Rating(user, artist, rating) =>
      if (actualArtists.contains(artist)) {
        (rating, 1.0)
      } else {
        (rating, 0.0)
      }
  }

  val metrics1 = new BinaryClassificationMetrics(sc.parallelize(predictionsAndLabels1))
  AUC1 += metrics1.areaUnderROC

  val recommendations2 = predictMostPopular(someUser, 25)
  val predictionsAndLabels2 = recommendations2.map { 
    case Rating(user, artist, rating) =>
      if (actualArtists.contains(artist)) {
        (rating, 1.0)
      } else {
        (rating, 0.0)
      }
  }

  val metrics2 = new BinaryClassificationMetrics(sc.parallelize(predictionsAndLabels2))
  AUC2 += metrics2.areaUnderROC
}

println("ALS-Recommender AUC: " + (AUC1/10.0))
println("Most-Popular AUC:    " + (AUC2/10.0))






