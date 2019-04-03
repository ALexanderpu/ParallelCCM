
// data structures here
case class Sample(indices:Array[Int], L: Int, tau: Int, E: Int)

// useful ccm functions here
object Utils{

  val epsilon = 1E-30

  def mean(arr: Array[Double]):Double = arr.sum / arr.length

  def euclideanDistance(x:Array[Double], y:Array[Double]):Double = {
     Math.sqrt((x, y).zipped.map((a, b) => Math.pow(a-b, 2)).sum)
  }

  def sortNeighbors(vecFocus: Array[Double], space: Array[Tuple2[Int, Array[Double]]]):Array[Tuple2[Int, Double]] = {
     space.map(point => (point._1, euclideanDistance(point._2, vecFocus))).sortBy(_._2).dropWhile(_._2 < epsilon)
  }

  def pearsonCoeff(xs: Array[Double], ys: Array[Double]):Double = {
    // not ranked in the rEDM packages
    val n = xs.length
    val xsSum = xs.sum
    val ysSum = ys.sum

    val numerator = n*(xs, ys).zipped.map(_ * _).sum - xsSum * ysSum

    val denominator = Math.sqrt((n * xs.map(x => x*x).sum - xsSum * xsSum) * (n * ys.map(y => y*y).sum - ysSum * ysSum))
    if(denominator != 0)
      numerator / denominator
    else
      0.0
  }

  def shadowManifold(arr:Array[Double], E:Int, tau:Int):Array[Tuple2[Int, Array[Double]]] = {
    val firstIndex = (E-1)*tau
    val indicesLaggedVector = (firstIndex until arr.length).toArray

    indicesLaggedVector.map(index => (index, Array.tabulate[Double](E)((i:Int) => arr(index - i*tau))))
  }


  def ccmCore(sample:Sample, table:Array[Tuple2[Int, Array[Tuple2[Int, Double]]]], X:Array[Double], Y:Array[Double]):Double={

    val whichPred = ((sample.E - 1)*sample.tau until X.length).toArray
    val indices = sample.indices.toSet
    pearsonCoeff(whichPred.map(index => Y(index)), whichPred.map(index => {
      // find nearest neighbor on sample.indices
      val (indexOfNN, distanceOfNN) = table.find(_._1 == index).get._2.filter(x => indices.contains(x._1)).take(sample.E+1).unzip

      val distToNearest = Math.max(distanceOfNN(0), epsilon)
      val weights = distanceOfNN.map(dist => Math.exp(-dist/distToNearest))
      val weightsTotal = weights.sum
      val normalWeights = weights.map(_/weightsTotal)
      (indexOfNN.map(t => Y(t)), normalWeights).zipped.map(_*_).sum
    }))
  }
}

