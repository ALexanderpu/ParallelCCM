import org.scalactic.TolerantNumerics
import org.scalatest._

class CCMTests extends FlatSpec with Matchers{

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-11)
  /*
  "stdDev" should "correct calculate array of length 2" in {
    assert(CCM.stdDev(Array(1,1)) === 0.0)
    assert(CCM.stdDev(Array(1,2)) === 0.5)
    assert(CCM.stdDev(Array(1000.9,1)) === 499.95)
  }

  it should "correct calculate array of length 20 " in {
    assert(CCM.stdDev(Array(1,2,3,4,5,6,7,8,9,10,13,2,38,23,38,23,21,36,76,12)) === 18.003541318307)
    assert(CCM.stdDev(Array(10,2,38,23,38,23,21,36,76,12,10,2,38,23,38,23,21,36,76,12)) === 19.806312125179)
  }

  "mean" should "correct calculate array of length 2" in {
    assert(CCM.mean(Array(1,1)) === 1.0)
    assert(CCM.mean(Array(1,2)) === 1.5)
    assert(CCM.mean(Array(1000.9,1)) === 500.95)
  }

  it should "correct calculate array of length 20 " in {
    assert(CCM.mean(Array(1,2,3,4,5,6,7,8,9,10,13,2,38,23,38,23,21,36,76,12)) === 16.85)
    assert(CCM.mean(Array(10,2,38,23,38,23,21,36,76,12,10,2,38,23,38,23,21,36,76,12)) === 27.9)
  }
  */

}
