/*
 * A custom hack mlllib version of low rank matrix factorization using RSVD ALS
 * Support implicit + baseline function
 * Support Attribute based Learning 
 * Integrate item features via deep attribute ALS projection
 * Support Nonnegative Least Square
 */


import java.{util => ju}
import java.io.IOException
import java.io.PrintWriter
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Sorting
import scala.util.hashing.byteswap64

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.jblas.DoubleMatrix
import org.netlib.util.intW

import org.apache.spark.{Logging, Partitioner}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.optimization.NNLS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet, SortDataFormat, Sorter}
import org.apache.spark.util.random.XORShiftRandom
import java.util.{Random => JavaRandom}

def double_round(value:Double, precision:Int) : Double = {
        math.round(value * math.pow(10, precision)) / math.pow(10, precision)
}

class NormalEquation(val k: Int) extends Serializable {

    /** Number of entries in the upper triangular part of a k-by-k matrix. */
    val triK = k * (k + 1) / 2
    /** A^T^ * A */
    val ata = new Array[Float](triK)
    /** A^T^ * b */
    val atb = new Array[Float](k)

    //private val da = new Array[Double](k)
    private val upper = "U"

    /*
    private def copyToDouble(a: Array[Float]): Unit = {
      var i = 0
      while (i < k) {
        da(i) = a(i)
        i += 1
      }
    }
    */

    /** Adds an observation. */
    def add(a: Array[Float], b: Float, c: Float = 1.0F): this.type = {
      require(c >= 0.0)
      require(a.length == k)
      //copyToDouble(a)
      blas.sspr(upper, k, c, a, 1, ata)
      if (b != 0.0) {
        blas.saxpy(k, b, a, 1, atb, 1)
      }
      this
    }

    /** Merges another normal equation object. */
    def merge(other: NormalEquation): this.type = {
      require(other.k == k)
      blas.saxpy(ata.length, 1.0F, other.ata, 1, ata, 1)
      blas.saxpy(atb.length, 1.0F, other.atb, 1, atb, 1)
      this
    }

    /** Resets everything to zero, which should be called after each solve. */
    def reset(): Unit = {
      ju.Arrays.fill(ata, 0.0F)
      ju.Arrays.fill(atb, 0.0F)
    }
}

trait LeastSquaresNESolver extends Serializable {
    def solve(ne: NormalEquation, lambda: Float): Array[Float]
}

class CholeskySolver extends LeastSquaresNESolver  {

    private val upper = "U" 

    /** 
     * Solves a least squares problem with L2 regularization:
     *
     *   min norm(A x - b)^2^ + lambda * norm(x)^2^
     *
     * @param ne a [[NormalEquation]] instance that contains AtA, Atb, and n (number of instances)
     * @param lambda regularization constant
     * @return the solution x
     */
    def solve(ne: NormalEquation, lambda: Float): Array[Float] = {
      val k = ne.k
      // Add scaled lambda to the diagonals of AtA.
      var i = 0
      var j = 2 
      while (i < ne.triK) {
        ne.ata(i) += lambda
        i += j
        j += 1
      }
      val info = new intW(0)
      lapack.sppsv(upper, k, 1, ne.ata, ne.atb, k, info)
      val code = info.`val`
      assert(code == 0, s"lapack.dppsv returned $code.")
      val x = new Array[Float](k)
      i = 0
      while (i < k) {
        x(i) = ne.atb(i).toFloat
        i += 1
      }
      ne.reset()
      x
    }
}


class NNLSSolver extends LeastSquaresNESolver {
    private var rank: Int = -1
    //private var workspace: NNLS.Workspace = _
    private var ata: Array[Float] = _
    private var initialized: Boolean = false

    var scratch:Array[Float] = _
    var grad:Array[Float] = _
    var x:Array[Float] = _
    var dir:Array[Float] = _
    var lastDir:Array[Float] = _
    var res:Array[Float] = _

    private def initialize(rank: Int): Unit = {
        this.rank = rank
        //workspace = NNLS.createWorkspace(rank)
        ata = new Array[Float](rank * rank)
        initialized = true

        scratch = new Array[Float](rank)
        grad = new Array[Float](rank)
        x = new Array[Float](rank)
        dir = new Array[Float](rank)
        lastDir = new Array[Float](rank)
        res = new Array[Float](rank)
        require(this.rank == rank)
    }

    def wipe(): Unit = {
      ju.Arrays.fill(scratch, 0.0f)
      ju.Arrays.fill(grad, 0.0f)
      ju.Arrays.fill(x, 0.0f)
      ju.Arrays.fill(dir, 0.0f)
      ju.Arrays.fill(lastDir, 0.0f)
      ju.Arrays.fill(res, 0.0f)
    }

    /**
     * Solves a nonnegative least squares problem with L2 regularization:
     *
     *   min_x_  norm(A x - b)^2^ + lambda * n * norm(x)^2^
     *   subject to x >= 0
     */
    def solve(ne: NormalEquation, lambda: Float): Array[Float] = {
      val rank = ne.k
      initialize(rank)
      fillAtA(ne.ata, lambda)
      val xt = solve0(ata, ne.atb)
      ne.reset()
      xt.map(x => x.toFloat)
    }

    /**
     * Given a triangular matrix in the order of fillXtX above, compute the full symmetric square
     * matrix that it represents, storing it into destMatrix.
     */
    private def fillAtA(triAtA: Array[Float], lambda: Float) {
      var i = 0
      var pos = 0
      var a = 0.0f
      while (i < rank) {
        var j = 0
        while (j <= i) {
          a = triAtA(pos)
          ata(i * rank + j) = a
          ata(j * rank + i) = a
          pos += 1
          j += 1
        }
        ata(i * rank + i) += lambda
        i += 1
      }
    }

  def solve0(ata: Array[Float], atb: Array[Float]): Array[Float] = {    
    val n = atb.length
    //val scratch = ws.scratch

    // find the optimal unconstrained step
    def steplen(dir: Array[Float], res: Array[Float]): Double = {
      val top = blas.sdot(n, dir, 1, res, 1)
      blas.sgemv("N", n, n, 1.0f, ata, n, dir, 1, 0.0f, scratch, 1)
      // Push the denominator upward very slightly to avoid infinities and silliness
      top / (blas.sdot(n, scratch, 1, dir, 1) + 1e-20)
    }

    // stopping condition
    def stop(step: Double, ndir: Double, nx: Double): Boolean = {
        ((step.isNaN) // NaN
      || (step < 1e-7) // too small or negative
      || (step > 1e40) // too small; almost certainly numerical problems
      || (ndir < 1e-12 * nx) // gradient relatively too small
      || (ndir < 1e-32) // gradient absolutely too small; numerical issues may lurk
      )
    }

    //val grad = ws.grad
    //val x = ws.x
    //val dir = ws.dir
    //val lastDir = ws.lastDir
    //val res = ws.res
    val iterMax = math.max(400, 20 * n)
    var lastNorm = 0.0f
    var iterno = 0
    var lastWall = 0 // Last iteration when we hit a bound constraint.
    var i = 0
    while (iterno < iterMax) {
      // find the residual
      blas.sgemv("N", n, n, 1.0f, ata, n, x, 1, 0.0f, res, 1)
      blas.saxpy(n, -1.0f, atb, 1, res, 1)
      blas.scopy(n, res, 1, grad, 1)

      // project the gradient
      i = 0
      while (i < n) {
        if (grad(i) > 0.0 && x(i) == 0.0) {
          grad(i) = 0.0f
        }
        i = i + 1
      }
      val ngrad = blas.sdot(n, grad, 1, grad, 1)

      blas.scopy(n, grad, 1, dir, 1)

      // use a CG direction under certain conditions
      var step = steplen(grad, res)
      var ndir = 0.0f
      val nx = blas.sdot(n, x, 1, x, 1)
      if (iterno > lastWall + 1) {
        val alpha = ngrad / lastNorm
        blas.saxpy(n, alpha, lastDir, 1, dir, 1)
        val dstep = steplen(dir, res)
        ndir = blas.sdot(n, dir, 1, dir, 1)
        if (stop(dstep, ndir, nx)) {
          // reject the CG step if it could lead to premature termination
          blas.scopy(n, grad, 1, dir, 1)
          ndir = blas.sdot(n, dir, 1, dir, 1)
        } else {
          step = dstep
        }
      } else {
        ndir = blas.sdot(n, dir, 1, dir, 1)
      }

      // terminate?
      if (stop(step, ndir, nx)) {
        return x.clone
      }

      // don't run through the walls
      i = 0
      while (i < n) {
        if (step * dir(i) > x(i)) {
          step = x(i) / dir(i)
        }
        i = i + 1
      }

      // take the step
      i = 0
      while (i < n) {
        if (step * dir(i) > x(i) * (1 - 1e-14)) {
          x(i) = 0
          lastWall = iterno
        } else {
          x(i) -= (step * dir(i)).toFloat
        }
        i = i + 1
      }

      iterno = iterno + 1
      blas.scopy(n, dir, 1, lastDir, 1)
      lastNorm = ngrad
    }
    x.clone
  }

}

object Recommender extends Serializable{

def als_train(nnls:Boolean, rootfolder:String, silo:String, rank:Int, iter:Int, l2reg:Double=0.01, alpha:Int=40, baseline:Boolean=false, implicitpref:Boolean=false, computeYtY:Boolean=false, attrpref:Boolean=false) : (Array[Array[Float]], Array[Array[Float]], Array[Array[Float]]) = {

 //val nnls = true
 //val rootfolder = "/nmhc40/testmws2/"
 //val silo = "1"
 //val rank = 20
 //val iter = 2
 //val l2reg = 0.01
 //val alpha = 40
 //val baseline = false
 //val implicitpref = true
 //val computeYtY = true
 //val attrpref = false

 println("loading file ...")
 val lines = sc.textFile("hdfs://nameservice1/" + rootfolder + "/scoreset_" + silo + ".csv", rank)
 val raw_ratings = lines.map(_.split(',') match { case Array(user, item, rate) => (user.toInt, item.toInt, rate.toDouble) })
 val u_ratings = lines.map(_.split(',') match { case Array(user, item, rate) => (user.toInt, (item.toInt, rate.toFloat)) }).groupByKey.map( t => (t._1, t._2.toArray)).sortByKey(true)
 val i_ratings = lines.map(_.split(',') match { case Array(user, item, rate) => (item.toInt, (user.toInt, rate.toFloat)) }).groupByKey.map( t => (t._1, t._2.toArray)).sortByKey(true)

 var attrs:Array[Array[Float]] = null
 var attrBV:org.apache.spark.broadcast.Broadcast[Array[Array[Float]]] = null

 if(attrpref){
   println("loading attrs file ...")
   val attr_lines = sc.textFile("hdfs://nameservice1/" + rootfolder + "/productft_" + silo + ".csv", rank)
   attrs = attr_lines.map(_.split(',').map(_.toFloat)).collect
   attrBV = sc.broadcast(attrs)
 }

 val u_count = u_ratings.count.toInt
 val i_count = i_ratings.count.toInt
 val random = new JavaRandom()

 println("init random factors ...")
 var ufactor:Array[Array[Float]] = Array.fill(u_count, rank)(random.nextGaussian().toFloat)
 ufactor.map{ factor =>
  val nrm = blas.snrm2(rank, factor, 1)
  blas.sscal(rank, 1.0f / nrm, factor, 1)
 }
 
 if(baseline) ufactor = ufactor.map{ t => t(0) = 1; t}
 var YtY:NormalEquation = new NormalEquation(rank)
 var ifactor:Array[Array[Float]] = null
 var attrfactor:Array[Array[Float]] = null
 //var factor:Array[Array[Float]] = null 
 var factor:RDD[Array[Float]] = null

 for(loop <- 1 to iter) {
  println(">>>>> iter=" + loop.toString)

  if(implicitpref & computeYtY){
   println("YtY ufactor")
   YtY.reset()
   ufactor.foreach ( t => YtY.add(t, 0.0f) )
  }

 println("processing ifactor ...")
 ifactor = i_ratings.map{ item =>
   val ratings = item._2
   val ls = new NormalEquation(rank)
   if(implicitpref & computeYtY) ls.merge(YtY)
   for(i <- 0 to ratings.size-1){
    val rating = ratings(i)
    if(implicitpref){
      val c1 = alpha * math.abs(rating._2).toFloat
      if(computeYtY) ls.add(ufactor(rating._1-1), c1 + 1.0f, c1)
      else ls.add(ufactor(rating._1-1), c1 + 1.0f, c1 + 1.0f)
    } else {
      ls.add(ufactor(rating._1-1), rating._2)
    }
   }

  val solver = if(nnls) new NNLSSolver() else new CholeskySolver()
  val factor = solver.solve(ls, (ratings.size*l2reg).toFloat) // ALS-WR
  if(baseline) factor(0) = 1
  factor
 }.collect

 // perform attribute solver
 println("perform attribute solver ...")
 println("attrs.size*regl2")
 if(attrpref){
  val ifactorT = ifactor.transpose
  val ifactorRDD = sc.parallelize(ifactorT)
  attrfactor = ifactorRDD.map{ ifactorT =>
   val attrValue = attrBV.value
   val ls = new NormalEquation(attrValue(0).length)
   for(i <- 0 to attrValue.length - 1){
    ls.add(attrValue(i), ifactorT(i))
   }

   val solver = if(nnls) new NNLSSolver() else new CholeskySolver()
   val factor = solver.solve(ls, (attrs.size*l2reg).toFloat)
   factor
  }.collect

  // recalculate ifactor based on attribute latent dim
  ifactor = attrs.map(i_attr => attrfactor.map(attr_f => blas.sdot(attr_f.size, i_attr, 1, attr_f, 1) ))
 }

 if(implicitpref & computeYtY){
  println("YtY ifactor")
  YtY.reset()
  ifactor.foreach ( t => YtY.add(t, 0.0f) )
 }

 println("processing ufactor ...")
 ufactor = u_ratings.map{ user =>
  val ratings = user._2
  val ls = new NormalEquation(rank)
  if(implicitpref & computeYtY) ls.merge(YtY)
  for(i <- 0 to ratings.size-1){
    val rating = ratings(i)
    if(implicitpref){
      val c1 = alpha * math.abs(rating._2).toFloat
      if(computeYtY) ls.add(ifactor(rating._1-1), c1 + 1.0f, c1)
      else ls.add(ifactor(rating._1-1), c1 + 1.0f, c1 + 1.0f)
    } else {
      ls.add(ifactor(rating._1-1), rating._2)
    }
  }

  val solver = if(nnls) new NNLSSolver() else new CholeskySolver()
  val factor = solver.solve(ls, (ratings.size*l2reg).toFloat) // ALS-WR  
  if(baseline) factor(0) = 1
  factor
 }.collect

 println("***** validating preds *****")
 val preds = raw_ratings.map{ rating =>
  val uid = rating._1
  val iid = rating._2
  val score = rating._3
  if(baseline){
    val pred = double_round(blas.ddot(rank-1, ufactor(uid-1).map(_.toDouble).takeRight(rank-1), 1, ifactor(iid-1).map(_.toDouble).takeRight(rank-1), 1), 3)
    (uid, iid, score, pred, score - pred)
  } else {
    val pred = double_round(blas.ddot(rank, ufactor(uid-1).map(_.toDouble), 1, ifactor(iid-1).map(_.toDouble), 1), 3)
    (uid, iid, score, pred, score - pred)
  }
 }

 val rmse = math.sqrt(preds.map(x => x._5 * x._5).mean())
 val sse = math.sqrt(preds.map(x => x._5 * x._5).sum())
 println("[CI] ALS RMSE:" + rmse.toString + " SSE:" + sse.toString)

} // iter

 // return factors
 if(attrpref){
  (ufactor, ifactor, attrfactor)
 }else{
  (ufactor, ifactor, null)
 }
} // als_train


def print_factors(factors:Array[Array[Float]], filename:String, silo:String) = {
 var pw:PrintWriter = new PrintWriter(filename + silo + ".dat")
 factors.map{t =>
  t.map{x =>
   pw.print(x)
   pw.print(" ")
  }
  pw.println()
 }

 pw.close()
}

def output_factors(ufactors:Array[Array[Float]], ifactors:Array[Array[Float]], filename:String, silo:String) = {
 var pw:PrintWriter = new PrintWriter(filename + "_u" + silo + ".dat")
 ufactors.map{t => 
  t.map{x => 
   pw.print(x)
   pw.print(" ")
  }
  pw.println()
 }

 pw.close()

 pw = new PrintWriter(filename + "_i" + silo + ".dat")
 ifactors.map{t =>
  t.map{x => 
   pw.print(x)
   pw.print(" ")
  }
  pw.println()
 }
 
 pw.close()

}

def output_factors_rdd(u_mat:RDD[(Int, Array[Double])], i_mat:RDD[(Int, Array[Double])], filename:String, silo:String) = {
 val ufactors = u_mat.sortByKey(true).map(_._2.map(_.toFloat)).collect
 val ifactors = i_mat.sortByKey(true).map(_._2.map(_.toFloat)).collect
 output_factors(ufactors, ifactors, filename, silo)
}

def als_train_attr(nnls:Boolean, factor:Array[Array[Float]], rootfolder:String, silo:String, l2reg:Double=0.01) : ( Array[Array[Float]], Array[Array[Float]] ) = {

 var attrs:Array[Array[Float]] = null
 var attrBV:org.apache.spark.broadcast.Broadcast[Array[Array[Float]]] = null

 println("loading attrs file ...")
 val attr_lines = sc.textFile("hdfs://nameservice1/" + rootfolder + "/productft_" + silo + ".csv", factor(0).size)
 attrs = attr_lines.map(_.split(',').map(_.toFloat)).collect
 attrBV = sc.broadcast(attrs)
 var attrfactor:Array[Array[Float]] = null

 println("perform attribute solver ...")
 val ifactorT = factor.transpose
 val ifactorRDD = sc.parallelize(ifactorT)
 attrfactor = ifactorRDD.map{ ifactorT =>
  val attrValue = attrBV.value
  val ls = new NormalEquation(attrValue(0).length)
  for(i <- 0 to attrValue.length - 1){
   ls.add(attrValue(i), ifactorT(i))
  }

  val solver = if(nnls) new NNLSSolver() else new CholeskySolver()
  val factor = solver.solve(ls, (attrs.size*l2reg).toFloat)
  factor
 }.collect

  // recalculate ifactor based on attribute latent dim
 val ifactor = attrs.map(i_attr => attrfactor.map(attr_f => blas.sdot(attr_f.size, i_attr, 1, attr_f, 1) ))

(ifactor, attrfactor)

}

}
